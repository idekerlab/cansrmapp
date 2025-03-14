#!/usr/bin/env python
# coding: utf-8
if __name__ == '__main__'  :
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--lambda_gene',default=3.0,action='store')
    parser.add_argument('--lambda_system',default=3.0,action='store')
    parser.add_argument('--lambda_gb',default=1.5,action='store')
    parser.add_argument('--alpha_partition',default=2.5,action='store')
    parser.add_argument('--indir',required=True,action='store')
    parser.add_argument('--outdir',required=True,action='store')
    parser.add_argument('--npats',action='store',default=498.0)
    parser.add_argument('--resample',action='store',default=0)
    parser.add_argument('--cheat',action='store_true',default=False)
    parser.add_argument('--normalize_synteny',action='store_true',default=False)
    ns=parser.parse_args()


import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
opj=os.path.join

import builtins
print(hasattr(builtins,'__IPYTHON__'))
import torch
import cansrmapp
import cansrmapp.utils as cmu
from cansrmapp import DEVICE,np
from cansrmapp.utils import _tcast,mask_sparse_columns,msg,time
from functools import partial
from torch.distributions import Geometric,Exponential,Gamma,Binomial,Dirichlet
import pickle

EPS=torch.finfo(torch.float32).eps
LEPS=torch.log(torch.tensor(EPS)).to(DEVICE)
ONEMINUSEPS=torch.tensor(1.0).to(DEVICE).float()-EPS
LONEMINUSEPS=torch.log(ONEMINUSEPS)

class BaseAdditiveModel(torch.nn.Module) : 
    def __init__(self,
                 ngenes,
                 init_iweights,
                 init_hweights,
                 init_jweights,
                 init_intercept,
                 synteny_broadcast,
                 init_partition=None,
                 device=DEVICE,
                 normalize_synteny=False,
                 ) : 

        super(BaseAdditiveModel,self).__init__()
        self.ngenes=ngenes
        self.relu_i=torch.nn.ReLU()
        self.relu_h=torch.nn.ReLU()
        self.relu_j=torch.nn.ReLU()
        self.smax=torch.nn.Softmax(dim=0)
        self.synteny_broadcast=synteny_broadcast
        #NOTE you will need to change this if you expand partitioning
        self.iweights=torch.nn.Parameter(_tcast(init_iweights))
        self.hweights=torch.nn.Parameter(_tcast(init_hweights))
        self.jweights=torch.nn.Parameter(_tcast(init_jweights))
        
        #self.intercept=torch.nn.Parameter(torch.ones((4,1)).to(DEVICE).double()*_tcast(init_intercept))
        self.intercept=torch.nn.Parameter(_tcast(init_intercept))
        if init_partition is None : 
            self.partition=torch.nn.Parameter( torch.ones((4,ngenes),device=DEVICE,dtype=torch.float32)*0.25, )
        else :
            self.partition=torch.nn.Parameter(init_partition)

        self.ntotalweights=self.iweights.shape[0]+self.jweights.shape[0]+self.hweights.shape[0]
        self.normalize_synteny=normalize_synteny
        if self.normalize_synteny : 
            self.relu_sy=torch.nn.ReLU()
                            
    def forward(self,I,H,fJ,return_params=False,debug=False) :

        corrected_iweights=self.relu_i(self.iweights)
        corrected_hweights=self.relu_h(self.hweights)
        corrected_jweights=self.relu_j(self.jweights)

        if debug : 
            print(self.iweights.dtype, self.hweights.dtype, self.jweights.dtype)
            print(I.dtype, H.dtype, J.dtype)
        
        corrected_partition=torch.clip(
                            self.smax(self.partition),
                            0.0,1.0) 


        idot=torch.matmul(I,corrected_iweights)
        jdot=torch.matmul(fJ,corrected_jweights) # exj * j -> e


        if len(H.shape) > 1 and not (torch.tensor(H.shape) == 0.0).any() : 
            proto_sely=corrected_partition.view(-1)*(torch.matmul(I,corrected_iweights)+torch.matmul(H,corrected_hweights))
        else :
            proto_sely=corrected_partition.view(-1)*torch.matmul(I,corrected_iweights)

        if self.normalize_synteny : 
            psy2=torch.pow(proto_sely,2)
            synteny_term=torch.sqrt(self.relu_sy(torch.matmul(torch.pow(self.synteny_broadcast,2),psy2) - psy2))
            yhat=proto_sely+synteny_term+jdot+self.intercept.repeat_interleave(self.ngenes)
        else : 
            yhat=torch.matmul(self.synteny_broadcast,proto_sely)+jdot+self.intercept.repeat_interleave(self.ngenes)


        if not return_params : 
            return yhat
        else : 
            return yhat,corrected_iweights,corrected_hweights,corrected_jweights,corrected_partition,self.intercept


    def dissect(self,fI,H,fJ) : 
        corrected_iweights=self.relu(self.iweights)
        corrected_hweights=self.relu(self.hweights)
        corrected_jweights=self.relu(self.jweights)

        corrected_partition=torch.clip(
                            self.smax(self.partition),
                            0.0,1.0).transpose(0,1)


        wiweights=corrected_partition*corrected_iweights
        wiweights=wiweights.ravel()
        idot=torch.matmul(fI,wiweights)

        hdot=(corrected_partition*torch.matmul(H,corrected_hweights)).ravel()

        jdot=torch.matmul(fJ,corrected_jweights) # exj * j -> e
        
        yhat=idot+hdot+jdot+self.intercept.repeat_interleave(self.ngenes)

        return {
            'yhat' : yhat , 
            'corrected_iweights' : corrected_iweights , 
            'corrected_jweights' : corrected_jweights , 
            'corrected_hweights' : corrected_hweights , 
            'idot' : idot, 
            'hdot' : hdot,
            'jdot' : jdot,
            'intercept' : self.intercept.detach().clone(),
            'partition'  : corrected_partition
        }

class LogPosterior(torch.nn.Module) : 
    def __init__(self,
                   working_loglambdas,
                   partition_alpha=10.0,
                  ) : 
        super(LogPosterior,self).__init__()
        
        self.working_loglambdas=working_loglambdas.float()
        self.partition_alpha=partition_alpha

    def likelihood(self,output_log_odds,target_event_cts,n) : 
        return Binomial(logits=output_log_odds,total_count=n).log_prob(target_event_cts).sum()

    def parameter_prior(self,iweights,hweights,jweights) : 
        iparsum=Exponential(rate=torch.exp(self.working_loglambdas[0])).log_prob(iweights).sum()
        hparsum=Exponential(rate=torch.exp(self.working_loglambdas[1])).log_prob(hweights).sum()
        jparsum=Exponential(rate=torch.exp(self.working_loglambdas[2])).log_prob(jweights).sum()
        return iparsum+hparsum+jparsum

    def k_prior(self,iweights,hweights,jweights) : 
        shapes=_tcast([iweights.shape[0],hweights.shape[0],jweights.shape[0]])
        return Geometric(probs=(1-1/torch.exp(self.working_loglambdas))).log_prob(shapes).sum()

    def partition_prior(self,partition) : 
        part=partition.transpose(0,1)
        return Dirichlet(concentration=torch.ones_like(part)*self.partition_alpha).log_prob(part).sum()

    def _assemble_posterior_components(
        self,
        output_log_odds,
        target_event_cts,
        n,
        iweights,
        hweights,
        jweights,
        partition,
        intercept=None, # not used
        ) : 

        outpkg=dict(
            param_prior=self.parameter_prior(iweights,hweights,jweights),
            k_prior=self.k_prior(iweights,hweights,jweights), 
            partition_prior=self.partition_prior(partition),
            ll=self.likelihood(output_log_odds,target_event_cts,n),
        )

        outpkg.update({
            'posterior' : torch.stack(list(outpkg.values())).sum()
        })
            
        #outpkg.update({ 'posterior' : torch.sum(torch.tensor([ v for v in outpkg.values() ],requires_grad=True)) })

        return outpkg

    def forward(self,**kwargs) : 
        return self._assemble_posterior_components(**kwargs)['posterior']

    def decompose(self,
                  output_log_odds,
                  target_event_cts,
                  n,
                  iweights,
                  hweights,
                  jweights,
                  partition,
                  intercept, # not used
                 ) : 
        return { k : v.item() for k,v in self._assemble_posterior_components(
                output_log_odds,
                target_event_cts,
                n,
                iweights,
                hweights,
                jweights,
                partition,
                ).items() }
                  

class Solver(object) : 

    def __init__(self,
                I,
                H,
                J,
                y,
                npats,
                init_iweights,
                init_hweights,
                init_jweights,
                init_intercept,
                synteny_broadcast,
                init_loglambdas,
                init_partition,
                partition_alpha,
                lr=1e-3,
                schedule=True,
                optimizer_method='adam',
                normalize_synteny=False,
                device=DEVICE,
                ): 

        #self.fI=cmu.as_directsum(cmu.flatten_minorly(I),n_blocks=4).to(device)
        self.fI=I.cpu().to_dense().tile((4,1)).to_sparse_coo().to(device)
        self.fH=H.cpu().to_dense().tile((4,1)).to_sparse_coo().to(device)
        self.fJ=cmu.flatten_minorly(J).to(device)
        self.y=y
        self.synteny_broadcast=cmu.as_directsum(
                cmu.flatten_minorly(synteny_broadcast),
                n_blocks=4,
                ).float().to(device)

        self.ngenes=_tcast(I.shape[0]).int()
        self.init_iweights=_tcast(init_iweights)
        self.init_hweights=_tcast(init_hweights)
        self.init_jweights=_tcast(init_jweights)
        self.init_intercept=_tcast(init_intercept)
        self.init_loglambdas=_tcast(init_loglambdas)
        if init_partition is not None : 
            self.init_partition=_tcast(init_partition)
        else : 
            self.init_partition=init_partition
            
        self.npats=_tcast(npats)
        self.partition_alpha=partition_alpha
        self.lr=lr
        self.normalize_synteny=normalize_synteny

        #assert self.X.shape[1] == self.init_weights.shape[0]
        #assert self.X.shape[0] == self.y.shape[0]

        self.spawn(schedule=schedule,which=optimizer_method)
        

    def spawn(self,schedule=None,which='adam',**kwargs) : 

        self.model=self.spawn_model(**kwargs)
        self.losser=self.spawn_losser()
        self.optimizer=self.spawn_optimizer(which=which)

        if schedule is not None or not schedule: 
            self.scheduler=self.spawn_scheduler()
        else : 
            self.scheduler=None
        

    def spawn_model(self,**kwargs) : 
        return BaseAdditiveModel(
                  ngenes=kwargs.get('ngenes',self.ngenes.clone()),
                  init_iweights=kwargs.get('init_iweights',self.init_iweights.clone()),
                  init_hweights=kwargs.get('init_hweights',self.init_hweights.clone()),
                  init_jweights=kwargs.get('init_jweights',self.init_jweights.clone()),
                  init_intercept=kwargs.get('init_intercept',self.init_intercept.clone()),
                  synteny_broadcast=kwargs.get('synteny_broadcast',self.synteny_broadcast.clone()),
                  init_partition=kwargs.get('init_partition',self.init_partition),
                  normalize_synteny=kwargs.get('normalize_synteny',self.normalize_synteny),
                  )

    def spawn_losser(self) : 
        return LogPosterior(
            working_loglambdas=self.init_loglambdas,
            partition_alpha=self.partition_alpha,
        )

    def spawn_optimizer(self,which='adam') : 
        if hasattr(self,'optimizer') and self.optimizer is not None : 
            self.optimizer.zero_grad()
        if which == 'adam' : 
            return torch.optim.Adam(self.model.parameters(),lr=self.lr,maximize=True)
        #return torch.optim.AdamW(self.model.parameters(),lr=self.lr,maximize=True,weight_decay=self.weightdecay)
        else: 
            return torch.optim.SGD(self.model.parameters(),lr=1e-4,maximize=True)

    def spawn_scheduler(self) : 
        scheduler=torch.optim.lr_scheduler.LinearLR(
               optimizer=self.optimizer,
               start_factor=0.1,
               end_factor=1.0,
               total_iters=int(5e3),
           )


    def fwargs(self,fI=None,fH=None,fJ=None,**kwargs) : 
        if fI is None : 
            fI=self.fI
        if fH is None : 
            H=self.fH
        if fJ is None : 
            fJ=self.fJ
            
        olo,iweights,hweights,jweights,partition,intercept=self.model(fI,H,fJ,return_params=True)
        
        _fwargs=dict(
                output_log_odds=kwargs.get('output_log_odds',olo),
                target_event_cts=kwargs.get('target_event_cts',self.y),
                n=kwargs.get('n',self.npats),
                iweights=kwargs.get('iweights',iweights),
                hweights=kwargs.get('hweights',hweights),
                jweights=kwargs.get('jweights',jweights),
                partition=kwargs.get('partition',partition),
                intercept=kwargs.get('intercept',intercept),
         )

        return _fwargs
                

    def _closure(self,**kwargs) : 

        self.optimizer.zero_grad()
        loss=self.losser(**self.fwargs(**kwargs))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
        return loss

    def tick(self,**kwargs) : 

        closs=partial(self._closure,**kwargs)
        self.optimizer.step(closs)

        if self.scheduler : 
            scheduler.step()

    def snap(self,**kwargs) : 
        with torch.no_grad() :
            goods=self.losser.decompose(**self.fwargs(**kwargs))
            return goods

    def meta(self,mask_keys=['gene','system','signature'])   : 

        metad=dict()
        nwt=[self.model.iweights.detach(),self.model.hweights.detach(),self.model.jweights.detach()]
        #metad.update({  'partition_'+str(x) : v.item() for x,v in enumerate(self.model.partition)  })
        metad.update({ 'n_nonzero_'+mask_keys[x] : (m>0).sum().cpu().numpy() for x,m in enumerate(nwt) })

        return metad

    def dump(self,dense=False) : 
        wt=torch.clip(torch.concatenate([
            self.model.iweights.detach(),
            self.model.hweights.detach(),
            self.model.jweights.detach(),
            ]),0,torch.inf)
        inter=self.model.intercept.detach()

        if not dense : 
            nzwta=torch.nonzero(wt).ravel()
            nzwt=wt[nzwta]
            return (nzwta,nzwt),inter
        return wt, inter

    def reinit(self) : 
        self.optimizer.zero_grad()

        old_iweights=torch.clip(self.model.iweights.detach(),0,torch.inf)
        old_hweights=torch.clip(self.model.hweights.detach(),0,torch.inf)
        old_jweights=torch.clip(self.model.jweights.detach(),0,torch.inf)
        old_intercept=self.model.intercept.detach()
        old_partition=torch.clip(self.model.partition.detach(),0,20)

        self.spawn( 
            init_iweights=old_iweights,
            init_hweights=old_hweights,
            init_jweights=old_jweights,
            init_intercept=old_intercept,
            init_partition=old_partition,
        )

class Slooper(object) : 
    def __init__(
            self,
            solver,
            checkevery=50,
            converge_on_n=5,
            dumpfilename=None,
            ) :

        super(Slooper,self).__init__()
        self.solver=solver
        self.checkevery=checkevery
        self.best_posterior=-1*torch.inf
        self.best_params=None
        self.best_datum=None
        self.converge_on_n=converge_on_n
        if self.converge_on_n > 0 : 
            self.lastn=-1*torch.inf*torch.ones((self.converge_on_n,),device=DEVICE,dtype=torch.float)

        self.dumpfilename = dumpfilename

        self.opt_data=list()
        self.epoch_counter=0

    def crank(self,total_epochs=int(2e4),monitor=False) : 

        if self.dumpfilename is not None : 
            dumpfile=open(dumpfilename,'ab')

        if monitor :
            msg();

        for epoch in range(total_epochs) : 

            self.epoch_counter += 1

            if ( self.epoch_counter % self.checkevery == 0 ) : 
                datum=dict(epoch=self.epoch_counter) | self.solver.meta() | self.solver.snap()
                self.opt_data.append(
                    datum
                )

                dumpgoods=self.solver.dump()
                
                #if self.epoch_counter > 5*self.checkevery : 
                    #self.ofinterest[nzwta]=True
                    
                if self.dumpfilename is not None : 
                    pickle.dump(dumpgoods,dumpfile)

                thissnap=datum['posterior']
                if thissnap>self.best_posterior : 
                    
                    self.best_posterior = thissnap
                    self.best_datum=datum
                    self.best_params=self.solver.fwargs()

                if monitor : 
                    mstr='    Epoch {: >6} : {: >6.3e} {: >-6.3f} {: >-6.3f}'.format(self.epoch_counter,thissnap,thissnap-self.best_posterior,thissnap-self.lastn[-1])
                    time.sleep(0)
                    if  hasattr(builtins,'__IPYTHON__') : 
                        display(mstr,clear=True)
                    else : 
                        msg(mstr,end='\r')
                    
                if self.converge_on_n > 0 : 
                    if (thissnap <= self.lastn).all() or ((self.best_posterior > thissnap ) and ( self.best_posterior < (self.lastn+1.0)).all()) :
                        if monitor : 
                            msg()
                            msg('    Converged at epoch {}.'.format(self.epoch_counter))
                        break
                        
                    self.lastn[:-1]=self.lastn[1:].clone()
                    self.lastn[-1]=thissnap

            self.solver.tick()

        if self.dumpfilename is not None : 
            dumpfile.close()

    def wrap(self) : 
        import pandas as pd
        return self.best_params,self.best_datum,pd.DataFrame(self.opt_data)

    def export(self) : 

        bam=BaseAdditiveModel( ngenes=self.solver.ngenes.cpu(),
                 init_iweights=self.best_params['iweights'].detach().clone().cpu(),
                 init_hweights=self.best_params['hweights'].detach().clone().cpu(),
                 init_jweights=self.best_params['jweights'].detach().clone().cpu(),
                 init_intercept=self.best_params['intercept'].detach().clone().cpu(),
                 init_partition=self.best_params['partition'].detach().clone().cpu(),
                 synteny_broadcast=self.solver.synteny_broadcast,
                 normalize_synteny=self.solver.normalize_synteny,
                 device=torch.device('cpu'),
        )

        return bam.eval()

if __name__ == '__main__' :  

    rf=ns.indir
    os.makedirs(ns.outdir,exist_ok=True)
    I=torch.load(opj(rf,'I.pt'),weights_only=False)
    H=torch.load(opj(rf,'H.pt'),weights_only=False).to_sparse_coo()
    J=torch.load(opj(rf,'J.pt'),weights_only=False).to_sparse_coo()
    synteny_broadcast=torch.load(opj(rf,'synteny_broadcast.pt'),weights_only=False)
    #y=torch.load(opj(rf,'y.pt'),weights_only=False).to(DEVICE)
    sparse_omics=torch.load(opj(rf,'sparse_omics.pt'),weights_only=False)
    y=sparse_omics.to_dense().sum(axis=2).to_sparse_coo().coalesce()
    print(y.shape)
   
    yd=y.to_dense()
    NPATS=ns.npats

    import scipy
    import scipy.linalg
    import scipy.sparse

    guessbp=torch.load(opj(rf,'guessbp.pt'),weights_only=False)

    init_iweights=_tcast(guessbp['iweights'])
    init_hweights=_tcast(guessbp['hweights'])
    init_jweights=_tcast(guessbp['jweights'])
    init_intercept=_tcast(guessbp['intercept'])
    init_partition=_tcast(guessbp['partition'])

    solver=Solver(
        I=I.to(DEVICE),
        H=H.to(DEVICE),
        J=J.to(DEVICE),
        y=yd.ravel().to(DEVICE),
        npats=torch.tensor(float(ns.npats)).to(DEVICE),
        init_iweights=init_iweights,
        init_hweights=init_hweights,
        init_jweights=init_jweights,
        init_intercept=init_intercept,
        synteny_broadcast=synteny_broadcast,
        init_partition=init_partition,
        init_loglambdas=_tcast([float(ns.lambda_gene),float(ns.lambda_system),float(ns.lambda_gb)]).float(),
        partition_alpha=float(ns.alpha_partition),
        lr=1e-2,
        #lr=1e-3,
        normalize_synteny=ns.normalize_synteny,
        schedule=True,
        optimizer_method='adam',
    )

    slooper=Slooper(solver)
    slooper.crank(monitor=True)
    bp,bd,df=slooper.wrap()

    import os
    torch.save({ k : v.detach() for k,v in bp.items() },os.path.join(ns.outdir,'bp.pt'))
    torch.save(bd,os.path.join(ns.outdir,'bd.pt'))
    df.to_csv(os.path.join(ns.outdir,'df.csv'))
    with open(os.path.join(ns.outdir,'asns.pickle'),'wb') as f : 
        pickle.dump(vars(ns),f)

    if int(ns.resample) > 0 : 
        fold_generator_0=np.random.Generator(np.random.MT19937(int('0xdeadbeef',16)))
        folds_labels=torch.tensor(fold_generator_0.choice(
                            a=np.cast['int'](np.arange(I.shape[0]) // (I.shape[0]/int(ns.resample))),
                            size=I.shape[0],
                            replace=False
                        ),device=I.device)
        if ns.cheat : 
            sampler_init_iweights=bp['iweights'].detach().clone()
            sampler_init_hweights=bp['hweights'].detach().clone()
            sampler_init_jweights=bp['jweights'].detach().clone()
            sampler_init_intercept=bp['intercept'].detach().clone()
            sampler_init_partition=bp['partition'].detach().clone()
        else :
            sampler_init_iweights=init_iweights
            sampler_init_hweights=init_hweights
            sampler_init_jweights=init_jweights
            sampler_init_intercept=init_intercept
            sampler_init_partition=init_partition

        for rsep in range(int(ns.resample)) :
            sampargs=torch.argwhere(folds_labels != rsep).ravel()

            thisI=I.to_dense()[sampargs,:].to_sparse_coo()
            if len(H.shape) < 2 :  
                thisH=H
            else : 
                thisH=H.to_dense()[sampargs,:].to_sparse_coo()
            thisJ=J.to_dense()[:,sampargs,:].to_sparse_coo()
            thisy=y.to_dense()[:,sampargs].ravel()
            thissb=synteny_broadcast.to_dense()[:,sampargs,:][:,:,sampargs].to_sparse_coo()


            solver=Solver(
                I=thisI.to(DEVICE),
                H=thisH.to(DEVICE),
                J=thisJ.to(DEVICE),
                y=thisy.to(DEVICE),
                npats=torch.tensor(float(ns.npats)).to(DEVICE),
                init_iweights=sampler_init_iweights,
                init_hweights=sampler_init_hweights,
                init_jweights=sampler_init_jweights,
                init_intercept=sampler_init_intercept,
                init_partition=sampler_init_partition[:,sampargs],
                synteny_broadcast=thissb,
                init_loglambdas=_tcast([float(ns.lambda_gene),float(ns.lambda_system),float(ns.lambda_gb)]).float(),
                partition_alpha=float(ns.alpha_partition),
                lr=1e-2,
                #lr=1e-3,
                normalize_synteny=ns.normalize_synteny,
                schedule=True,
                optimizer_method='adam',
            )
            slooper=Slooper(solver)
            slooper.crank(monitor=True)
            samp_bp,samp_bd,samp_df=slooper.wrap()

            thisoutdir=opj(ns.outdir,'resample_'+str(rsep))
            os.makedirs(thisoutdir,exist_ok=True)
            torch.save({ k : v.detach() for k,v in samp_bp.items() },os.path.join(thisoutdir,'bp.pt'))
            torch.save(samp_bd,os.path.join(thisoutdir,'bd.pt'))
            samp_df.to_csv(os.path.join(thisoutdir,'df.csv'))
                


####fjguess=J.to_dense().max(axis=0).values
####yguess=torch.special.logit((y.to_dense().max(axis=0).values+1)/NPATS)
####figuess=torch.diag(torch.ones(yguess.shape))
####Xguess=torch.concatenate([figuess,H.to_dense(),fjguess,torch.ones((yguess.shape[0],1))],axis=1).to_sparse_coo()

####sXnp=scipy.sparse.csr_matrix(
####    (
####        Xguess.values().cpu().numpy(),
####        Xguess.indices().cpu().numpy(),
####    ),shape=Xguess.shape)

####outs=scipy.sparse.linalg.lsmr(sXnp,yguess.cpu().numpy())
####corr=cmu.cc2tens(Xguess,yguess.unsqueeze(-1)).squeeze()
####baseguess=torch.stack([_tcast(outs[0]).cpu(),2*corr],axis=-1).max(axis=1).values[:-1].float()

####if len(H.shape) < 2 : 
####    ihlen=I.shape[1]
####else : 
####    ihlen=I.shape[1]+H.shape[1]

####guessi=torch.clip(baseguess[:I.shape[1]],0,np.inf).clone().contiguous()
####guessh=torch.clip(baseguess[I.shape[1]:ihlen],0,np.inf).clone().contiguous()
####guessj=torch.clip(baseguess[ihlen:],0,np.inf).clone().contiguous()
####guessint=(yguess.cpu()-torch.matmul(Xguess.to_dense()[:,:-1],baseguess)).min().expand(4,).float()
