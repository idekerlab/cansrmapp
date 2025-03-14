#!/usr/bin/env python
# coding: utf-8
if __name__ == '__main__'  :
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--lambda_selection',default=3.0,action='store')
    parser.add_argument('--lambda_gb',default=1.5,action='store')
    parser.add_argument('--alpha_partition',default=2.5,action='store')
    parser.add_argument('--indir',required=True,action='store')
    parser.add_argument('--outdir',required=True,action='store')
    parser.add_argument('--npats',action='store',default=498.0)
    parser.add_argument('--n_cycles',action='store',default=30)
    parser.add_argument('--n_chains',action='store',default=10)
    ns=parser.parse_args()


import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
opj=os.path.join

import builtins
print(hasattr(builtins,'__IPYTHON__'))
import cansrmapp
import cansrmapp.utils as cmu
from cansrmapp import DEVICE,np,torch,random,summarize_random_states
from cansrmapp.utils import _tcast,mask_sparse_columns,msg,time
from functools import partial
from torch.distributions import Geometric,Exponential,Gamma,Binomial,Dirichlet
import pickle

EPS=torch.finfo(torch.float32).eps
LEPS=torch.log(torch.tensor(EPS)).to(DEVICE)
ONEMINUSEPS=torch.tensor(1.0).to(DEVICE).float()-EPS
LONEMINUSEPS=torch.log(ONEMINUSEPS)

_POSTERIOR='posterior'


#sampling_generator=np.random.Generator(np.random.MT19937(seed=cmu.word_to_seed('Chupacabra')))

def cansrmapp_function(I,H,J,E,R,w,v,z,c) : 
    """
        This should handle dimensionality and domain issues; not others

    """
    if len(I.shape) > 1 and not (torch.tensor(I.shape) == 0.0).any() : 
        idot=torch.matmul(I,w)
    else : 
        idot=torch.zeros_like(R.view(-1))

    if len(H.shape) > 1 and not (torch.tensor(H.shape) == 0.0).any() : 
        proto_sely=E.view(-1)*(idot+torch.matmul(H,v))
    else :
        proto_sely=E.view(-1)*idot

    if len(J.shape) == 3  : 
        fJ=cmu.flatten_minorly(J)
    else : 
        fJ=J

    if len(fJ.shape) > 1 and not (torch.tensor(fJ.shape) == 0.0).any() : 
        jdot=torch.matmul(fJ,z) # exj * j -> e
    else : 
        jdot=torch.zeros_like(R.view(-1))

    tall_intercept=c.repeat_interleave(R.shape[0]//c.shape[0])
    yhat=torch.matmul(R,proto_sely)+jdot+tall_intercept

    return yhat


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
                            
    def forward(self,I,H,fJ,synteny_broadcast=None,intercept=None,return_params=False,debug=False) :

        corrected_iweights=self.relu_i(self.iweights)
        corrected_hweights=self.relu_h(self.hweights)
        corrected_jweights=self.relu_j(self.jweights)

        corrected_partition=torch.clip(
                            self.smax(self.partition),
                            0.0,1.0) 

        if synteny_broadcast is None : 
            synteny_broadcast=self.synteny_broadcast

        if intercept is None : 
            intercept=self.intercept

        if debug : 
            print(self.iweights.dtype, self.hweights.dtype, self.jweights.dtype)
            print(I.dtype, H.dtype, J.dtype)
        
        yhat=cansrmapp_function(
                I=I,
                H=H,
                J=fJ,
                E=corrected_partition,
                R=synteny_broadcast,
                w=corrected_iweights,
                v=corrected_hweights,
                z=corrected_jweights,
                c=intercept,
                )

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
        hparsum=Exponential(rate=torch.exp(self.working_loglambdas[0])).log_prob(hweights).sum()
        jparsum=Exponential(rate=torch.exp(self.working_loglambdas[1])).log_prob(jweights).sum()
        return iparsum+hparsum+jparsum

    def k_prior(self,iweights,hweights,jweights) : 
        shapes=_tcast([iweights.shape[0]+hweights.shape[0],jweights.shape[0]])
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

        #assert self.X.shape[1] == self.init_weights.shape[0]
        #assert self.X.shape[0] == self.y.shape[0]

        self.spawn(schedule=schedule,which=optimizer_method)
        

    def spawn(self,schedule=None,which='adam',**kwargs) : 

        self.model=self.spawn_model(**kwargs)
        self.losser=self.spawn_losser()
        self.optimizer=self.spawn_optimizer(which=which)

        if schedule is None or not schedule : 
            self.scheduler=None
        else : 
            self.scheduler=self.spawn_scheduler()
        

    def spawn_model(self,**kwargs) : 
        return BaseAdditiveModel(
                  ngenes=kwargs.get('ngenes',self.ngenes.clone()),
                  init_iweights=kwargs.get('init_iweights',self.init_iweights.clone()),
                  init_hweights=kwargs.get('init_hweights',self.init_hweights.clone()),
                  init_jweights=kwargs.get('init_jweights',self.init_jweights.clone()),
                  init_intercept=kwargs.get('init_intercept',self.init_intercept.clone()),
                  synteny_broadcast=kwargs.get('synteny_broadcast',self.synteny_broadcast.clone()),
                  init_partition=kwargs.get('init_partition',self.init_partition),
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

    def spawn_scheduler(self,
                        scheduler=torch.optim.lr_scheduler.CyclicLR,
                        scheduler_kws=dict(base_lr=5e-4,max_lr=1e-2,step_size_up=500)) : 

        return scheduler(optimizer=self.optimizer,**scheduler_kws)

       #scheduler=torch.optim.lr_scheduler.LinearLR,
       #scheduler_kws=dict(start_factor=1.0,end_factor=0.02,total_iters=int(1e3))) : 
       #scheduler=torch.optim.lr_scheduler.LinearLR(
       #       optimizer=self.optimizer,
       #       start_factor=1.0, # 5e-2
       #       end_factor=0.02,  # 1e-3
       #       total_iters=int(1e3),
       #   )
        #scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
               #optimizer=self.optimizer,
               #T_max=int(7.5e2),
           #)
        # ^ this was working pretty well
        #return scheduler

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

    def tick(self,noise=False,**kwargs) : 

        closs=partial(self._closure,**kwargs)

        #NEW
        if noise : 
            for param in self.model.parameters() : 
                if param.grad is not None : 
                    param.grad += 0.01*torch.randn_like(param.grad)

        self.optimizer.step(closs)

        if self.scheduler : 
            self.scheduler.step()

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

   #def dump(self,dense=False) : 
   #    wt=torch.clip(torch.concatenate([
   #        self.model.iweights.detach(),
   #        self.model.hweights.detach(),
   #        self.model.jweights.detach(),
   #        ]),0,torch.inf)
   #    inter=self.model.intercept.detach()
   #    partition=self.model.partition.detach()

   #    if not dense : 
   #        nzwta=torch.nonzero(wt).ravel()
   #        nzwt=wt[nzwta]
   #        return (nzwta,nzwt),inter,partition
   #    return wt,inter,partition
    def dump(self) : 
            iwt=torch.clip(self.model.iweights.detach(),0,torch.inf)
            hwt=torch.clip(self.model.hweights.detach(),0,torch.inf)
            jwt=torch.clip(self.model.jweights.detach(),0,torch.inf)
            inter=self.model.intercept.detach()
            partition=self.model.partition.detach()

            return iwt,hwt,jwt,inter,partition

    def reinit(self,iweights=None,hweights=None,jweights=None,intercept=None,partition=None) : 
        self.optimizer.zero_grad()

        old_iweights=torch.clip(iweights if iweights is not None else self.model.iweights.detach(),0,torch.inf)
        old_hweights=torch.clip(hweights if hweights is not None else self.model.hweights.detach(),0,torch.inf)
        old_jweights=torch.clip(jweights if jweights is not None else self.model.jweights.detach(),0,torch.inf)
        old_intercept=intercept if intercept is not None else self.model.intercept.detach()
        old_partition=partition if partition is not None else torch.clip(self.model.partition.detach(),0,20)

        self.spawn( 
            init_iweights=old_iweights,
            init_hweights=old_hweights,
            init_jweights=old_jweights,
            init_intercept=old_intercept,
            init_partition=old_partition,
            schedule=(hasattr(self,'scheduler') and self.scheduler is not None),
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
            self.clear_lastn()

        self.dumpfilename = dumpfilename

        self.opt_data=list()
        self.epoch_counter=0

    def clear_lastn(self) : 
        self.lastn=-1*torch.inf*torch.ones((self.converge_on_n,),device=DEVICE,dtype=torch.float)

    def crank(self,total_epochs=int(2e4),monitor=False,burnin=0) : 

        best_posterior_this_crank=-1*torch.inf
        best_datum_this_crank=dict()

        pbar_wiped=False

        if self.dumpfilename is not None : 
            dumpfile=open(self.dumpfilename,'ab')

        if monitor :
            msg();

        if self.converge_on_n > 0 : 
            self.clear_lastn()

        for epoch in range(total_epochs) : 
            self.epoch_counter += 1
            datum=None

            if (epoch > burnin) : 
                thissnap=self.solver.snap()
                thispos=thissnap[_POSTERIOR]

                if thispos > self.best_posterior : 
                    self.best_posterior = thispos
                    datum=dict(epoch=self.epoch_counter) | self.solver.meta() | thissnap
                    self.best_datum=datum
                    self.best_params=self.solver.fwargs()

                if thispos > best_posterior_this_crank : 
                    best_posterior_this_crank=thispos
                    if datum is None : 
                        datum=dict(epoch=self.epoch_counter) | self.solver.meta() | thissnap
                    best_datum_this_crank=datum
                    best_params_this_crank=self.solver.fwargs()

            # this now is _just_ about diagnostics/logging
                if ( self.epoch_counter % self.checkevery == 0 ) : 
                    if datum is None : datum=dict(epoch=self.epoch_counter) | self.solver.meta() | thissnap

                    if self.solver.scheduler is not None : 
                        lastlr=self.solver.scheduler.get_last_lr()[0]
                        datum.update({ 'lr' : self.solver.scheduler.get_last_lr()[0] })
                    else:
                        lastlr=self.solver.lr

                    self.opt_data.append(datum)
                    
                    if self.dumpfilename is not None : 
                        dumpgoods=self.solver.dump()
                        pickle.dump(dumpgoods,dumpfile)

                    if monitor : 
                        if self.converge_on_n > 0 : 
                            delta=thispos-self.lastn[-1]
                        else : 
                            delta=0.0

                        if not pbar_wiped : 
                            msg(' '*90,end='\r')
                            time.sleep(0)
                            pbar_wiped=True
                        mstr='    Epoch {: >6} : {: >8.0f} {: >-8.0f} {: >-8.2f} {: >-4.5f}'.format(self.epoch_counter,thispos,thispos-self.best_posterior,delta,lastlr)
                        msg(mstr,end='\r')
                        time.sleep(0)
                    
                    if self.converge_on_n > 0 : 
                        if self.best_posterior > thispos and (self.lastn.max()-self.lastn.min()) < 0.5 :
                            if monitor : 
                                time.sleep(0)
                                msg()
                                msg('    Converged at epoch {}.'.format(self.epoch_counter))
                            break
                        
                        self.lastn[:-1]=self.lastn[1:].clone()
                        self.lastn[-1]=thispos

                self.solver.tick()
            else : 
                if monitor and ((self.epoch_counter % self.checkevery) == 0) and epoch < burnin : 
                    mstr='    Epoch {: >6} : '.format(self.epoch_counter)
                    q=int(60*epoch/burnin)
                    aq=60-q
                    mstr+='#'*q
                    mstr+=' '*aq
                    mstr+='|'
                    msg(mstr,end='\r')
                    time.sleep(0)

                self.solver.tick(noise=True)

        if self.dumpfilename is not None : 
            dumpfile.close()

        return best_datum_this_crank | best_params_this_crank

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

    #guessbp=torch.load(opj(rf,'guessbp.pt'),weights_only=False)
    #init_iweights=_tcast(guessbp['iweights'])
    #init_hweights=_tcast(guessbp['hweights'])
    #init_jweights=_tcast(guessbp['jweights'])
    #init_intercept=_tcast(guessbp['intercept'])
    #init_partition=_tcast(guessbp['partition'])

    init_iweights=torch.ones((I.shape[-1],)).float().to(DEVICE)
    init_hweights=torch.ones((H.shape[-1],)).float().to(DEVICE)
    init_jweights=torch.ones((J.shape[-1],)).float().to(DEVICE)
    init_intercept=-10*torch.ones((4,)).float().to(DEVICE)
    init_partition=0.25*torch.ones((4,I.shape[0])).float().to(DEVICE)

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
        init_loglambdas=_tcast([float(ns.lambda_selection),float(ns.lambda_gb)]).float(),
        partition_alpha=float(ns.alpha_partition),
        lr=1e-2, 
        #lr=5e-2,
        #lr=1e-3,
        schedule=True,
        optimizer_method='adam',
    )
   #solver.optimizer=torch.optim.Adam(
   #    solver.model.parameters(),
   #    lr=solver.lr,
   #    maximize=True,
   #    betas=(0.8,0.999),)
   #solver.scheduler=solver.spawn_scheduler(
   #    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 
   #    scheduler_kws=dict(T_0=int(1e3))
   #)


    _mks=['iweights', 'hweights', 'jweights', 'intercept', 'partition']

    def _instantiate_sgrids(model,memory_size=5) : 
        return {  mk : torch.zeros((memory_size,*getattr(model,mk).shape)).to(getattr(model,mk).device).float() for mk in _mks
                } | { 'posterior' : -1*torch.inf*torch.ones((memory_size,)).float().to(DEVICE) }

    def _update_sgrid(sgrid,new,pos) :

        if pos < 0 : 
            sgrid[:-1]=sgrid[1:].clone()
            sgrid[-1]=new.clone()
        else : 
            sgrid[pos]=new.clone()

        return sgrid

    sgrids=_instantiate_sgrids(solver.model,memory_size=int(ns.n_chains)*int(ns.n_cycles))

    sloopers=[ Slooper(solver,checkevery=200,converge_on_n=0) for x in range(int(ns.n_chains)) ]

    #last_bds=torch.zeros((int(ns.n_chains),)).to(DEVICE).float()
    last_bds=[dict()]*int(ns.n_chains)

    try: 
        x=0
        msg('{: >3}|{: >3}|{: >3}|{: >9}|{: >9}'.format(
            'cyc',
            'chn',
            'tot',
            'This cyc.',
            'All cyc.',
            ))
        for me in range(int(ns.n_cycles)) : 
            for chain in range(int(ns.n_chains)) :
                slooper=sloopers[chain]
                if me > 0 : 
                    slooper.solver.reinit(
                        iweights =( last_bds[chain]['iweights'].detach() + slooper.best_params['iweights'].detach() + 0.25)/3,
                        hweights =( last_bds[chain]['hweights'].detach() + slooper.best_params['hweights'].detach() + 0.25)/3,
                        jweights =( last_bds[chain]['jweights'].detach() + slooper.best_params['jweights'].detach() + 0.25)/3,
                        intercept=( last_bds[chain]['intercept'].detach() + slooper.best_params['intercept'].detach() - 10.0)/3,
                        partition=( last_bds[chain]['partition'].detach() + slooper.best_params['partition'].detach() - +0.25)/3,
                    )

                best_datum_this=slooper.crank(monitor=False,burnin=int(3e3),total_epochs=int(6e3))
                last_bds[chain]=best_datum_this

                for mk in _mks : 
                    sgrids[mk]=_update_sgrid(sgrids[mk],best_datum_this[mk].detach(),x)
                sgrids[_POSTERIOR]=_update_sgrid(sgrids[_POSTERIOR],torch.tensor(best_datum_this[_POSTERIOR]).to(DEVICE),x)

                msg('{: >3d}|{: >3d}|{: >3d}|{: >9.0f}|{: >9.0f}'.format(
                    me,
                    chain,
                    x,
                    best_datum_this[_POSTERIOR],
                    slooper.best_datum[_POSTERIOR]
                    ))

                x += 1
    
    except KeyboardInterrupt : 
        pass

    aobest=-1
    best=-1*torch.inf
    for c,slooper in enumerate(sloopers) : 
        sc=str(c)

        bp,bd,df=slooper.wrap()
        if bd[_POSTERIOR] > best : 
            aobest=c
            best=bd[_POSTERIOR]
        import os
        torch.save({ k : v.detach() for k,v in bp.items() },opj(ns.outdir,'singleton_bp_'+str(c)+'.pt'))
        torch.save(bd,opj(ns.outdir,'singleton_bd_'+str(c)+'.pt'))
        df.to_csv(opj(ns.outdir,'singleton_df_'+str(c)+'.csv'))

    torch.save(sgrids,opj(ns.outdir,'sgrids.pt'))
    torch.save({ k : v.mean(axis=0).cpu() for k,v in sgrids.items() },opj(ns.outdir,'smeans.pt'))
    with open(opj(ns.outdir,'asns.pickle'),'wb') as f : 
        pickle.dump(vars(ns),f)

    os.symlink(opj('singleton_bp_'+str(aobest)+'.pt'), opj(ns.outdir,'bp.pt'))
    os.symlink(opj('singleton_bd_'+str(aobest)+'.pt'), opj(ns.outdir,'bd.pt'))
    os.symlink(opj('singleton_df_'+str(aobest)+'.csv'),opj(ns.outdir,'df.csv'))

   #msg('Forming consensus...')

   #iwt=sgrids['iweights'].mean(axis=0)
   #hwt=sgrids['hweights'].mean(axis=0)
   #jwt=sgrids['jweights'].mean(axis=0)
   ##Ebar=torch.clip(sgrids['partition'].mean(axis=0),EPS,ONEMINUSEPS)
   #Ebar=torch.softmax(sgrids['partition'].mean(axis=0),dim=0)
   #cbar=sgrids['intercept'].mean(axis=0)


   #consensus_slooper=Slooper(solver,checkevery=200,converge_on_n=0)
   #consensus_slooper.solver.reinit(
   #                    iweights=iwt,
   #                    hweights=hwt,
   #                    jweights=jwt,
   #                    intercept=cbar,
   #                    partition=Ebar,
   #                )
   #consensus_slooper.solver.scheduler=None
   #consensus_slooper.solver.lr=1e-3
   #consensus_slooper.solver.optimizer=consensus_slooper.solver.spawn_optimizer()
   #consensus_slooper.crank(monitor=False,burnin=0,total_epochs=int(1e3))
   #bp,bd,df=consensus_slooper.wrap()
   #import os
   #torch.save({ k : v.detach() for k,v in bp.items() },os.path.join(ns.outdir,'bp.pt'))
   #torch.save(bd,os.path.join(ns.outdir,'bd.pt'))
   #df.to_csv(os.path.join(ns.outdir,'df.csv'))
   #msg('Done.')


    
        
   # this block was getting results in the range of 784000
   #solver.optimizer=torch.optim.Adam(
   #    solver.model.parameters(),
   #    lr=solver.lr,
   #    maximize=True,
   #    betas=(0.8,0.999),)
   #solver.scheduler=solver.spawn_scheduler(
   #    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 
   #    scheduler_kws=dict(T_0=int(1e3))
   #)

   #slooper=Slooper(solver,checkevery=50,converge_on_n=0)
   #best_datum_this=slooper.crank(monitor=True,burnin=5000,total_epochs=int(1e4))

   #try : 
   #    for me in range(int(ns.n_cycles)) : 
   #        best_datum_this=slooper.crank(monitor=True,burnin=5000,total_epochs=int(1e4))
   #        slooper.solver.reinit(
   #            iweights =( best_datum_this['iweights'].detach() + slooper.best_params['iweights'].detach() + 0.25)/3,
   #            hweights =( best_datum_this['hweights'].detach() + slooper.best_params['hweights'].detach() + 0.25)/3,
   #            jweights =( best_datum_this['jweights'].detach() + slooper.best_params['jweights'].detach() + 0.25)/3,
   #            intercept=( best_datum_this['intercept'].detach() + slooper.best_params['intercept'].detach() - 10.0)/3,
   #            partition=( best_datum_this['partition'].detach() + slooper.best_params['partition'].detach() - +0.25)/3,
   #        )
   #        solver.optimizer=torch.optim.Adam(
   #            solver.model.parameters(),
   #            lr=solver.lr,
   #            maximize=True,
   #            betas=(0.8,0.999),)
   #        solver.scheduler=solver.spawn_scheduler(
   #            scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 
   #            scheduler_kws=dict(T_0=int(1e3))
   #        )
   #
   #except KeyboardInterrupt : 
   #    pass
   #bp,bd,df=slooper.wrap()
   #import os
   #torch.save({ k : v.detach() for k,v in bp.items() },os.path.join(ns.outdir,'bp.pt'))
   #torch.save(bd,os.path.join(ns.outdir,'bd.pt'))
   #df.to_csv(os.path.join(ns.outdir,'df.csv'))
   #with open(os.path.join(ns.outdir,'asns.pickle'),'wb') as f : 
   #    pickle.dump(vars(ns),f)

   ###########################################################################
   # this block was getting results slightly higher, but more reproducible:
   # it is in the 'old_possible_keepers' folders
   #try: 
   #    for x in range(int(ns.n_cycles)) : 

   #        if x == 0 : 
   #            app='.'
   #        elif x < (0.9*float(ns.n_cycles)) and x < (int(ns.n_cycles)-1) : 
   #            print('from moving init')
   #            slooper.solver.reinit(
   #                iweights =( best_datum_this['iweights'].detach() + slooper.best_params['iweights'].detach() + 1.0)/3,
   #                hweights =( best_datum_this['hweights'].detach() + slooper.best_params['hweights'].detach() + 1.0)/3,
   #                jweights =( best_datum_this['jweights'].detach() + slooper.best_params['jweights'].detach() + 1.0)/3,
   #                intercept=( best_datum_this['intercept'].detach() + slooper.best_params['intercept'].detach() - 10.0)/3,
   #                partition=( best_datum_this['partition'].detach() + slooper.best_params['partition'].detach() - +0.25)/3,
   #            )
   #            app=' '
   #        else : 
   #            print('from means')
   #            slooper.solver.reinit(
   #                iweights =sgrids['iweights'][:x].mean(axis=0),
   #                hweights =sgrids['hweights'][:x].mean(axis=0),
   #                jweights =sgrids['jweights'][:x].mean(axis=0),
   #                intercept=sgrids['intercept'][:x].mean(axis=0)-1.0,
   #                partition=sgrids['partition'][:x].mean(axis=0),
   #            )
   #            app='~'
   #            
   #        solver.optimizer=torch.optim.Adam(
   #            solver.model.parameters(),
   #            lr=solver.lr,
   #            maximize=True,
   #            betas=(0.8,0.999),)
   #        
   #        solver.scheduler=solver.spawn_scheduler(
   #            scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 
   #            scheduler_kws=dict(T_0=int(1e3))
   #        )
   #        best_datum_this=slooper.crank(monitor=True,burnin=5000,total_epochs=10000)

   #        for mk in _mks : 
   #            sgrids[mk]=_update_sgrid(sgrids[mk],best_datum_this[mk].detach(),x)
   #        sgrids[_POSTERIOR]=_update_sgrid(sgrids[_POSTERIOR],best_datum_this[_POSTERIOR].detach(),x)


   #        msg()
   #        msg('{: >4d}|{: >9d}|{: >9.0f}|{: >9.0f}'.format(
   #            x,
   #            slooper.epoch_counter,
   #            best_datum_this[_POSTERIOR],
   #            slooper.best_datum[_POSTERIOR]
   #            )+app)

   #except KeyboardInterrupt : 
   #    pass






   #def crank(self,total_epochs=int(2e4),monitor=False,burnin=0) : 

   #    if self.dumpfilename is not None : 
   #        dumpfile=open(self.dumpfilename,'ab')

   #    if monitor :
   #        msg();

   #    if self.converge_on_n > 0 : 
   #        self.clear_lastn()

   #    for epoch in range(total_epochs) : 

   #        self.epoch_counter += 1
   #        if self.solver.scheduler is not None : 
   #            lastlr=self.solver.scheduler.get_last_lr()[0]
   #        else:
   #            lastlr=self.lr

   #        if (epoch > burnin ) and ( self.epoch_counter % self.checkevery == 0 ) : 
   #            datum=dict(epoch=self.epoch_counter) | self.solver.meta() | self.solver.snap()
   #            if self.solver.scheduler is not None : 
   #                datum.update({ 'lr' : self.solver.scheduler.get_last_lr()[0] })
   #            self.opt_data.append(
   #                datum
   #            )

   #            dumpgoods=self.solver.dump()
   #            
   #            #if self.epoch_counter > 5*self.checkevery : 
   #                #self.ofinterest[nzwta]=True
   #                
   #            if self.dumpfilename is not None : 
   #                pickle.dump(dumpgoods,dumpfile)

   #            thissnap=datum['posterior']
   #            if thissnap>self.best_posterior : 
   #                
   #                self.best_posterior = thissnap
   #                self.best_datum=datum
   #                self.best_params=self.solver.fwargs()

   #            if monitor  > 0: 
   #                if self.converge_on_n > 0 : 
   #                    delta=thissnap-self.lastn[-1]
   #                else : 
   #                    delta=0.0

   #                mstr='    Epoch {: >6} : {: >6.3e} {: >-8.2f} {: >-8.2f} {: >-4.5f}'.format(self.epoch_counter,thissnap,thissnap-self.best_posterior,delta,lastlr)
   #                time.sleep(0)
   #                msg(mstr,end='\r')
   #                
   #            if epoch > burnin and self.converge_on_n > 0 : 
   #                #if (thissnap <= self.lastn).all() or\
   #                       # ((self.best_posterior > thissnap ) and\
   #                       # ( self.best_posterior < (self.lastn+1.0)).all()) :
   #                if self.best_posterior > thissnap and self.lastn.std() < 0.5 :
   #                    if monitor : 
   #                        msg()
   #                        msg('    Converged at epoch {}.'.format(self.epoch_counter))
   #                    break
   #                    
   #                self.lastn[:-1]=self.lastn[1:].clone()
   #                self.lastn[-1]=thissnap

   #        if lastlr < 1e-2 : 
   #            self.solver.tick()
   #        else : 
   #            self.solver.tick(noise=True)

   #    if self.dumpfilename is not None : 
   #        dumpfile.close()


   #def _update_sgrid(sgrid,new) :

   #    sgrid[:-1]=sgrid[1:].clone()
   #    sgrid[-1]=new.clone()
   #    return sgrid
