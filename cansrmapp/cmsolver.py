import torch
from . import DEVICE
from . import np
from .utils import _tcast,mask_sparse_columns,msg
from functools import partial
from torch.distributions import Geometric,Exponential,Gamma,Binomial

class BaseLinearModel(torch.nn.Module) : 
    def __init__(self,nfeats,
             init_vals,
             init_intercept,
             init_loglambdas,
             device=DEVICE) : 
        
        super(BaseLinearModel,self).__init__()
        
        self.relu=torch.nn.ReLU()
        self.weights=torch.nn.Parameter( _tcast(init_vals) )
        self.intercept=torch.nn.Parameter( _tcast(init_intercept) )
        self.param_loglambdas=torch.nn.Parameter(_tcast(init_loglambdas))

    def forward(self,X,return_params=False) :
        corrected_weight=self.relu(self.weights) ;
        corrected_loglambda=self.relu(self.param_loglambdas) ;

        bigdot=torch.matmul(X,corrected_weight)+self.intercept
        
        if not return_params : 
            return bigdot
        else : 
            return bigdot,corrected_weight,corrected_loglambda

class LogPosterior(torch.nn.Module):
    def __init__(self,
                 weights_gamma_alpha=1,
                 weights_gamma_beta=1,
                 lgprior=False,
                ) : 
        
        super(LogPosterior,self).__init__()
        
        self.weights_gamma_alpha = _tcast(weights_gamma_alpha )
        self.weights_gamma_beta  = _tcast(weights_gamma_beta  )
        self.lgprior=lgprior

        self.eps=torch.finfo(torch.float32).eps

        
    def likelihood(self,output_log_odds,target_event_cts,n) : 
        return Binomial(logits=output_log_odds,total_count=n).log_prob(target_event_cts).sum()
        
    def parameter_prior(self,weights,masks,working_lambdas) : 
        assert len(working_lambdas) == len(masks) 
    
        shapes=_tcast([m.sum() for m in masks ])
        lpsum=_tcast(0.0)
        for rate,mask,shape in zip(working_lambdas,masks,shapes) : 
            lpsum += torch.distributions.Exponential(rate=rate).log_prob(weights[mask]).sum()
        return lpsum

    def k_prior(self,masks,working_lambdas) : 
        
        shapes=_tcast([m.sum() for m in masks ])
        return Geometric(probs=(1.0-1.0/working_lambdas)).log_prob(shapes).sum()
    
    def expon_hyper_prior(self,working_lambdas) :
        return Gamma(concentration=self.weights_gamma_alpha,rate=self.weights_gamma_beta).log_prob(working_lambdas-1.0).sum()

    def _assemble_posterior_components( 
        self,
        output_log_odds,
        weights,
        working_loglambdas,
        masks,
        target_event_cts,
        n) : 
        """
        """
    
        working_lambdas=torch.exp(working_loglambdas)
        outpkg=dict(
            exp_hyper_prior=self.expon_hyper_prior(working_lambdas),
            param_prior=self.parameter_prior(weights,masks,working_lambdas),
            k_prior=self.k_prior(masks,working_lambdas),
            ll=self.likelihood(output_log_odds,target_event_cts,n)
        )
        outpkg.update(dict(posterior=outpkg['exp_hyper_prior']+outpkg['param_prior']+outpkg['k_prior']+outpkg['ll']))
    
        return outpkg

    def forward( 
        self,
        output_log_odds,
        weights,
        working_loglambdas,
        masks,
        target_event_cts,
        n) : 

        return self._assemble_posterior_components(
                output_log_odds,
                weights,
                working_loglambdas,
                masks,
                target_event_cts,
                n)['posterior']
        
    def decompose( 
        self,
        output_log_odds,
        weights,
        working_loglambdas,
        masks,
        target_event_cts,
        n) : 
    
        return { k : v.item() for k,v in self._assemble_posterior_components(
                output_log_odds,
                weights,
                working_loglambdas,
                masks,
                target_event_cts,
                n).items() }


class Solver(object) : 

    def __init__(self,
                X, # torch sparse tensor
                y, # weights
                npats,
                init_weights,
                init_intercept,
                masks,
                init_loglambdas,
                weights_gamma_alpha,
                weights_gamma_beta,
                lr=1e-3,
                schedule=None,
                optimizer_method='adam',
                lgprior=False,
                #weightdecay=1e-2,
                ): 

        self.X=_tcast(X)
        self.y=_tcast(y).ravel()
        self.init_weights=_tcast(init_weights)
        self.init_intercept=_tcast(init_intercept)
        self.init_loglambdas=_tcast(init_loglambdas)
        self.masks=_tcast(masks)
        self.npats=_tcast(npats)
        self.weights_gamma_alpha=weights_gamma_alpha
        self.weights_gamma_beta=weights_gamma_beta
        #self.weightdecay=weightdecay
        self.lr=lr
        self.lgprior=lgprior

        assert self.X.shape[1] == self.init_weights.shape[0]
        assert self.X.shape[0] == self.y.shape[0]

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
        return BaseLinearModel(nfeats=self.init_weights.shape[0],
                  init_vals=kwargs.get('init_vals',self.init_weights.clone()),
                  init_intercept=kwargs.get('init_intercept',self.init_intercept.clone()),
                  init_loglambdas=kwargs.get('init_loglambdas',self.init_loglambdas),
                  )

    def spawn_losser(self) : 
        return LogPosterior(
            weights_gamma_alpha=self.weights_gamma_alpha,
            weights_gamma_beta=self.weights_gamma_beta,
            lgprior=self.lgprior,
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
        scheduler=torch.optim.lr_scheduler.CyclicLR(
               optimizer=self.optimizer,
               base_lr=5e-4,
               max_lr=1e-2,
               step_size_up=2000,
               step_size_down=2000,
               cycle_momentum=False,
           )


    def fwargs(self,X=None,**kwargs) : 
        if X is None : 
            X=self.X
        olo,weights,loglambdas=self.model(X,return_params=True)
        _fwargs=dict(
                output_log_odds=kwargs.get('output_log_odds',olo),
                target_event_cts=kwargs.get('target_event_cts',self.y),
                n=kwargs.get('n',self.npats),
                weights=kwargs.get('weights',weights),
                masks=kwargs.get('masks',self.masks),
                working_loglambdas=kwargs.get('working_loglambdas',loglambdas),
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

    def meta(self,mask_keys=['system','signature'])   : 

        metad=dict()
        nwt=self.model.weights.detach()
        metad.update({  'loglambda_'+mask_keys[x] : v.item() for x,v in enumerate(self.model.param_loglambdas)  })
        metad.update({ 'n_nonzero_'+mask_keys[x] : ((nwt>0)&m).sum().cpu().numpy() for x,m in enumerate(self.masks) })

        return metad

    def dump(self,dense=False) : 
        wt=torch.clip(self.model.weights.detach(),0,torch.inf)
        inter=self.model.intercept.detach()

        if not dense : 
            nzwta=torch.nonzero(wt).ravel()
            nzwt=wt[nzwta]
            return (nzwta,nzwt),inter
        return wt, inter

    def reinit(self) : 
        self.optimizer.zero_grad()

        old_weights=torch.clip(self.model.weights.detach(),0,torch.inf)
        old_intercept=self.model.intercept.detach()
        old_loglambdas=torch.clip(self.model.param_loglambdas.detach(),0,20)

        self.spawn( init_vals=old_weights,
                    init_intercept=old_intercept,
                    init_loglambdas=old_loglambdas,
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
        self.ofinterest=torch.zeros(
                (self.solver.X.shape[1],),
                dtype=torch.bool,
                device=self.solver.X.device)

        if self.converge_on_n > 0 : 
            self.lastn=-1*torch.inf*torch.ones((self.converge_on_n,),device=DEVICE,dtype=torch.float)

        self.dumpfilename = dumpfilename

        self.opt_data=list()
        self.epoch_counter=0

    def crank(self,total_epochs=int(2e4),monitor=False) : 

        if self.dumpfilename is not None : 
            dumpfile=open(dumpfilename,'ab')

        for epoch in range(total_epochs) : 

            self.epoch_counter += 1

            if ( self.epoch_counter % self.checkevery == 0 ) : 
                datum=dict(epoch=self.epoch_counter) | self.solver.meta() | self.solver.snap()
                self.opt_data.append(
                    datum
                )

                (nzwta,nzwt),inter=self.solver.dump()
                
                if self.epoch_counter > 5*self.checkevery : 
                    self.ofinterest[nzwta]=True
                    
                if self.dumpfilename is not None : 
                    pickle.dump(((nzwta.cpu(),nzwt.cpu()),inter.cpu()),dumpfile)

                thissnap=datum['posterior']
                if thissnap>self.best_posterior : 
                    
                    self.best_posterior = thissnap
                    self.best_datum=datum
                    self.best_params={ k : v for k,v in 
                            zip(
                                ['weights','intercept','loglambdas'],
                                [
                                    torch.clip(self.solver.model.weights.detach(),0,torch.inf),
                                    self.solver.model.intercept.detach(),
                                    self.solver.model.param_loglambdas.detach(),
                                    ]
                                ) }

                if monitor : 
                    msg('    Epoch {: >6} : {: >6.3e} {: >-6.3f}'.format(self.epoch_counter,thissnap,thissnap-self.best_posterior),end='\r')
                    
                if self.converge_on_n > 0 : 
                    if (thissnap < self.lastn).all() :
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



def path_likelihood(X,y,w_path,i_path,n) :
    X=_tcast(X)
    y=_tcast(y)
    w_path=_tcast(w_path)
    i_path=_tcast(i_path)
    n=_tcast(n)

    eps=torch.finfo(w_path.dtype).eps

    noutputs=max(y.shape)
    nfeats=(set(X.shape)-{noutputs,}).pop()
    nepochs=max(i_path.shape)

    if w_path.shape[0] != nepochs : 
        w_path=w_path.transpose(0,1)


    logodds=torch.matmul(w_path,X.transpose(0,1)).to_dense() + i_path.reshape(-1,1)
    phats=torch.clip(torch.special.expit(logodds),eps,1-eps)
    comb=torch.lgamma(n+1)-torch.lgamma(y+1)-torch.lgamma(n-y+1)

    indiv_ells=(torch.log(phats)*y + torch.log(1-phats)*(n-y))
    llterm=(comb+indiv_ells).sum(axis=1)

    return llterm

def _arg_gen_core(lg,ofinterest=None) : 
    if ofinterest is None : 
        ofinterest=torch.ones((lg.X.shape[1],)).bool().to(lg.X.device)
        X=lg.X.clone()
    else : 
        #oimask=torch.isin(torch.arange(ofinterest.shape[0]).to(ofinterest.device),ofinterest).bool()
        X=mask_sparse_columns(lg.X.clone(),ofinterest).coalesce()
        
    y=lg.t_y.ravel()
    
    npatients=_tcast(lg.omics.shape[0])
    isagene=_tcast(lg.featuretypes() == 'gene')
    isasys=_tcast(lg.featuretypes() == 'system')
    isasig=( ~isagene & ~isasys)[ofinterest].ravel()
    
    masks=[ ~isasig,isasig]
    
    
    return { 'X' : X,
             'y' : y,
             'masks' : masks,
             'npats' : npatients,
             'ofinterest' : ofinterest,
           }
    
    
def solver_args(lg,ofinterest=None) :     
    
    core_args=_arg_gen_core(lg,ofinterest)
    init_weights,init_intercept=lg.guess_weights()
    ofinterest=core_args.pop('ofinterest')
    
    return core_args | { 
             'init_weights' : init_weights[ofinterest],
             'init_intercept' : init_intercept,
    }

    
def pyro_args(lg,ofinterest=None) : 
    core_args=_arg_gen_core(lg,ofinterest)
    core_args.pop('ofinterest')
    masks=core_args['masks']
    X=core_args.pop('X') 
    IH=mask_sparse_columns(X,masks[0])
    J=mask_sparse_columns(X,masks[1])
    
    return core_args | { 'IH' : IH , 'J' : J }
    
    
    


#   class _LegacyLogPosterior(torch.nn.Module) :
#       def __init__(self,
#                    weights_gamma_alpha=1,
#                    weights_gamma_beta=1,
#                    lgprior=False,
#                   ) : 
#           
#           super(_LegacyLogPosterior,self).__init__()
#           
#           self.weights_gamma_alpha = _tcast(weights_gamma_alpha )
#           self.weights_gamma_beta  = _tcast(weights_gamma_beta  )
#           self.lgprior=lgprior

#           self.eps=torch.finfo(torch.float32).eps

#           
#       def likelihood(self,output_log_odds,target_event_cts,n) : 
#           eps=torch.finfo(output_log_odds.dtype).eps
#           output_probs=torch.clip(torch.special.expit(output_log_odds),eps,1-eps)
#           log_output_probs=torch.log(output_probs)
#           k=target_event_cts
#           comb=torch.lgamma(n+1)-torch.lgamma(k+1)-torch.lgamma(n-k+1)
#           indiv_ells=(log_output_probs*k + torch.log(1-output_probs)*(n-k))
#           llterm=(comb+(indiv_ells)).sum() # this is the binomial likelihood
#           
#           return llterm
#           
#       def parameter_prior(self,weights,masks,working_lambdas) : 
#           assert len(working_lambdas) == len(masks) 
#           lamsubs=torch.zeros_like(working_lambdas)
#           for e,(m,wl) in enumerate(zip(masks,working_lambdas)) : 
#               #lamsubs[e]=(-1*wl*weights[m] + torch.log(wl)).sum()
#               lamsubs[e]=(-1*(wl+1.0)*weights[m] + torch.log(wl+1.0)).sum()
#               # 240923, making the conjugate prior actually support lambda
#           return lamsubs.sum()

#       def k_prior(self,masks,working_lambdas) : 
#          lamsubs=torch.zeros_like(working_lambdas)
#          for e,(m,wl) in enumerate(zip(masks,working_lambdas)) : 
#              lamsubs[e]=(-1*(m.sum()+1)*torch.log(wl+1) + torch.log(wl))
#               # added 1 to fix conjugate prior issue, 240923 

#              #lamsubs[e]=(-1*(m.sum()+1)*torch.log(wl) + torch.log(wl-1)) 
#               # in the discrete case(above), lambda ^ -x converges to 
#               # l/(l-1), which becomes the denominator.
#               # in the continuous case, the _denominator_ is 1/log(lambda)

#          return lamsubs.sum()
#       
#       def expon_hyper_prior(self,working_lambdas) :
#           return (self.weights_gamma_alpha*torch.log(self.weights_gamma_beta) - \
#                   torch.lgamma(self.weights_gamma_alpha) + \
#                   (self.weights_gamma_alpha-1)*torch.log(working_lambdas) - \
#                   self.weights_gamma_beta*working_lambdas ).sum()

#       def _assemble_posterior_components(self,output_log_odds,
#                   working_loglambdas,
#                   masks,
#                   target_event_cts,
#                   weights,
#                   n) : 

#           working_lambdas=torch.clip(torch.exp(working_loglambdas),1+self.eps,torch.inf)

#           if self.lgprior : 
#               ehp=self.expon_hyper_prior(working_loglambdas)
#           else : 
#               ehp=self.expon_hyper_prior(working_lambdas)

#           outpkg=dict(
#               exp_hyper_prior=ehp,
#               param_prior=self.parameter_prior(weights,masks,working_lambdas),
#               k_prior=self.k_prior(masks,working_lambdas),
#               ll=self.likelihood(output_log_odds,target_event_cts,n),
#           )
#           outpkg.update(dict(posterior=outpkg['exp_hyper_prior']+outpkg['param_prior']+outpkg['k_prior']+outpkg['ll']))

#           return outpkg

#       def forward(self,output_log_odds,
#                   working_loglambdas,
#                   masks,
#                   target_event_cts,
#                   weights,
#                   n) : 

#           return self._assemble_posterior_components(output_log_odds,
#                   working_loglambdas,
#                   masks,
#                   target_event_cts,
#                   weights,
#                   n)['posterior']
#               

#           
#       def decompose(self,output_log_odds,
#                   working_loglambdas,
#                   masks,
#                   target_event_cts,
#                   weights,
#                   n) : 

#           return { k : v.item() for k,v in self._assemble_posterior_components(output_log_odds,
#                   working_loglambdas,
#                   masks,
#                   target_event_cts,
#                   weights,
#                   n).items() }
