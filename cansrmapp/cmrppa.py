import torch
import os
opj=os.path.join
import kidgloves as kg
kg._get_geneinfo()
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.preprocessing import RobustScaler,MaxAbsScaler,PowerTransformer,QuantileTransformer
from sklearn.model_selection import KFold,ShuffleSplit,GridSearchCV
from scipy.stats import pearsonr,spearmanr
import warnings
import pickle
import time
from scipy import odr

#root='/cellar/users/mrkelly/Data/largesse_paper/batched_runs/240619/'
#target=opj(root,'H_IAS_clixo_hidef_Nov17.edges','main','main_4:2.pt')

if __name__ == '__main__' : 

    parser=argparse.ArgumentParser()
    parser.add_argument('--runoutputpath',help='Directory with arrays.npz, run output log files, ets',action='store',required=True)
    parser.add_argument('--spoofid',help='Random seed for resampling hierarchy; "orig"[default] omits this step',action='store',required=False,default='orig')
    parser.add_argument('--omics',help='omics.csv',action='store',required=True)
    parser.add_argument('--rppa',help='rppa.tsv',action='store',required=True)
    parser.add_argument('--sig',help='signature.tsv',action='store',required=True)
    parser.add_argument('--probe',help='specific probe to model',required=True)
    parser.add_argument('--randseed',help='random seed for train/test splitting',required=False,default='Whisky')
    parser.add_argument('--pref',help='prefix for saving output.npz file',required=True)

def read_omics(opath) : 
    #omics=pd.read_csv('/cellar/users/mrkelly/Data/largesse_paper/cohort_rules/LUAD_gdc_manual_240208/omics_240228.csv',index_col=0)
    omics=pd.read_csv(opath,index_col=0)
    omics.index.name='Tumor_Sample_Barcode'

    return omics

def read_rppa(rppath) : 
    rppa=pd.read_csv(rppath).set_index('Tumor_Sample_Barcode').drop(columns='Unnamed: 0')
    rppiv_raw=rppa.reset_index().pivot_table(index='Tumor_Sample_Barcode',columns='peptide_target',values='protein_expression')

    ratios=list()
    probes=rppiv_raw.columns
    for probe in rppiv_raw.columns : 
        if '_p' not in probe: continue
        apoprotein=probe.split('_')[0]
        phosphos='_'.join(probe.split('_')[1:])
    
        if apoprotein in probes : 
            ratios.append((apoprotein,probe))

    for rt in ratios : 
        # ratio tuple -- first value is apoprotein, second is 
        # phospho probe. Ratio values are phosopho - apo,
        # so higher values are increased phosphorylation
        rppiv_raw['|'.join(rt)]=rppiv_raw[rt[1]]-rppiv_raw[rt[0]]

    rppiv_raw.columns=[ c.replace('_','-') for c in rppiv_raw.columns ]

    return rppiv_raw


def prep_frames(probe,**kwargs) : 

    rppa=kwargs['rppiv']
    omics=kwargs['omics']
    signatures=kwargs['signatures']
    featurenames=kwargs['featurenames']
    featuretypes=kwargs['featuretypes']
    systransform=kwargs['systransform']

    sigfeats=featurenames[ featuretypes == 'signature' ]
    wsig=signatures[sigfeats]

    if type(probe) == str  : 
        rf=rppa[probe].dropna().copy()
        sharepats=np.intersect1d(omics.index,rf.index)
        rf=rf.reindex(sharepats)
    #elif type(probe) == tuple : 
        #apo=probe[1]
        #phospho=probe[0]
    else :
        raise ValueError('Bad argument for rppa, which was {}'.format(rppa))

    oo=omics.reindex(sharepats)
    ot=pd.DataFrame(data=(np.dot(oo.values,systransform)>0),index=oo.index,columns=featurenames[np.isin(featuretypes,['gene','system'])])
    #ot=pd.DataFrame(data=MaxAbsScaler().fit_transform(np.dot(oo.values,systransform)),index=oo.index,columns=featurenames[np.isin(featuretypes,['gene','system'])])
    of=ot.join(wsig,how='left')

    return rf,of

def from_pickle(picklefilename) : 

    with open(picklefilename,'rb') as f : 
        hier=pickle.load(f) 

    alleids=list({ e for k,v in hier.items() for e in v })
    from itertools import product
    eventtups=list(product(alleids,['_mut','_fus','_up','_dn']))
    r_genes=alleids*4

    frame=pd.DataFrame(
            index=[ '_'.join(et) for et in eventtups ],
            columns=list(hier.keys()),
            data=np.zeros((len(eventtups),len(hier))),
    )

    for k,v in hier.items() :
        for e in v : 
            rows_to_alter=np.argwhere(r_genes == e)
            frame.iloc[rows_to_alter,k]=True

    return frame
    # TODO : this should take a pickle _already pruned_ etc
    # and return its values as a matrix and its events as ylabels
    # it should broadcast all genes to all four event types, and allow a reduction into
    # omics-space to handle discrepancies later

hasmut=lambda s : s.endswith('_mut')
hasfus=lambda s : s.endswith('_fus')
hasup=lambda s : s.endswith('_up')
hasdn=lambda s : s.endswith('_dn')


def resolve_omics_frame_ylabel_clash(omics_ylabels,frame_ylabels) : 

    muts=np.union1d(omics_ylabels,frame_ylabels[np.vectorize(hasmut)(frame_ylabels)])
    ups=np.intersect1d(omics_ylabels,frame_ylabels[np.vectorize(hasup)(frame_ylabels)])
    dns=np.intersect1d(omics_ylabels,frame_ylabels[np.vectorize(hasdn)(frame_ylabels)])
    fuss=np.intersect1d(omics_ylabels,frame_ylabels[np.vectorize(hasfus)(frame_ylabels)])

    return np.r_[ muts,ups,dns,fuss]


def get_nbinom_fit(eventcts) : 
    u=np.mean(eventcts)
    s2=np.var(eventcts)
    nbp=u/s2
    nbr=u*nbp/(1-nbp)

    from scipy.stats import nbinom

    mynbinom=nbinom(nbr,nbp)

    wts=mynbinom.pmf(eventcts)
    ps=wts/wts.sum()
    from scipy.stats import nbinom
    mynbinom=nbinom(nbr,nbp)
    return mynbinom

def word_to_seed(word) :
    import hashlib
    return int(hashlib.shake_128(word.encode()).hexdigest(4),16)


def main_script_input_handling(omicspath,rppapath,signaturepath,runoutputpath) : 

    omics=read_omics(omicspath)
    signatures=pd.read_csv(signaturepath,index_col=0)
    signatures=pd.DataFrame(data=MaxAbsScaler().fit_transform(signatures),
                        index=signatures.index,
                        columns=signatures.columns,
                    )
    rppiv_raw=read_rppa(rppapath)

    if runoutputpath == 'allgenes' : 

        esums=omics.sum(axis=0)
        esums=esums[ esums > 10]
        ylabels=esums.index

        ugenes=np.unique(np.vectorize(lambda s : s.split('_')[0])(ylabels))
        print(len(ugenes))
        ugene2col=dict(zip(ugenes,np.arange(len(ugenes))))
        event2gene=dict(zip(ylabels,np.vectorize(lambda s : s.split('_')[0])(ylabels)))
        event2row=dict(zip(ylabels,np.arange(len(ylabels))))

        systransform=np.zeros((len(ylabels),len(ugenes)))
        row_indices=np.arange(len(ylabels))
        col_indices=np.vectorize(lambda e : ugene2col.get(event2gene.get(e)))(ylabels)
        systransform[row_indices,col_indices]=1 

        featuretypes=np.array(['gene']*len(ugenes))
        featurenames=ugenes

    elif runoutputpath.lower().startswith('top') : 

        topno=int(runoutputpath[3:])

        esums=omics.sum(axis=0).sort_values(ascending=False)
        ylabels=esums.index[:topno]

        ugenes=np.unique(np.vectorize(lambda s : s.split('_')[0])(ylabels))
        print(len(ugenes))
        ugene2col=dict(zip(ugenes,np.arange(len(ugenes))))
        event2gene=dict(zip(ylabels,np.vectorize(lambda s : s.split('_')[0])(ylabels)))
        event2row=dict(zip(ylabels,np.arange(len(ylabels))))

        systransform=np.zeros((len(ylabels),len(ugenes)))
        row_indices=np.arange(len(ylabels))
        col_indices=np.vectorize(lambda e : ugene2col.get(event2gene.get(e)))(ylabels)
        systransform[row_indices,col_indices]=1 

        featuretypes=np.array(['gene']*len(ugenes))
        featurenames=ugenes
        

    else : 
        bpd=torch.load(opj(runoutputpath,'best_posterior_solution.pt'),map_location=torch.device('cpu'))
        bpsw=bpd['weights']
        hitfeatsmask=(bpsw>0).numpy()
        hitfeatsarg=np.argwhere(hitfeatsmask).ravel()
        arrays=np.load(opj(runoutputpath,'arrays.npz'),allow_pickle=True)

        nonsigoi=( np.isin(arrays['featuretypes'],['gene','system']) & hitfeatsmask )
        featurenames=arrays['feats'][hitfeatsmask]
        featuretypes=arrays['featuretypes'][hitfeatsmask]

        ylabels=arrays['ylabels']
        systransform=torch.load(opj(runoutputpath,'spX.pt'),map_location=torch.device('cpu')).to_dense().numpy()[:,nonsigoi]

    omics=omics.reindex(columns=ylabels).fillna(0)

    rppivtransformer=RobustScaler(quantile_range=(0.1,0.9))
    #rppivtransformer=RobustScaler(unit_variance=True)
    #rppivtransformer=PowerTransformer()
    #rppivtransformer=QuantileTransformer()
    # altered 240712
    rppivdata=rppivtransformer.fit_transform(rppiv_raw)
    #rppivdata=rppiv_raw.values
    rppiv=pd.DataFrame(columns=rppiv_raw.columns,index=rppiv_raw.index,data=rppivdata)

    packet={ 'omics' : omics, 
             'rppiv' : rppiv,
             'signatures' : signatures,
             'featurenames' : featurenames, 
             'featuretypes' : featuretypes,
             'systransform' : systransform,
            }

    return packet

def do_fitting(probe,packet,randseed='Whisky') : 

    gen=np.random.RandomState(np.random.MT19937(seed=word_to_seed(randseed)))
    sur,sux=prep_frames(probe,**packet)
    indices=np.arange(sur.shape[0])
    test=gen.choice(indices,sur.shape[0]//5)
    train=indices[~np.isin(indices,test)]

    xtrain=sux.values[train,:]
    xtest=sux.values[test,:]
    ytrain=sur.values[train].astype(float)
    ytest=sur.values[test].astype(float)

    starttime=time.time()
    model=trainer(
                xtrain,
                ytrain,
            )

    elapsed=time.time()-starttime
    pred=model.predict(xtest)

    wcoef=model.coef_.copy()
    wi=model.intercept_.copy()

    pearson=pearsonr(ytest,pred)
    spearman=spearmanr(ytest,pred)

    return  {
             'probe'  : probe,
             'wcoef' : wcoef,
             'intercept' : wi , 
             'runtime' : elapsed,
             'pearson_rho' : pearson.statistic,
             'pearson_p' : pearson.pvalue,
             'spearman_rho' : spearman.statistic,
             'spearman_p' : spearman.pvalue,
             }

class ModelDummy(object) : 
    def __init__(self,coef,intercept) :
        super(ModelDummy,self).__init__()
        self.coef_=coef
        self.intercept_=intercept
    def predict(self,x) :
        return np.matmul(x,self.coef_)+self.intercept_

L1_RATIOS=1-1/np.linspace(2,20,9)
ALPHAS=np.exp(np.log(10)*np.linspace(-2,1,13))
NSPLITS=100

def trainer(X,y) : 
    with warnings.catch_warnings() : 
        warnings.simplefilter("ignore")
        gscv=GridSearchCV(
                ElasticNet(),
                param_grid={
                    'l1_ratio' : L1_RATIOS,
                    'alpha' : ALPHAS,
                    },
                refit=False,
                cv=KFold(n_splits=10,shuffle=True,random_state=word_to_seed('Yankee')),
                )
        gscv.fit(X,y)

        enet=ElasticNet(**gscv.best_params_)
        shus=ShuffleSplit(n_splits=NSPLITS,test_size=0.1,random_state=word_to_seed('Zulu'))
        cgrid=np.zeros((NSPLITS,X.shape[1]))
        igrid=np.zeros((NSPLITS,))

        for c,(tr,ti) in enumerate(shus.split(X)) : 
            enet.fit(X[tr,:],y[tr])
            cgrid[c,:]=enet.coef_.copy()
            igrid[c]=enet.intercept_.copy()

        return ModelDummy(np.median(cgrid,axis=0),np.median(igrid[c]))
    

def trainer_legacy(X,y) : 
    l1_ratios=1-1/np.linspace(2,20,9)
    with warnings.catch_warnings() : 
        warnings.simplefilter("ignore")
        model=ElasticNetCV(l1_ratio=l1_ratios,
                n_jobs=4,
                random_state=word_to_seed('Yankee'),
                cv=KFold(n_splits=5,shuffle=True,random_state=word_to_seed('Yankee')),
                )
        model.fit(X,y)
    return model 

def main_script_process(omicspath,rppapath,signaturepath,probename,runoutputpath,randseed='Whisky',monitor=True) : 

    packet=main_script_input_handling(omicspath,
            rppapath,
            signaturepath,
            runoutputpath=runoutputpath,
    )

    omics=packet['omics']
    rppiv=packet['rppiv']
    featurenames=packet['featurenames']
    systransform=packet['systransform']
    signatures=packet['signatures']
    
    fitpacket=do_fitting(probename,packet,randseed=randseed)

    return fitpacket


if __name__ == '__main__' : 

    ns=parser.parse_args()
    opacket=main_script_process(ns.omics,ns.rppa,ns.sig,ns.probe,runoutputpath=ns.runoutputpath,randseed=ns.randseed)

    np.savez(ns.pref+'.npz',
        **opacket)

