import torch
import os
opj=os.path.join
import argparse

#import cansrmapp.cmbioinfo as cmbi
import cansrmapp.cmpresentation as cmp
#cmbi._get_geneinfo()
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.preprocessing import RobustScaler,MaxAbsScaler,PowerTransformer,QuantileTransformer
from sklearn.model_selection import KFold,ShuffleSplit,GridSearchCV
from scipy.stats import pearsonr,spearmanr
from scipy.sparse import coo_array,csr_array
import warnings
import pickle
import time
from scipy import odr


if __name__ == '__main__' : 

    parser=argparse.ArgumentParser()

    subparsers=parser.add_subparsers(
            help='which command to run from set (build,fit)'
            )

    parser_build=subparsers.add_parser('build',
           help='build feature table'+\
           'from files stored at <omics> and <sig>,'+\
           'stored at <output>.'
    )

    parser_build.add_argument('--omics_path',
                  help='omics.csv',
                  action='store',
                  required=True)
     

    feature_source_group=parser_build.add_mutually_exclusive_group(
            required=True)

    feature_source_group.add_argument(
            '--assembly_path',
            help='path to directory containing CM assembly files.'+\
                    'if used, requires cm_output_path and'+\
                    'signature_path as well'
                    )
    parser_build.add_argument(
            '--cm_output_path',
            help='path to directory containing CM output files.'
                    )
    parser_build.add_argument('--signature_path',
                  help='signature.tsv',
                  action='store',
                  required=False)

    parser_build.add_argument(
            '--map_threshold',
            help='float: threshold below which to discard features'+\
                   'from features in an <assembly_path>. When > 0.0,'+\
                   'implicitly checks for feature sturdiness.' ,

                   default=0.00,
                    )

    feature_source_group.add_argument(
            '--panel',
            help='path to pickle file with a container of entrez ids '+\
                   'representing a gene panel')

    parser_fit=subparsers.add_parser('fit',
           help='fit model'+\
           'from files stored at <rppa_path> and <feature_table_path>,'+\
           'targeting probe <probe>,'+\
           'subsampling with random seed <randseed>,'+\
           'stored at <output>.')
    
    parser_fit.add_argument('--feature_table_path',
                            help='specific probe to model',
                            required=True)

    parser_fit.add_argument('--probe',
                            help='specific probe to model',
                            required=True)
    
    parser_fit.add_argument('--rppa_path',
                            help='rppa.tsv',
                            action='store',
                            required=True)
     
    parser_fit.add_argument('--randseed',
                            help='random seed for train/test splitting',
                            required=True,
                            default='Whisky')
    parser.add_argument('--output_path',
            help='prefix for saving output.npz file',
            required=True)
    
    ns=vars(parser.parse_args())

def read_omics(opath) : 
    #omics=pd.read_csv('/cellar/users/mrkelly/Data/largesse_paper/cohort_rules/LUAD_gdc_manual_240208/omics_240228.csv',index_col=0)
    omics=pd.read_csv(opath,index_col=0)
    omics.index.name='Tumor_Sample_Barcode'

    return omics

def read_rppa(rppath,rescale=True) : 
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

    if rescale : 
        rppivtransformer=RobustScaler(quantile_range=(0.1,0.9))
        rppivdata=rppivtransformer.fit_transform(rppiv_raw)
        rppiv=pd.DataFrame(
                columns=rppiv_raw.columns,
                index=rppiv_raw.index,
                data=rppivdata)
        return rppiv
    else : 
        return rppiv_raw

def read_signatures(signature_path) : 
    signatures=pd.read_csv(signature_path,index_col=0)
    signatures=pd.DataFrame(data=MaxAbsScaler().fit_transform(signatures),
                        index=signatures.index,
                        columns=signatures.columns,
                    )
    return signatures




hasmut=lambda s : s.endswith('_mut')
hasfus=lambda s : s.endswith('_fus')
hasup=lambda s : s.endswith('_up')
hasdn=lambda s : s.endswith('_dn')

def word_to_seed(word) :
    import hashlib
    return int(hashlib.shake_128(word.encode()).hexdigest(4),16)

def feature_table_by_cmrun(**kwargs) : 

    try : 
        omics=kwargs['omics']
        signatures=kwargs['signatures']
        assembly_path=kwargs['assembly_path']
        cm_output_path=kwargs['cm_output_path']
        map_threshold=float(kwargs.get('map_threshold',0.00))
    except KeyError as e :
        raise(e,'feature_table_by_cmrun_requires'+\
                '<assembly_path>,<cm_output_path>,'+\
                'and <signature_path> as arguments.'
              )
    from functools import reduce
    from itertools import chain
    bp_resample_targets=chain(
            *map(lambda triplet :
                 [ opj(triplet[0],t) for t in triplet[-1] if t== 'bp.pt']
                 if len(triplet[1]) == 0 else [],os.walk(cm_output_path))
            )


    bp=cmp.tload(opj(cm_output_path,'bp.pt'))
    I=cmp.tload(opj(assembly_path,'I.pt'))
    H=cmp.tload(opj(assembly_path,'H.pt'))
    J=cmp.tload(opj(assembly_path,'J.pt'))
    synteny_broadcast=cmp.tload(opj(assembly_path,'synteny_broadcast.pt'))
    arrays=np.load(opj(assembly_path,'arrays.npz'),allow_pickle=True)

    cma=cmp.CMAnalyzer(I=I,
                       H=H,
                       J=J,
                       synteny_broadcast=synteny_broadcast,
                       arrays=arrays,
                       best_params=bp)

    dfus=cma.underselection(cutoff=np.log(4))
    nz0,nz1=dfus.values.nonzero()
    selection_events=[
                        '_'.join([arrays['gene_index'][nz0[e]],cmp.ETYPES[nz1[e]]])
                        for e in range(len(nz0)) 
                    ]

    ot=omics.transpose().loc[selection_events]
    ot['gene']=np.vectorize(lambda s : s.split('_')[0] )(ot.index)
    otm=ot.groupby('gene').max()
    oframe=otm.transpose() #This is a gene-wise omics table with all genes
                            # whose events are selected for. It will be
                            # pared down
    row_indices_of_oframe_genes=np.argwhere(
            np.isin(cma.gene_index,oframe.columns)).ravel()
    
    oframe=oframe[cma.gene_index[row_indices_of_oframe_genes]]

    df=cma.feature_summary_frame(bp_resample_targets,parallel=False)
    slci=df.query('feattype=="gene"').set_index('feat').reindex(cma.gene_index)
    slch=df.query('feattype=="sys"').set_index('feat').reindex(cma.sys_index)
    slcj=df.query('feattype=="gb"').set_index('feat').reindex(cma.gb_index)

    if map_threshold > 0 : 
        worthy_i_indices=slci.is_sturdy & slci.map_estimate.gt(map_threshold)
        worthy_h_indices=slch.is_sturdy & slch.map_estimate.gt(map_threshold)
        worthy_j_indices=slcj.is_sturdy & slcj.map_estimate.gt(map_threshold)
    else : 
        worthy_i_indices=slci.map_estimate.gt(0.0)
        worthy_h_indices=slch.map_estimate.gt(0.0)
        worthy_j_indices=slcj.map_estimate.gt(0.0)

    hnpd=H.to_dense().numpy()[row_indices_of_oframe_genes][:,worthy_h_indices]
    rppai=oframe[cma.gene_index[worthy_i_indices]].copy()
    rppah=(oframe.dot(hnpd) > 0)
    rppah.columns=cma.sys_index[worthy_h_indices]
    rppaj=signatures[np.intersect1d(cma.gb_index[worthy_j_indices],signatures.columns)]
    from functools import reduce
    rppax=reduce(pd.DataFrame.join,[rppai,rppah,rppaj]).astype('float')

    return rppax

def feature_table_by_panel(omics,panel) :

    if type(panel) == str : 
        panel_path=str(panel)
        with open(panel_path,'rb') as f : 
            panel=pickle.load(f)
    else : 
        panel=list(panel)

    selection_events=list()
    for g in panel : 
        mutform=g+'_mut'
        fusform=g+'_fus'
        upform=g+'_up'
        dnform=g+'_dn'

        if mutform in omics.columns : 
            selection_events.append(mutform)
        if fusform in omics.columns : 
            selection_events.append(fusform)
        if upform in omics.columns : 
            if dnform in omics.columns : 
                if omics[upform].sum() > omics[dnform].sum() :
                    selection_events.append(upform)
                else : 
                    selection_events.append(dnform)
            else : 
                selection_events.append(upform)
        elif dnform in omics.columns :  
            selection_events.append(dnform)

    ot=omics.transpose().loc[selection_events]
    ot['gene']=np.vectorize(lambda s : s.split('_')[0] )(ot.index)
    otm=ot.groupby('gene').max()
    kosherpanel=np.intersect1d(list(panel),otm.index)
    rppax=otm.transpose()[list(kosherpanel)]

    return rppax

def read_feature_table(feature_table_path): 
    return pd.read_csv(feature_table_path,index_col=0)


def prep_frames(probe,rppa,featuretable) : 

    if type(probe) == str  : 
        rf=rppa[probe].dropna().copy()
        sharepats=np.intersect1d(featuretable.index,rf.index)
        rf=rf.reindex(sharepats)
    else :
        raise ValueError('Bad argument for rppa, which was {}'.format(rppa))

    ft=featuretable.reindex(sharepats)

    return ft,rf

def do_fitting(featuretable,probevec,randseed='Whisky') : 

    gen=np.random.RandomState(np.random.MT19937(seed=word_to_seed(randseed)))

    indices=np.arange(probevec.shape[0])
    test=gen.choice(indices,probevec.shape[0]//5)
    train=indices[~np.isin(indices,test)]

    xtrain=featuretable.values[train,:]
    xtest=featuretable.values[test,:]
    ytrain=probevec.values[train].astype(float)
    ytest=probevec.values[test].astype(float)



    starttime=time.time()
    model=trainer(
                xtrain,
                ytrain,
            )

    elapsed=time.time()-starttime
    pred=model.predict(xtest)

    wcoef=model.coef_.copy()
    wi=model.intercept_.copy()

    with warnings.catch_warnings() : 
        warnings.simplefilter('ignore')
        pearson=pearsonr(ytest,pred)
        spearman=spearmanr(ytest,pred)

    return  {
             'probe'  : probevec.name,
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

        #if False : 
        #if X.shape[1] > int(1e3) : 
        #    X=csr_array(coo_array(X))
        # this does not aid performance even in the whole-genome case

        gscv.fit(X,y)

        enet=ElasticNet(**gscv.best_params_)
        shus=ShuffleSplit(n_splits=NSPLITS,test_size=0.1,random_state=word_to_seed('Zulu'))
        cgrid=np.zeros((NSPLITS,X.shape[1]))
        igrid=np.zeros((NSPLITS,))

        for c,(tr,ti) in enumerate(shus.split(X)) : 
            enet.fit(X[tr],y[tr])
            #enet.fit(X[tr,:],y[tr])
            cgrid[c,:]=enet.coef_.copy()
            igrid[c]=enet.intercept_.copy()

        return ModelDummy(np.median(cgrid,axis=0),np.median(igrid[c]))



def model_build_process(omics_path,signature_path,output_path,**kwargs) :
    omics=read_omics(omics_path)

    if kwargs['assembly_path'] is not None and kwargs['cm_output_path'] is not None : 
        signatures=read_signatures(signature_path)
        feature_table=feature_table_by_cmrun(
                omics=omics,
                signatures=signatures,
                **kwargs
                )
    elif kwargs['panel'] is not None : 
        feature_table=feature_table_by_panel(
                omics=omics,
                panel=kwargs['panel'],
                )
    else : 
        raise ValueError('The following set of arguments '+\
                'does not point to a mode of feature table '+\
                'generation : ',kwargs)

    return feature_table


def model_fit_process(rppa_path,
                      feature_table_path,
                      probe,
                      output_path,
                      randseed='Whisky'
                      ) : 

    rppa=read_rppa(rppa_path)
    ft=read_feature_table(feature_table_path)

    featuretable,probevec=prep_frames(probe,rppa,ft)
    fitpacket=do_fitting(featuretable,probevec,randseed=randseed)

    fitpacket.update({ 'feature_table_path' : feature_table_path,
                       'feature_names' : featuretable.columns,
                      })

    return fitpacket

if __name__ == '__main__' : 

    #print(ns)
    saving_folder=os.path.split(ns['output_path'])[0]
    if not os.path.exists(saving_folder) : 
        os.makedirs(saving_folder,exist_ok=True)


    if ns.get('omics_path') is not None : 
        feature_table=model_build_process(**ns)
        feature_table.to_csv(ns['output_path'])

    else : 
        fitpacket=model_fit_process(**ns)
        np.savez(ns['output_path'],**fitpacket)

