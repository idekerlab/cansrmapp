#!/usr/bin/env python
"""
Build matrices and tensors for CanSRMaPP Model solving.
"""
if __name__ == '__main__'  :
    import argparse
    parser=argparse.ArgumentParser()
    exgrp=parser.add_mutually_exclusive_group(required=True)
    exgrp.add_argument('--json',action='store',help='JSON file containing settings fields. Choosing this will ignore all other arguments.')
    exgrp.add_argument('--omics_path',action='store',help='file path of omics .csv')
    parser.add_argument('--signature_path',action='store',help='file path of signature .csv',required=False)
    parser.add_argument('--length_timing_path',action='store',help='file path of length/timing hdf',required=False)
    parser.add_argument('--output_path',action='store',help='directory in which to store output (will be created if necessary)',default='.')
    parser.add_argument('--sm_path',action='store',help='file path of systems map .pickle',required=False)
    parser.add_argument('--no_arm_pcs',action='store_true',help='if selected, arm pc features will not be incorporated',default=False)
    parser.add_argument('--signature_sparsity',action='store',help='minimum portion of patients with signature activity for signature incorporation',default=0.0)
    parser.add_argument('--spoof_seed',
                            required=False,
                            action='store',
                            default='orig',
                            help='random seed for spoofing. Can be created from a (memorable) string. `orig` (default) signifies no randomization'
                            )
    parser.add_argument('--force_zero_path',
                        action='store',
                        help='file path of pickled gene set for which to force zero values in the genomic background tensor J. (not used)',
                        required=False,
                        )
    parser.add_argument('--blacklist_path',
                        action='store',
                        help='file path of pickled gene set to exclude from consideration entirely (in publication: olfactory receptor genes)',
                        required=False,
                        )
    parser.add_argument('--spoof_smsize',
                        required=False,
                        action='store',
                        default=-1,
                        help='size of systems map to be spoofed. -1 [default] copies the size of the template map'
                        )
    parser.add_argument('--arm_quotient',
                        required=False,
                        action='store',
                        default=0.95,
                        help='For synteny broadcasting, requires that a broadcasting gene'+\
                            'must have this value or more of the maximum alteration count on that arm.'
                        )
    parser.add_argument('--critical_synteny_quantile',
                        required=False,
                        action='store',
                        default=0.80,
                        help='For synteny broadcasting, requires that a broadcasting gene'+\
                            'must be at this quantile or greater for the relevant alteration.'
                        )
    parser.add_argument('--normalize_synteny',
                        required=False,
                        action='store_true',
                        default=False,
                        help='When guessing parameters, normalize synteny (identical to cmsolver argument)'
                        )
    ns=parser.parse_args()

import cansrmapp
from cansrmapp import pd
from cansrmapp import np
from cansrmapp import torch
from cansrmapp import random
import cansrmapp.utils as cmu
from cansrmapp.utils import _tcast
import cansrmapp.cmbioinfo as cmbi
from dataclasses import dataclass
import scipy
import scipy.sparse
import json
import os
import sys
import time
import pickle
import numpy.typing as npt
import typing
import warnings

from functools import reduce

opj=os.path.join

ETYPES=('mut',
'fus','up','dn')

SYSTEM_LIMIT_UPPER=257
SYSTEM_LIMIT_LOWER=4


@dataclass(kw_only=True)
class BuilderSettings(object) : 
    omics_path : str
    signature_path : str
    length_timing_path : str
    output_path : str = '.'
    sm_path : str = ''
    force_zero_path : str = ''
    blacklist_path : str = ''
    no_arm_pcs : bool=False
    signature_sparsity : float=0.0
    spoof_seed : typing.Union[str,int] = 'orig'
    spoof_smsize : int=-1
    arm_quotient : float=0.95
    critical_synteny_quantile : float=0.80
    normalize_synteny : bool=False
    

def read_omics(a) :
    if type(a) == str :
        return pd.read_csv(a,index_col=0)
    if type(a) == BuilderSettings :
        return pd.read_csv(a.omics_path,index_col=0)
    raise ValueError("a should be either a string(=file path) or a BuilderSettings object")

def read_signatures(a) :
    sigframe=pd.read_csv(a.signature_path,index_col=0)
    sigframe=sigframe[ sigframe.columns[(sigframe>0).sum(axis=0) > a.signature_sparsity*sigframe.shape[0]]]
    return sigframe
    #raise ValueError("a should be either a  a BuilderSettings object")

def get_valid_gene_ids(blacklist=None) :
    if blacklist is None : blacklist=set()
    cmbi.boot_bioinfo()
    return cmbi._gi[
                    cmbi._gi.type_of_gene.eq('protein-coding') &
                    ~cmbi._gi.Ensembl.isnull() &
                    ~cmbi._gi.GeneID.isin(blacklist) ].GeneID.unique()


def read_length_timing(a,master_gene_index=None) : 
    if type(a) == str : length_timing_path =a
    if type(a) == BuilderSettings : length_timing_path=a.length_timing_path

    dfrt=pd.read_hdf(length_timing_path,key='dfrt')
    if master_gene_index is not None : 
        dfrt=dfrt.reindex(master_gene_index)
    dfrt=dfrt.fillna(dict(length=6.8e4,timing=2.4))
    lengths=dfrt.length
    lengths=(lengths-lengths.min())/(lengths.max()-lengths.min())
    timings=dfrt.coord
    timings=(timings-timings.min())/(timings.max()-timings.min())

    return lengths,timings

def align_pandas(*args,how='intersect') :
    if how == 'intersect'  :
        joint_index=reduce(np.intersect1d,[a.index for a in args])
    elif how == 'union'  : 
        joint_index=reduce(np.union1d,[a.index for a in args])
    else : 
        raise ValueError("<how> must be 'intersect' or 'union'")

    return tuple([ a.reindex(joint_index) for a in args ])

def stratify_signatures(signatures) :
    sigs_for_cn_fus=[ c for c in signatures.columns if any([ c.startswith(pref) for pref in (['CN','arm_pc_'])]) ]
    sigs_for_muts=np.setdiff1d(signatures.columns,sigs_for_cn_fus)
    sig_indices_for_cn_fus=torch.tensor(np.argwhere(np.isin(signatures.columns,sigs_for_cn_fus)).ravel())
    sig_indices_for_muts=torch.tensor(np.argwhere(np.isin(signatures.columns,sigs_for_muts)).ravel())

    return sig_indices_for_muts,sig_indices_for_cn_fus

def read_eidpickle(pickle_file_path) :
    with open(pickle_file_path,'rb') as f : 
        return pickle.load(f)

def omics_to_nonzeros(omics_frame,valid_gene_ids,coerce_ids=False): 
    o2=omics_frame.transpose().copy()
    o2['GeneID']=np.vectorize(lambda s : s.split('_')[0])(o2.index)
    o2['etype']=np.vectorize(lambda s : s.split('_')[1])(o2.index)
    o2=o2.set_index(['etype','GeneID'])
    subomics={ etype : o2.xs(etype,level=0) for etype in ETYPES}

    if coerce_ids : 
        master_gene_index=valid_gene_ids
    else :
        master_gene_index=np.intersect1d(
            reduce(np.union1d,[ v.index for v in subomics.values()]),
            valid_gene_ids)

    subomics={ etype :  subomics[etype].reindex(master_gene_index).fillna(0.0) for etype in ETYPES }
    nzomics={ etype : subomics[etype].values.nonzero() for etype in subomics }

    return master_gene_index,nzomics,subomics

#DEPREC
def get_matching_chromosome_tensor(master_gene_index) : 
    cmbi.boot_gff()
    ggis=cmbi._gff.copy().set_index('GeneID')['chromosome'].reindex(master_gene_index).fillna('')
    chrmatch=torch.tensor((ggis.values[:,None]==ggis.values),dtype=torch.bool).to_sparse_coo()
    return chrmatch

def get_arm_vec(master_gene_index) : 
    if _dfarm is None : 
        _prep_arm_relative(master_gene_index)
    return _dfarm.copy().set_index('GeneID')['arm'].reindex(master_gene_index).fillna('')

def get_matching_arm_tensor(master_gene_index) : 
    if _dfarm is None : 
        _prep_arm_relative(master_gene_index)

    ggis=get_arm_vec(master_gene_index)
    chrmatch=torch.tensor((ggis.values[:,None]==ggis.values)&(ggis.values!=''),dtype=torch.bool).to_sparse_coo()
    return chrmatch

_dfarm=None

@np.vectorize
def getarm(map_location) : 
    x=0
    while x < len(map_location) : 
        if map_location[x] == 'p' or map_location[x] == 'q':
            return map_location[:x+1]
        x+=1
    return None

def _prep_arm_relative(master_gene_index) :
    global _dfarm
    _dfarm=cmbi._gi.query('GeneID in @master_gene_index')[['GeneID','Symbol','map_location']].copy()
    _dfarm['arm']=getarm(_dfarm.map_location)

def get_synteny_I(cnaframe,matching_chromosome_tensor,npats,master_gene_index,by='focal',arm_quotient=0.95,critical_synteny_quantile=0.9): 

    global _dfarm

    cnaframe=cnaframe.copy().transpose()
    cnaframe=cnaframe.reindex(columns=master_gene_index).fillna(0)
    cnasum=cnaframe.sum(axis=0)

    scnat=torch.tensor(cnaframe.values).to_sparse_coo().float()
    ccnat=cmu.regularize_cc_torch(cmu.cc2tens(scnat,scnat),npats,correlation_p=1e-3)

    if by == 'focal' : 
        if _dfarm is None : 
            _prep_arm_relative(master_gene_index) 
        assert type(master_gene_index) is not type(None)

        # total copy alterations
        localarm=_dfarm.set_index('GeneID').join(cnasum.rename('cna')).reset_index()

        # maximum copy alteration count by arm
        dfagb=localarm.groupby('arm')[['cna']].max().rename(columns=lambda x : x+'_max').reset_index()

        # putting arm max in another columns
        localarm=localarm.merge(dfagb,on='arm',how='left')

        # relative frequency of alteration 
        localarm['quo']=localarm.cna/np.clip(localarm.cna_max,1,np.inf)

        # 90th percentile frequency _OVERALL_ for this event
        # "localarm" is not just this arm
        critical=np.quantile(localarm.cna,critical_synteny_quantile)

        eligible=localarm.query('quo > @arm_quotient and cna > @critical').GeneID.unique()
        ccnat[:,~np.isin(master_gene_index,eligible)]=0.0
    elif by == 'quantile' : 
        cnacut=np.quantile(cnasum,0.99)
        lowcna=np.argwhere(cnasum.values<cnacut).ravel()
        ccnat[:,lowcna]=0
    else : 
        raise ValueError('"by" must be "focal" or "quantile"')

    ccnat.fill_diagonal_(1.0)
    cnamask=(ccnat*matching_chromosome_tensor).coalesce()
    cna_metamask=torch.argwhere(cnamask.values()>torch.finfo(cnamask.dtype).eps).ravel()
    synteny_I_slice=torch.sparse_coo_tensor(
        values=cnamask.values()[cna_metamask],
        indices=cnamask.indices()[:,cna_metamask],
    ).coalesce()

    return synteny_I_slice

def get_cofusion_I(fusframe,npats,master_gene_index) : 
    fusframe=fusframe.copy().transpose().reindex(columns=master_gene_index)
    fussum=fusframe.sum(axis=0)
    eligible=fussum[fussum.gt(1.0)].index
    sfust=torch.tensor(fusframe.values)
    cfust=cmu.regularize_cc_torch(cmu.cc2tens(sfust,sfust),npats,correlation_p=1e-3)   
    cfust[:,~np.isin(master_gene_index,eligible)]=0.0
    cfust.fill_diagonal_(1.0)

    return cfust.to_sparse_coo()

    return cfust.to_sparse_coo()

def eightpeetwenty() : 
    return np.intersect1d(cmbi._gi[
                                    cmbi._gi.map_location.str.startswith('8p20') | 
                                    cmbi._gi.map_location.str.startswith('8p21') | 
                                    cmbi._gi.map_location.str.startswith('8p22') | 
                                    cmbi._gi.map_location.str.startswith('8p23') | 
                                    cmbi._gi.map_location.str.startswith('8p24') 
                                    ].GeneID.unique(),master_gene_index)

def create_sparse_omics(nzomics,osize) : 
    tindices=np.concatenate([
        np.stack([
            np.ones((len(nzomics[etype][0]),),dtype=int)*x,
            *nzomics[etype],
                 ],
                 axis=-1
                )
    for x,etype in enumerate(ETYPES)
    ],axis=0).transpose()

    sparse_omics=torch.sparse_coo_tensor(indices=tindices,
                                    values=torch.ones(tindices.shape[1]),
                                    dtype=torch.float32,
                                    size=(4,osize,tindices[-1].max()+1),
                                   ).coalesce()
    return sparse_omics

def create_J_and_gbindex(sparse_omics,signatures,lengths,timings,master_gene_index=None,add_eightpeetwenty=False,force_zero_indices=None) : 

    sig_indices_for_muts,sig_indices_for_cn_fus=stratify_signatures(signatures)
    tsig=torch.tensor(signatures.values,dtype=torch.float32)

    protojslices=[ cmu.regularize_cc_torch(
                    cmu.cc2tens(
                        sparse_omics.to_dense()[x].transpose(0,1),tsig
                    ),
                    n=torch.tensor(signatures.shape[0]),
                    correlation_p=1e-3) 
                    for x in range(sparse_omics.shape[0]) ]
    J=torch.clip(torch.stack(protojslices,axis=0),0.0,1.0)
    #J[0,sig_indices_for_cn_fus]=0.0
    #J[1:,sig_indices_for_muts]=0.0
    J[0,:,sig_indices_for_cn_fus]=0.0
    J[1:,:,sig_indices_for_muts]=0.0

    if force_zero_indices is not None : 
        J[:,force_zero_indices,:]=0.0

    if add_eightpeetwenty : 
        assert master_gene_index is not None
        eightpeetwenties=eightpeetwenty()
        gb_index=np.r_[signatures.columns,['8p2x','length','timings']]
        extrablock=torch.zeros((J.shape[0],J.shape[1],3))
        extrablock[3,:,0]=torch.tensor(np.isin(master_gene_index,eightpeetwenties))
        extrablock[0,:,1]=torch.tensor(lengths.values)
        extrablock[0,:,2]=torch.tensor(timings.values)
    else :
        gb_index=np.r_[signatures.columns,['length','timings']]
        extrablock=torch.zeros((J.shape[0],J.shape[1],2))
        extrablock[0,:,0]=torch.tensor(lengths.values)
        extrablock[0,:,1]=torch.tensor(timings.values)

    J=torch.concatenate([J,extrablock],axis=2)
    J[torch.isnan(J)]=0.0


    return J,gb_index


def create_I(s) : 
    return torch.diag(torch.ones((s,),dtype=torch.float32))

def create_synteny_broadcast(synteny_up,synteny_dn,cofusion=None) : 
    ii=create_I(synteny_up.shape[0])
    if cofusion is None :
        synteny_broadcast=torch.stack([ii,ii,synteny_up.to_dense(),synteny_dn.to_dense()],axis=0).to_sparse_coo()
    else : 
        synteny_broadcast=torch.stack([ii,cofusion.to_dense(),synteny_up.to_dense(),synteny_dn.to_dense()],axis=0).to_sparse_coo()
    return synteny_broadcast

def create_null_sm() : 
    return torch.tensor([]),np.array([])

def load_sm(a,master_gene_index) : 
    if type(a) == str : 
        sm_path=a
    elif type(a) == BuilderSettings : 
        sm_path=a.sm_path
    else : 
        raise ValueError("a should be either a string(=file path) or a BuilderSettings object")

    with open(sm_path,'rb') as f: 
        nh0=pickle.load(f)

    nh1=dict()
    sys_index=list()
    for k,v in nh0.items() :
        newv=set(v) & set(master_gene_index)
        if len(newv) >= SYSTEM_LIMIT_LOWER and len(newv) < SYSTEM_LIMIT_UPPER :
            nh1.update({ k : newv})
            sys_index.append(k)

    return nh1,sys_index

def sm_to_tensor(systems_map_dict,system_index,master_gene_index) : 
    nsargs=0
    nzi=list()
    nzc=list()
    nzv=list()

    glookup=dict(zip(master_gene_index,np.arange(master_gene_index.shape[0])))
    for k in system_index  : 
        v=systems_map_dict[k]
        nzc.extend([nsargs]*len(v))
        nzi.extend([ glookup.get(nvg) for nvg in v ])
        nzv.extend([ 1/np.sqrt(np.cast['float'](len(v))/float(SYSTEM_LIMIT_LOWER))]*len(v))
        nsargs +=1 

    H=np.zeros((len(master_gene_index),len(system_index)),dtype=np.float32)
    H[nzi,nzc]=nzv
    return torch.tensor(H)
    

def spoof(sm,preserve_rowsums=True,seed=2749463602,smsize=None,replace=False) : 
    """
    Generates "spoofed" systems maps from an original dict-format systems map `sm`.
    Preserving rowsums means that in addition to the systems having the same
    size distribution, each gene will appear (approximately) the same number of 
    times in the systems map.
    """

    systemsizes=[ len(v) for k,v in sm.items() ]
    if smsize is None or smsize < 0  : 
        smsize=len(systemsizes)
    memberships=[ vi for k,v in sm.items() for vi in v ]

    gen=np.random.RandomState(np.random.MT19937(seed=seed))

    if smsize is None and preserve_rowsums : 
        mq=list(gen.choice(memberships,size=len(memberships),replace=False))
    else: 
        mq=list(gen.choice(list(set(memberships)),size=sum(systemsizes)*2,replace=True))
    
    outsm=dict()
    sysind=0

    if smsize is None : 
        sizepicks=gen.choice(systemsizes,size=len(systemsizes),replace=replace)
    else : 
        sizepicks=gen.choice(systemsizes,size=smsize,replace=True)
    
    sysnames=list()
    for syssize in sizepicks : 
        
        #syssize=systemsizes.pop()
        newsyscontents=set()
        for i in range(syssize) : 
            if len(mq) < 1 : mq=list(np.random.choice(memberships,size=len(memberships),replace=False))
            #240408 : added to prevent "draining" of systems during spoofing
            newsyscontents.add(mq.pop())
                
        sn='sys{:0>5}'.format(sysind)
        sysnames.append(sn)
        outsm.update({sn : newsyscontents })
        sysind += 1
        
    return outsm,sysnames

def torch_sparse_to_scipy_sparse(tst) : 

    if not tst.is_sparse :
        tst=tst.to_sparse_coo()

    return scipy.sparse.csr_matrix(
        (
            tst.values().cpu().numpy(),
            tst.indices().cpu().numpy(),
        ),shape=tst.shape)

def generate_guesser_tensor(*args) :

    tensors=[ a.to_dense() for a in args ]
    return torch.concatenate(tensors+[torch.ones((tensors[0].shape[0],1)),],axis=1).to_sparse_coo()


def create_guess_weights(I,H,J,synteny_broadcast,sparse_omics,n,normalize_synteny=False) :
    import cansrmapp.cmsolver as cms
    from cansrmapp.utils import _tcast
    from cansrmapp import DEVICE

    realy=sparse_omics.to_dense().sum(axis=2)
    y=torch.special.logit((realy+1)/n)

    guesses=list()
    for x in range(4) : 
        sX=torch_sparse_to_scipy_sparse(generate_guesser_tensor(I,H,J[x]))
        lrr=scipy.sparse.linalg.lsmr(sX,y[x].cpu().numpy())
        guesses.append(lrr[0])

    guessmat=np.stack(guesses,axis=0)
    baseguess=guessmat.max(axis=0)[:-1]

    if len(H.shape) < 2 : 
        ihlen=I.shape[1]
    else : 
        ihlen=I.shape[1]+H.shape[1]

    guessi=np.clip(baseguess[:I.shape[1]],0,np.inf)
    guessh=np.clip(baseguess[I.shape[1]:ihlen],0,np.inf)
    guessj=np.clip(baseguess[ihlen:],0,np.inf)
    guessint=_tcast(guessmat[:,-1])

    solver=cms.Solver(
        I=I.to(DEVICE),
        H=H.to(DEVICE),
        J=J.to_sparse_coo().to(DEVICE),
        y=realy.ravel().to(DEVICE),
        npats=torch.tensor(float(n)).to(DEVICE),
        init_iweights=_tcast(guessi).float().to(DEVICE),
        init_hweights=_tcast(guessh).float().to(DEVICE),
        init_jweights=_tcast(guessj).float().to(DEVICE),
        #init_jweights=torch.ones(J.shape[-1],device=DEVICE,dtype=torch.float32),
        init_intercept=guessint.float().to(DEVICE),
        synteny_broadcast=synteny_broadcast,
        init_loglambdas=_tcast([0.5,0.5]).float(),
        init_partition=None,
        partition_alpha=float(1.0),
        lr=5e-2,
        #lr=1e-2,
        #lr=1e-3,
        schedule=True,
        optimizer_method='adam',
    )

    slooper_lax=cms.Slooper(solver)
    slooper_lax.crank(monitor=True)
    bp,bd,df=slooper_lax.wrap()

    return bp


event2eid=lambda c: c.split('_')[0]

if __name__ == '__main__' : 


    #print('Numpy:',np.random.get_state())
    #print('Torch:',torch.get_rng_state())
    #print('Torch (CUDA):',torch.cuda.get_rng_state())
    #print('Random:',random.getstate())

    if ns.json is not None : 
        with open(ns.json,'r') as f : 
            settings=BuilderSettings(**json.load(f))
    else : 
        settings=BuilderSettings(
            omics_path=ns.omics_path,
            signature_path=ns.signature_path,
            length_timing_path=ns.length_timing_path,
            output_path=ns.output_path,
            no_arm_pcs=ns.no_arm_pcs,
            sm_path=ns.sm_path,
            force_zero_path=ns.force_zero_path,
            blacklist_path=ns.blacklist_path,
            spoof_seed=ns.spoof_seed,
            spoof_smsize=int(ns.spoof_smsize),
            signature_sparsity=float(ns.signature_sparsity),
            arm_quotient=float(ns.arm_quotient),
            normalize_synteny=ns.normalize_synteny,
            )

    
    if settings.blacklist_path is not None : 
        cmu.msg('Reading blacklist file...')
        blacklist=read_eidpickle(settings.blacklist_path)
    else :
        blacklist=None

    cmu.msg('Getting valid gene ids...')
    valid_gene_ids=get_valid_gene_ids(blacklist)
    cmu.msg('Done.')

    cmu.msg('Reading omics...')
    omics=read_omics(settings)
    cmu.msg('Done.')
    cmu.msg('Reading signatures...')
    signatures=read_signatures(settings)
    if settings.no_arm_pcs :
        signatures=signatures[signatures.columns[~signatures.columns.str.startswith('arm_pc_')]]

    # as far as I can tell SigProfiler is _not_ handling artifact signatures during table generation
    # therefore,
    cols2drop=np.intersect1d(signatures.columns,['CN22','CN23','CN24'])
    if len(cols2drop) > 0  : 
        signatures=signatures.drop(columns=cols2drop)

    cmu.msg('Done.')
    cmu.msg('Aligning omics and signatures...')
    omics,signatures=align_pandas(omics,signatures,how='intersect')
    cmu.msg('Done.')
    cmu.msg('Reformatting omics...')
    master_gene_index,nzomics,subomics=omics_to_nonzeros(omics,valid_gene_ids)
    cmu.msg('Done.')
    cmu.msg('Getting lengths and timings...')
    lengths,timings=read_length_timing(settings,master_gene_index)
    cmu.msg('Done.')

    cmu.msg('Making sparse omics...')
    sparse_omics=create_sparse_omics(nzomics,osize=len(master_gene_index))
    cmu.msg('Done.')

    if settings.force_zero_path is not None  and settings.force_zero_path != '': 
        cmu.msg('Reading force-zero file...')
        force_zero=read_eidpickle(settings.force_zero_path)
        force_zero_indices=np.argwhere(np.isin(master_gene_index,force_zero)).ravel()
        cmu.msg('Done.')
    else :
        force_zero_indices=None


    cmu.msg('Creating J...')
    J,gb_index=create_J_and_gbindex(sparse_omics,signatures,lengths,timings,master_gene_index=master_gene_index,add_eightpeetwenty=False,force_zero_indices=force_zero_indices)
    cmu.msg('Done.')

    cmu.msg('Creating synteny broadcast...')
    #mct=get_matching_chromosome_tensor(master_gene_index)
    mct=get_matching_arm_tensor(master_gene_index)
    siup=get_synteny_I(subomics['up'],
                       mct,
                       npats=omics.shape[0],
                       master_gene_index=master_gene_index,
                       arm_quotient=settings.arm_quotient,
                       critical_synteny_quantile=settings.critical_synteny_quantile)

    sidn=get_synteny_I(subomics['dn'],
                       mct,
                       npats=omics.shape[0],
                       master_gene_index=master_gene_index,
                       arm_quotient=settings.arm_quotient,
                       critical_synteny_quantile=settings.critical_synteny_quantile)

    sifus=get_cofusion_I(subomics['fus'],npats=omics.shape[0],master_gene_index=master_gene_index)
    #TODO something default option
    synteny_broadcast=create_synteny_broadcast(siup,sidn,sifus)
    cmu.msg('Done.')

    cmu.msg('Creating I...')
    I=create_I(len(master_gene_index)).to_sparse_coo()
    cmu.msg('Done.')

    cmu.msg('Creating H...')
    if settings.spoof_seed == "null" : 
        cmu.msg('    which is understood to be empty')
        H,system_index=create_null_sm()
    else : 
        cmu.msg('    Reading...')
        smd,system_index=load_sm(settings,master_gene_index)
        if settings.spoof_seed != "orig" : 
            cmu.msg('    Spoofing...')
            smd,system_index=spoof(smd,seed=cmu.word_to_seed(settings.spoof_seed),smsize=settings.spoof_smsize)
            cmu.msg('    Done.')

        cmu.msg('    Tensorizing...')
        H=sm_to_tensor(smd,system_index,master_gene_index)
        cmu.msg('    Done.')
    cmu.msg('Done.')

    cmu.msg('Preparing guess weights (this can take some time)')

    #guessbp=create_guess_weights(I,H,J,synteny_broadcast,sparse_omics,subomics['mut'].shape[0],normalize_synteny=settings.normalize_synteny)

    cmu.msg('Saving...')
    os.makedirs(settings.output_path,exist_ok=True)
    torch.save(I,opj(settings.output_path,'I.pt'))
    torch.save(H.to_sparse_coo(),opj(settings.output_path,'H.pt'))
    torch.save(J.to_sparse_coo(),opj(settings.output_path,'J.pt'))
    torch.save(sparse_omics,opj(settings.output_path,'sparse_omics.pt'))
    torch.save(synteny_broadcast,opj(settings.output_path,'synteny_broadcast.pt'))
    #torch.save({ k : v.detach().cpu() for k,v in guessbp.items() },opj(settings.output_path,'guessbp.pt'))

    np.savez(opj(settings.output_path,'arrays.npz'),gene_index=master_gene_index,sys_index=system_index,sig_index=gb_index)
    cmu.msg('Done.')
