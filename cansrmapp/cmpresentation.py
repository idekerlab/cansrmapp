import os
opj=os.path.join
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
from cansrmapp import torch
from scipy.special import expit
import colorsys
import pickle
import sys

torch.serialization.add_safe_globals(
    [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Int64DType
    ]
)

import cansrmapp.cmbioinfo as cmbi

ETYPES=('mut','fus','up','dn')


################################################################################
#   COLORS
################################################################################

class palette() : 
    def __init__(self,**kwargs) : 
        self.colornames=list()
        for k in kwargs : 
            self.colornames.append(k)
            self.__setattr__(k,kwargs[k])
    def __repr__(self) : 
            return '\n'.join(['{}={}'.format(k,getattr(self,k)) for k in self.colornames])

routes=palette(      
techne          =       '#D51D30',
filminspace     =       '#69A4BF',
materialidad    =       '#4B786C',
archivo         =       '#E3CB60',
poetika         =       '#D69C33',
circle          =       '#6F1CC7',
investigacion   =       '#756A66',
creacion        =       '#B4BFBD',
comisariado     =       '#9CC97D',
)

spots=palette(
red                 =   '#FD1115',
indigo              =   '#4C67B1',
lime                =   '#77C368',
yellow              =   '#FAE417',
orange              =   '#FF7D17',
purple              =   '#A03E9A',
brown               =   '#B8844E',
frenchblue          =   '#4CA4C9',
pink                =   '#FA80BC',
forest              =   '#2B8231',
)

fuscolor=spots.orange
dncolor=spots.indigo
mutcolor=spots.forest
upcolor=spots.red
white='#ffffff'
offwhite='#ffeedd'

beautify_model_name={
        'reactome' : 'reactome' , 
        'go_unity' : 'GO',
        'go_cc' : 'GO CC' , 
        'go_bp' : 'GO BP' , 
        'go_mf' : 'GO MF' , 
        'wk' : 'WK',
        'kuenzi' : 'Kuenzi',
        'actpwy' : 'ActPwy',
        'webster' : 'WEBSTER' ,
        'nest' : 'NeSTv0',
        'corum' : 'CORUM',
        'null' : 'null',
        'wikipathways' : 'WikiPathways',
        }

cell_block_width=1
cell_block_height=1
cell_padding_width=0.05
cell_padding_height=0.05
full_patch_height=cell_block_height-cell_padding_height*cell_block_height
full_patch_width=cell_block_width-cell_padding_height*cell_block_width
small_patch_height=0.4*cell_block_height
small_patch_upshift=full_patch_height/2-small_patch_height/2

cell_bg_color='#dddddd'


def invert_hue(ctf) : 
    h,s,v=colorsys.rgb_to_hsv(*ctf)
    newh=h+0.5 % 1.0
    return colorsys.hsv_to_rgb(newh,s,v)

def ct2h(ctup) : 
    return '#'+''.join([ '{:0>2}'.format(hex(int(ct*255))[2:]) for ct in ctup[:3] ]) 

def h2cti(h) : 
    return tuple([ int(h[2*x-1:2*x+1],16) for x in range(1,4) ])

def h2ctf(h) : 
    return tuple([ int(h[2*x-1:2*x+1],16)/255 for x in range(1,4) ])

def stripsuf(s) : 
    return s.split('_')[0]

def strippref(s) : 
    return s.split('_')[1]

def fade(chex,desat=0.2,lighten=1.2) : 
    rgb=h2ctf(chex)
    h,s,v=colorsys.rgb_to_hsv(*rgb)
    s*=desat
    v=min(v*lighten,1.0)

    h,s,v=np.clip([h,s,v],0.0,1.0)
    
    return ct2h(colorsys.hsv_to_rgb(h,s,v))

################################################################################
#   ONCOPRINT
################################################################################

    

def make_patchlist(exes,whys,**kwargs) : 
    olist=list()
    
    if 'upshift' in kwargs: 
        upshift=kwargs.pop('upshift')
    else : 
        upshift=0
    
    for x,y in zip(exes,whys) : 
        olist.append(Rectangle((x*cell_block_width,y*cell_block_height+upshift),**kwargs))
    return olist

def repindex(df,arraylike) : 
    df2=df.copy()
    df2.index=arraylike
    return df2

def get_indices_of_etype(megamelt,etype) :
    slc=megamelt.query('Event_type == @etype and Event > 0')[['colind','rowind']].astype(float)
    
    retval=slc.values.transpose()
    return retval

def oncogrid(genes,subomics,dfus,patients=None,axes=None,gene_order=None,restrict_to_selected=True) : 

    if patients is None : 
        patients=next(iter(subomics.values())).index

    cmbi.boot_bioinfo()

    full_patch_kwargs=dict( width=full_patch_width,
                   height=full_patch_height,
    )
    half_patch_kwargs=dict( width=full_patch_width,
                   height=small_patch_height,
                   upshift=small_patch_upshift
    )

    from functools import reduce
    from itertools import product

    genes=np.intersect1d(next(iter(subomics.values())).index,list(genes))

    genes_with={ k : np.intersect1d(dfus[ dfus[k] ].index,genes) for k in ETYPES }

    subsubomics_g={ k : v.loc[list(genes)] for k,v in subomics.items() }
    subsubomics_sel={ k : v.loc[list(genes_with[k])] for k,v in subomics.items() }

    if restrict_to_selected : 
        relevant_pats=reduce(
                np.bitwise_or,[ (v>0).any(axis=0) for v in subsubomics_sel.values() ]
                )
    else : 
        relevant_pats=reduce(
                np.bitwise_or,[ (v>0).any(axis=0) for v in subsubomics_g.values() ]
                )

   #relevant_pats=reduce(
   #        np.bitwise_or,[ (v>0).any(axis=0) for v in subsubomics_g.values() ]
   #        )


    print(len(genes))
    relevant_pats=relevant_pats[relevant_pats].index
    print(len(relevant_pats))
    keys=list(product(ETYPES,[True,False]))
    subsubomics_gp={
        (etype,selornot) :
            subsubomics_g[etype][relevant_pats]*dfus.loc[list(genes)][[etype]].values
            if selornot else
            subsubomics_g[etype][relevant_pats]*(
                ~dfus.loc[list(genes)][[etype]].values
                ) for etype,selornot in keys
            } 
    gene_order=reduce(
            pd.Series.__add__,[subsubomics_gp[(etype,True)].sum(axis=1)
           for etype in ETYPES]).sort_values(ascending=True).index


    pwise_ematrix=pd.concat([ 
                     repindex(subsubomics_gp[(etype,undersel)],
                         pd.MultiIndex.from_tuples([ (i,etype,undersel)
                            for i in subsubomics_gp[
                                (etype,undersel)].index ],
                                names=['gene','etype','undersel']))
                     for undersel in (True,False)
                     for etype in ETYPES ],axis=0)

    pwise_ematrix=pwise_ematrix.reindex(
            [ (g,etype,undersel) for undersel in (True,False)
             for g in reversed(gene_order) 
             for etype in ETYPES ])

    pwise_ematrix=pwise_ematrix.transpose().sort_values(
            list(pwise_ematrix.index),ascending=False)

    patient_order=pwise_ematrix.index

    subsubomics_nz={ k : 
            v.reindex(
                index=gene_order,columns=patient_order
                ).values.transpose().nonzero() for k,v in subsubomics_gp.items() }


    patchlists={ ('mut',True)  :
                    make_patchlist(*subsubomics_nz[('mut',True)  ],
                                         **half_patch_kwargs) ,
                ('mut',False) :
                    make_patchlist(*subsubomics_nz[('mut',False) ],
                                             **half_patch_kwargs) ,
                ('fus',True)  : 
                    make_patchlist(*subsubomics_nz[('fus',True)  ],
                                             **full_patch_kwargs) ,
                ('fus',False) : 
                    make_patchlist(*subsubomics_nz[('fus',False) ],
                                             **full_patch_kwargs) ,
                ('up',True)   : 
                    make_patchlist(*subsubomics_nz[('up',True)   ],
                                             **full_patch_kwargs) ,
                ('up',False)  : 
                    make_patchlist(*subsubomics_nz[('up',False)  ],
                                             **full_patch_kwargs) ,
                ('dn',True)   : 
                    make_patchlist(*subsubomics_nz[('dn',True)   ],
                                             **full_patch_kwargs) ,
                ('dn',False)  : 
                    make_patchlist(*subsubomics_nz[('dn',False)  ],
                                             **full_patch_kwargs) ,
            }

    if restrict_to_selected : 
        pcs={ 
                ('mut',True)  : PatchCollection(patchlists[('mut',True)  ],
                            facecolor=mutcolor,edgecolor=mutcolor,rasterized=True,zorder=10), 
                ('fus',True)  : PatchCollection(patchlists[('fus',True)  ],
                    facecolor='none',edgecolor=fuscolor,rasterized=True,zorder=6), 
                ('up',True)   : PatchCollection(patchlists[('up',True)   ],
                    facecolor=upcolor,edgecolor=upcolor,rasterized=True,zorder=4), 
                ('dn',True)   : PatchCollection(patchlists[('dn',True)   ],
                    facecolor=dncolor,edgecolor=dncolor,rasterized=True,zorder=2), 
        }

    else: 
        pcs={ 
                ('mut',True)  : PatchCollection(patchlists[('mut',True)  ],
                            facecolor=mutcolor,edgecolor=mutcolor,rasterized=True,zorder=10), 
                ('mut',False) : PatchCollection(patchlists[('mut',False) ],
                    facecolor=fade(mutcolor),edgecolor=fade(mutcolor),rasterized=True,zorder=7), 
                ('fus',True)  : PatchCollection(patchlists[('fus',True)  ],
                    facecolor='none',edgecolor=fuscolor,rasterized=True,zorder=6), 
                ('fus',False) : PatchCollection(patchlists[('fus',False) ],
                    facecolor='none',edgecolor=fade(fuscolor),rasterized=True,zorder=5), 
                ('up',True)   : PatchCollection(patchlists[('up',True)   ],
                    facecolor=upcolor,edgecolor=upcolor,rasterized=True,zorder=4), 
                ('up',False)  : PatchCollection(patchlists[('up',False)  ],
                    facecolor=fade(upcolor,lighten=2.0),edgecolor=fade(upcolor,lighten=2.0),
                                                    rasterized=True,zorder=3), 
                ('dn',True)   : PatchCollection(patchlists[('dn',True)   ],
                    facecolor=dncolor,edgecolor=dncolor,rasterized=True,zorder=2), 
                ('dn',False)  : PatchCollection(patchlists[('dn',False)  ],
                    facecolor=fade(dncolor),edgecolor=fade(dncolor),rasterized=True,zorder=1), 
        }

    #background_cells=make_patchlist(cbg,rbg,**full_patch_kwargs)
    ncols=len(patient_order)
    nrows=len(genes)
    #fsuv=np.array([ ncols*full_patch_width , nrows*full_patch_height ])
                 
    if axes is None : 
        f,a=plt.subplots(1,figsize=(3,3))
    else : 
        a=axes
    #pc_bg=PatchCollection(background_cells,facecolor=cell_bg_color, edgecolor='None')
    for k,v in pcs.items() : 
        a.add_collection(v)
    a.set_xlim([0,ncols*cell_block_width])
    a.set_ylim([0,nrows*cell_block_height])
    [ v.set_visible(False) for k,v in a.spines.items() ] ;

    oncogridticks=(np.arange(nrows)+0.5)*cell_block_height
    oncogridticklabels=[ cmbi._e2s.get(g) for g in gene_order ]

    a.set_yticks(oncogridticks) ; 
    a.set_yticklabels(oncogridticklabels,fontsize=6) ; 
    a.set_xlabel('Tumors (of {})'.format(subomics['mut'].shape[1]))

    return a

################################################################################
#   EVENT AND FEATURE DISSECTION
################################################################################



class CMAnalyzer(object) : 
    def __init__(self,I,H,J,synteny_broadcast,arrays,best_params,normalize_synteny=False): 
        super(CMAnalyzer,self).__init__()
        self.I=I
        self.H=H
        self.J=J
        self.synteny_broadcast=synteny_broadcast
        self.normalize_synteny=normalize_synteny
        _arr={ k : v for k,v in arrays.items() }
        _bp={ k : v for k,v in best_params.items() }
        self.gene_index=_arr['gene_index']
        self.sys_index=_arr['sys_index']
        self.gb_index=_arr['sig_index']
        self.iweights=_bp['iweights']
        self.hweights=_bp['hweights']
        self.jweights=_bp['jweights']
        self.intercept=_bp['intercept']
        self.output_log_odds=_bp['output_log_odds']
        self.partition=_bp['partition']
        self.target_event_counts=_bp['target_event_cts']
        self.n=_bp['n'].item()

        self.selection_pressures=self.get_selection_pressures()
        self.partitioned_sp=self.partition*self.selection_pressures

        proto_sely=(self.partition*self.selection_pressures)[:,:,None]
        self.de_facto_synteny=self.synteny_broadcast*(proto_sely).transpose(2,1)

        self.feattypes=np.r_[['gene']*len(arrays['gene_index']),
                        ['sys']*len(arrays['sys_index']),
                        ['gb']*len(arrays['sig_index']),]
        self.feats=np.r_[arrays['gene_index'],arrays['sys_index'],arrays['sig_index']]

        self.y=self.target_event_counts.ravel()
        self.yhat=expit(self.output_log_odds)*self.n

        if cmbi._e2s is None : 
            cmbi.boot_bioinfo()

    def get_gene_index(self,identifier): 
        try :
            int(identifier)
            eid=identifier
        except ValueError : 
            eid=cmbi._s2e.get(identifier)
        return np.argwhere(self.gene_index == eid).ravel()[0]
                
    def traceback(self,identifier,etype) : 
        try :
            int(identifier)
            eid=identifier
        except ValueError : 
            eid=cmbi._s2e.get(identifier)

        eindex=self.get_gene_index(eid)
        typeindex=list(ETYPES).index(etype)
        
        irow=self.I.to_dense()[eindex,:].numpy()
        ic_of_note=np.argwhere((irow>0) & (self.iweights.numpy()>0)).ravel()
        
        hrow=self.H.to_dense()[eindex,:].numpy()
        hc_of_note=np.argwhere((hrow>0) & (self.hweights.numpy()>0)).ravel()
        
        jrow=self.J.to_dense()[typeindex,eindex,:].numpy()
        jc_of_note=np.argwhere((jrow>0) & (self.jweights.numpy()>0)).ravel()

        dfsrow=self.de_facto_synteny.to_dense()[typeindex,eindex,:].numpy()
        dfs_of_note=np.setdiff1d(np.argwhere((dfsrow > 0)).ravel(),[eindex])
        
        orows=list()
        for icn in ic_of_note : 
            orows.append([self.gene_index[icn],irow[icn]*self.iweights[icn].numpy()*self.partition[typeindex,icn].numpy(),'gene',])
        for hcn in hc_of_note : 
            orows.append([self.sys_index[hcn],hrow[hcn]*self.hweights[hcn].numpy()*self.partition[typeindex,eindex].numpy(),'system',])
        for jcn in jc_of_note : 
            orows.append([self.gb_index[jcn],jrow[jcn]*self.jweights[jcn].numpy(),'gb',])
        for dfs in dfs_of_note : 
            orows.append([self.gene_index[dfs],dfsrow[dfs],'synteny',])
        orows.append(['_intercept',self.intercept[typeindex].item(),'intercept'])

        print(eid,cmbi._e2s.get(eid))
        print(self.output_log_odds[typeindex*len(self.iweights)+eindex].numpy())
        return orows

    def errors(self,identifier) : 
        try :
            int(identifier)
            eid=identifier
        except ValueError : 
            eid=cmbi._s2e.get(identifier)
        eindex=np.argwhere(self.gene_index == eid).ravel()[0]
        ri=np.ravel_multi_index(np.c_[np.arange(4,dtype='int'),[eindex]*4].transpose(),dims=self.J.shape[:-1])
        
        print(eid,cmbi._e2s.get(eid))
        print(*['{: <10}'.format(et) for et in ETYPES])
        print(*['{: >6.2f}    '.format(ec)  for ec in self.target_event_counts[torch.tensor(ri)]])
        print(*['{: >6.2f}    '.format(ec)  for ec in self.yhat[torch.tensor(ri)]])

    def synteny_broadcasting(self,proto_sely,synteny_broadcast=None) : 
        if  None is synteny_broadcast : 
            synteny_broadcast=self.synteny_broadcast
        if self.normalize_synteny : 
            psy2=torch.pow(proto_sely,2)
            return torch.sqrt(torch.matmul(torch.pow(self.synteny_broadcast,2),psy2)-psy2)+proto_sely

        else : 
            return torch.matmul(synteny_broadcast.to_dense(),proto_sely).squeeze(-1)

    def forward(self,**kwargs) : 
        I=kwargs.get('I',self.I)
        H=kwargs.get('H',self.H)
        J=kwargs.get('J',self.J)
        synteny_broadcast=kwargs.get('synteny_broadcast',self.synteny_broadcast)
        iweights=kwargs.get('iweights',self.iweights)
        hweights=kwargs.get('hweights',self.hweights)
        jweights=kwargs.get('jweights',self.jweights)
        intercept=kwargs.get('intercept',self.intercept)
        partition=kwargs.get('partition',self.partition)

        sp=self.get_selection_pressures(**kwargs)
        proto_sely=(partition*sp)[:,:,None]

        return (self.synteny_broadcasting(proto_sely,synteny_broadcast)+torch.matmul(J.to_dense(),jweights)+intercept[:,None]).ravel()

    def likelihood(self,logodds=None,n=None,y=None) : 
        from torch.distributions import Binomial
        #from scipy.stats.distributions import binom
        #from scipy.special import expit

        if None is logodds : 
            logodds=self.output_log_odds
        if None is n : 
            n=self.n
        if None is y : 
            y=self.target_event_counts

        return Binomial(logits=logodds,total_count=n).log_prob(y)

    def get_selection_pressures(self,**kwargs) : 
        I=kwargs.get('I',self.I)
        H=kwargs.get('H',self.H)
        iweights=kwargs.get('iweights',self.iweights)
        hweights=kwargs.get('hweights',self.hweights)
        if len(hweights) == 0  : 
            return torch.matmul(I,iweights)
        else : 
            return torch.matmul(I,iweights)+torch.matmul(H,hweights)

    def selectionless_likelihood(self,**kwargs) : 
        output_log_odds=kwargs.get('output_log_odds',self.output_log_odds)
        if not any([ k in kwargs for k in ('I','H','iweights','hweights')]) : 
            partitioned_sp=self.partitioned_sp
        else : 
            selection_pressures=self.get_selection_pressures(**kwargs)
            partitioned_sp=self.partition*selection_pressures

        return self.likelihood(output_log_odds-partitioned_sp.ravel())

    def underselection(self,cutoff=np.log(2),numeric=False,**kwargs) :
        output_log_odds=kwargs.get('output_log_odds',self.output_log_odds)

        fc=pd.DataFrame(data=(
                (self.likelihood(output_log_odds)-self.selectionless_likelihood(**kwargs)).reshape(4,len(self.gene_index))).numpy(),
                     index=ETYPES,
                     columns=self.gene_index).transpose()
        if numeric : 
            return fc
        else : 
            return fc > cutoff

    def feature_summary_frame(self,bp_subsample_files=None,quantiles=(0.025,0.975),parallel=True) : 
        import multiprocessing as mp
        if None is bp_subsample_files or\
            (type(bp_subsample_files) == list and len(bp_subsample_files) == 0) : 
            mapvec=np.r_[self.iweights.numpy(),self.hweights.numpy(),self.jweights.numpy()]
            df=pd.DataFrame().assign(
                feat=self.feats,
                feattype=self.feattypes,
                map_estimate=mapvec,
            )
            df['is_sturdy']=True
            df['Feature Type']=pd.Categorical(df.feattype.apply({ 'gene' : 'Gene' , 'sys' : 'System', 'gb' : 'Gen. Bkgd.' }.get),categories=['Gene','System','Gen. Bkgd.']) 
            return df

        if parallel: 
            with mp.Pool(processes=len(os.sched_getaffinity(0))) as p : 
                bpdatas=p.map(bptproc,bp_subsample_files)
        else : 
            bpdatas=map(bptproc,bp_subsample_files)

        iweight_slices=list()
        hweight_slices=list()
        jweight_slices=list()
        rseps_bp=list()
        for bpd in bpdatas : 
            iweight_slices.append(bpd['iweights'].numpy())
            hweight_slices.append(bpd['hweights'].numpy())
            jweight_slices.append(bpd['jweights'].numpy())
            rseps_bp.append(bpd['rsep'])
        rseps_bp=np.array(rseps_bp)
        iweights=np.stack(iweight_slices,axis=0)
        hweights=np.stack(hweight_slices,axis=0)
        jweights=np.stack(jweight_slices,axis=0)
        rseps_as=np.argsort(rseps_bp)
        iweights=iweights[rseps_as,:]
        hweights=hweights[rseps_as,:]
        jweights=jweights[rseps_as,:]
        rseps_bp=rseps_bp[rseps_as]

        expect_median=np.r_[np.median(iweights,axis=0),
                            np.median(hweights,axis=0),
                            np.median(jweights,axis=0) ]
        expect_mean=np.r_[iweights.mean(axis=0),
                            hweights.mean(axis=0),
                            jweights.mean(axis=0)]

        expect_lo=np.r_[np.quantile(iweights,quantiles[0],axis=0),
                        np.quantile(hweights,quantiles[0],axis=0),
                        np.quantile(jweights,quantiles[0],axis=0)]

        expect_hi=np.r_[np.quantile(iweights,quantiles[1],axis=0),
                        np.quantile(hweights,quantiles[1],axis=0),
                        np.quantile(jweights,quantiles[1],axis=0)]

        mapvec=np.r_[self.iweights,self.hweights,self.jweights]
        df=pd.DataFrame().assign(
            feat=self.feats,
            feattype=self.feattypes,
            map_estimate=mapvec,
            expect_median=expect_median,
            expect_mean=expect_mean,
            expect_lo=expect_lo,
            expect_hi=expect_hi,
        )
        df['interval']=(df.expect_hi - df.expect_lo)
        df['is_sturdy']=df.expect_lo.gt(0.0)
        df['Feature Type']=pd.Categorical(df.feattype.apply({ 'gene' : 'Gene' , 'sys' : 'System', 'gb' : 'Gen. Bkgd.' }.get),categories=['Gene','System','Gen. Bkgd.']) 

        return df

def loadcma(indir,outdir) : 
    thiscma=CMAnalyzer(
        I=tload(opj(indir,'I.pt')),
        H=tload(opj(indir,'H.pt')),
        J=tload(opj(indir,'J.pt')).to_dense(),
        synteny_broadcast=tload(opj(indir,'synteny_broadcast.pt')).float(),
        arrays=np.load(opj(indir,'arrays.npz'),allow_pickle=True),
        best_params=tload(opj(outdir,'bp.pt')),
        )
    return  thiscma


class CMAnalyzer_numpy(object) : 
    def __init__(self,I,H,J,synteny_broadcast,arrays,best_params,normalize_synteny=False): 
        super(CMAnalyzer_numpy,self).__init__()
        self.I=I.to_dense().numpy()
        self.H=H.to_dense().numpy()
        self.J=J.to_dense().numpy()
        self.synteny_broadcast=synteny_broadcast.to_dense().numpy()
        self.normalize_synteny=normalize_synteny
        _arr={ k : v for k,v in arrays.items() }
        _bp={ k : v.numpy() for k,v in best_params.items() }
        self.gene_index=_arr['gene_index']
        self.sys_index=_arr['sys_index']
        self.gb_index=_arr['sig_index']
        self.iweights=_bp['iweights']
        self.hweights=_bp['hweights']
        self.jweights=_bp['jweights']
        self.intercept=_bp['intercept']
        self.output_log_odds=_bp['output_log_odds']
        self.partition=_bp['partition']
        self.target_event_counts=_bp['target_event_cts']
        self.n=_bp['n'].item()

        self.selection_pressures=self.get_selection_pressures()
        self.partitioned_sp=self.partition*self.selection_pressures

        proto_sely=(self.partition*self.selection_pressures)[:,:,None]
        self.de_facto_synteny=self.synteny_broadcast*(proto_sely).transpose(0,2,1)

        self.feattypes=np.r_[['gene']*len(arrays['gene_index']),
                        ['sys']*len(arrays['sys_index']),
                        ['gb']*len(arrays['sig_index']),]
        self.feats=np.r_[arrays['gene_index'],arrays['sys_index'],arrays['sig_index']]

        self.y=self.target_event_counts.ravel()
        self.yhat=expit(self.output_log_odds)*self.n

        if cmbi._e2s is None : 
            cmbi.boot_bioinfo()

    def get_gene_index(self,identifier): 
        try :
            int(identifier)
            eid=identifier
        except ValueError : 
            eid=cmbi._s2e.get(identifier)
        return np.argwhere(self.gene_index == eid).ravel()[0]
                
    def traceback(self,identifier,etype) : 
        try :
            int(identifier)
            eid=identifier
        except ValueError : 
            eid=cmbi._s2e.get(identifier)

        eindex=self.get_gene_index(eid)
        typeindex=list(ETYPES).index(etype)
        
        irow=self.I[eindex,:]
        ic_of_note=np.argwhere((irow>0) & (self.iweights>0)).ravel()
        
        hrow=self.H[eindex,:]
        hc_of_note=np.argwhere((hrow>0) & (self.hweights>0)).ravel()
        
        jrow=self.J[typeindex,eindex,:]
        jc_of_note=np.argwhere((jrow>0) & (self.jweights>0)).ravel()

        dfsrow=self.de_facto_synteny[typeindex,eindex,:]
        dfs_of_note=np.setdiff1d(np.argwhere((dfsrow > 0)).ravel(),[eindex])
        
        orows=list()
        for icn in ic_of_note : 
            orows.append([self.gene_index[icn],irow[icn]*self.iweights[icn]*self.partition[typeindex,icn],'gene',])
        for hcn in hc_of_note : 
            orows.append([self.sys_index[hcn],hrow[hcn]*self.hweights[hcn]*self.partition[typeindex,eindex],'system',])
        for jcn in jc_of_note : 
            orows.append([self.gb_index[jcn],jrow[jcn]*self.jweights[jcn],'gb',])
        for dfs in dfs_of_note : 
            orows.append([self.gene_index[dfs],dfsrow[dfs],'synteny',])
        orows.append(['_intercept',self.intercept[typeindex],'intercept'])

        print(eid,cmbi._e2s.get(eid))
        print(self.output_log_odds[typeindex*len(self.iweights)+eindex])
        return orows

    def errors(self,identifier) : 
        try :
            int(identifier)
            eid=identifier
        except ValueError : 
            eid=cmbi._s2e.get(identifier)

        eindex=np.argwhere(self.gene_index == eid).ravel()[0]
        ri=np.ravel_multi_index(np.c_[np.arange(4,dtype='int'),[eindex]*4].transpose(),dims=self.J.shape[:-1])
        
        print(eid,cmbi._e2s.get(eid))
        print(*['{: <10}'.format(et) for et in ETYPES])
        print(*['{: >6.2f}    '.format(ec)  for ec in self.target_event_counts[ri]])
        print(*['{: >6.2f}    '.format(ec)  for ec in self.yhat[ri]])

    def synteny_broadcasting(self,proto_sely,synteny_broadcast=None) : 
        if  None is synteny_broadcast : 
            synteny_broadcast=self.synteny_broadcast
        if self.normalize_synteny : 
            psy2=np.pow(proto_sely,2)
            return np.sqrt(np.matmul(np.pow(self.synteny_broadcast,2),psy2)-psy2)+proto_sely

        else : 
            return np.matmul(synteny_broadcast,proto_sely).squeeze(-1)

    def forward(self,**kwargs) : 
        I=kwargs.get('I',self.I)
        H=kwargs.get('H',self.H)
        J=kwargs.get('J',self.J)
        synteny_broadcast=kwargs.get('synteny_broadcast',self.synteny_broadcast)
        iweights=kwargs.get('iweights',self.iweights)
        hweights=kwargs.get('hweights',self.hweights)
        jweights=kwargs.get('jweights',self.jweights)
        intercept=kwargs.get('intercept',self.intercept)
        partition=kwargs.get('partition',self.partition)

        sp=self.get_selection_pressures(**kwargs)
        proto_sely=(partition*sp)[:,:,None]

        return (self.synteny_broadcasting(proto_sely,synteny_broadcast)+np.matmul(J,jweights)+intercept[:,None]).ravel()

    def likelihood(self,logodds=None,n=None,y=None) : 
        from scipy.stats.distributions import binom
        from scipy.special import expit

        if None is logodds : 
            logodds=self.output_log_odds
        if None is n : 
            n=self.n
        if None is y : 
            y=self.target_event_counts

        return binom.logpmf(y,n,expit(logodds))

    def get_selection_pressures(self,**kwargs) : 
        I=kwargs.get('I',self.I)
        H=kwargs.get('H',self.H)
        iweights=kwargs.get('iweights',self.iweights)
        hweights=kwargs.get('hweights',self.hweights)
        return np.matmul(I,iweights)+np.matmul(H,hweights)

    def selectionless_likelihood(self,**kwargs) : 
        output_log_odds=kwargs.get('output_log_odds',self.output_log_odds)
        if not any([ k in kwargs for k in ('I','H','iweights','hweights')]) : 
            partitioned_sp=self.partitioned_sp
        else : 
            selection_pressures=self.get_selection_pressures(**kwargs)
            partitioned_sp=self.partition*selection_pressures

        return self.likelihood(output_log_odds-partitioned_sp.ravel())

    def underselection(self,cutoff=np.log(2),numeric=False,**kwargs) :
        output_log_odds=kwargs.get('output_log_odds',self.output_log_odds)

        fc=pd.DataFrame(data=(
                self.likelihood(output_log_odds)-self.selectionless_likelihood(**kwargs)).reshape(4,len(self.gene_index)),
                     index=ETYPES,
                     columns=self.gene_index).transpose()
        if numeric : 
            return fc

        else : 
            return fc > cutoff

    def feature_summary_frame(self,bp_subsample_files,quantiles=(0.025,0.975),parallel=True) : 
        import multiprocessing as mp
        if None is bp_subsample_files or\
            (type(bp_subsample_files) == list and len(bp_subsample_files) == 0) : 
            mapvec=np.r_[self.iweights,self.hweights,self.jweights]
            df=pd.DataFrame().assign(
                feat=self.feats,
                feattype=self.feattypes,
                map_estimate=mapvec,
            )
            df['is_sturdy']=True
            df['Feature Type']=pd.Categorical(df.feattype.apply({ 'gene' : 'Gene' , 'sys' : 'System', 'gb' : 'Gen. Bkgd.' }.get),categories=['Gene','System','Gen. Bkgd.']) 
            return df

        if parallel: 
            with mp.Pool(processes=len(os.sched_getaffinity(0))) as p : 
                bpdatas=p.map(bptproc,bp_subsample_files)
        else : 
            bpdatas=map(bptproc,bp_subsample_files)

        iweight_slices=list()
        hweight_slices=list()
        jweight_slices=list()
        rseps_bp=list()
        for bpd in bpdatas : 
            iweight_slices.append(bpd['iweights'].numpy())
            hweight_slices.append(bpd['hweights'].numpy())
            jweight_slices.append(bpd['jweights'].numpy())
            rseps_bp.append(bpd['rsep'])
        rseps_bp=np.array(rseps_bp)
        iweights=np.stack(iweight_slices,axis=0)
        hweights=np.stack(hweight_slices,axis=0)
        jweights=np.stack(jweight_slices,axis=0)
        rseps_as=np.argsort(rseps_bp)
        iweights=iweights[rseps_as,:]
        hweights=hweights[rseps_as,:]
        jweights=jweights[rseps_as,:]
        rseps_bp=rseps_bp[rseps_as]

        expect_median=np.r_[np.median(iweights,axis=0),
                            np.median(hweights,axis=0),
                            np.median(jweights,axis=0) ]
        expect_mean=np.r_[iweights.mean(axis=0),
                            hweights.mean(axis=0),
                            jweights.mean(axis=0)]

        expect_lo=np.r_[np.quantile(iweights,quantiles[0],axis=0),
                        np.quantile(hweights,quantiles[0],axis=0),
                        np.quantile(jweights,quantiles[0],axis=0)]

        expect_hi=np.r_[np.quantile(iweights,quantiles[1],axis=0),
                        np.quantile(hweights,quantiles[1],axis=0),
                        np.quantile(jweights,quantiles[1],axis=0)]

        mapvec=np.r_[self.iweights,self.hweights,self.jweights]
        df=pd.DataFrame().assign(
            feat=self.feats,
            feattype=self.feattypes,
            map_estimate=mapvec,
            expect_median=expect_median,
            expect_mean=expect_mean,
            expect_lo=expect_lo,
            expect_hi=expect_hi,
        )
        df['interval']=(df.expect_hi - df.expect_lo)
        df['is_sturdy']=df.expect_lo.gt(0.0)
        df['Feature Type']=pd.Categorical(df.feattype.apply({ 'gene' : 'Gene' , 'sys' : 'System', 'gb' : 'Gen. Bkgd.' }.get),categories=['Gene','System','Gen. Bkgd.']) 

        return df



        
def bptproc(bpfn) :
    bpdatas=tload(bpfn)
    bpdatas.update({ 'rsep' : bpfn.split(os.sep)[-2] })
    return bpdatas        

tload=lambda fp : torch.load(fp,map_location=torch.device('cpu'),weights_only=True)

def qpl(fp) : 
    with open(fp,'rb') as f: 
        return pickle.load(f)
        
def doubleup(a) : 
    return np.c_[a,a].ravel()[1:-1]


import pickle


def legacy_oncogrid(events,omics,patients=None,axes=None,gene_order=None) : 

    if patients is None : 
        patients=omics.index

    if cmbi._e2s is None : 
        cmbi.boot_bioinfo()

    genes=list({ stripsuf(e) for e in events})
    megamelt=make_megamelt(events,omics)
    ppiv=make_megapiv(megamelt,gene_order=gene_order)

    ucolgenes=ppiv.columns.get_level_values(0)[::-1]
    ucolgenes=[ ucg for e,ucg in enumerate(ucolgenes) if e == (len(ucolgenes)-1) or ucg not in ucolgenes[e+1:] ]
    pnz=ppiv.loc[ppiv.sum(axis=1).gt(0)]
    print(pnz.shape)
    rowind=dict(zip(ucolgenes,np.arange(len(ucolgenes))))
    colind=dict(zip(pnz.index,np.arange(pnz.shape[0])))

    megamelt['colind']=megamelt.Tumor_Sample_Barcode.apply(colind.get)
    megamelt['rowind']=megamelt.entrez.apply(rowind.get)
    megamelt=megamelt.dropna(subset='colind')

    row_indices=sorted(list(rowind.values()))
    col_indices=sorted(list(colind.values()))

    cmat,rmat=np.meshgrid(col_indices,row_indices)
    rbg=rmat.ravel()
    cbg=cmat.ravel()

    full_patch_kwargs=dict( width=full_patch_width,
                   height=full_patch_height,
    )
    half_patch_kwargs=dict( width=full_patch_width,
                   height=small_patch_height,
                   upshift=small_patch_upshift
    )

    #background_cells=make_patchlist(cbg,rbg,**full_patch_kwargs)
    fus_cells=make_patchlist(*get_indices_of_etype(megamelt,'fus'),**full_patch_kwargs)
    up_cells=make_patchlist(*get_indices_of_etype(megamelt,'up'),**full_patch_kwargs)
    dn_cells=make_patchlist(*get_indices_of_etype(megamelt,'dn'),**full_patch_kwargs)
    mut_cells=make_patchlist(*get_indices_of_etype(megamelt,'mut'),**half_patch_kwargs)

    ncols=pnz.shape[0]
    nrows=len(ucolgenes)
    #fsuv=np.array([ ncols*full_patch_width , nrows*full_patch_height ])
                 
    if axes is None : 
        f,a=plt.subplots(1,figsize=(3,3))
    else : 
        a=axes
    #pc_bg=PatchCollection(background_cells,facecolor=cell_bg_color, edgecolor='None')
    pc_up=PatchCollection(up_cells,facecolor=upcolor,
                          edgecolor='None',rasterized=True)
    pc_dn=PatchCollection(dn_cells,facecolor=dncolor,
                          edgecolor='None',rasterized=True)
    pc_fu=PatchCollection(fus_cells,facecolor='None',
                          edgecolor=fuscolor,rasterized=True)
    pc_mut=PatchCollection(mut_cells,facecolor=mutcolor,
                           edgecolor='None',rasterized=True)

    #a.add_collection(pc_bg)
    a.add_collection(pc_up)
    a.add_collection(pc_dn)
    a.add_collection(pc_fu)
    a.add_collection(pc_mut)


    a.set_xlim([0,ncols*cell_block_width])
    a.set_ylim([0,nrows*cell_block_height])
    [ v.set_visible(False) for k,v in a.spines.items() ] ;

    oncogridticks=(np.arange(nrows)+0.5)*cell_block_height
    oncogridticklabels=[ kg._e2s.get(g) for g in ucolgenes ]

    a.set_yticks(oncogridticks) ; 
    a.set_yticklabels(oncogridticklabels,fontsize=6) ; 
    a.set_xlabel('Patients')

    return megamelt,a

def legacy_make_melt_frame(omics,events,suffix,etypeval) : 
    etype_frame=omics[ omics.columns[omics.columns.str.endswith(suffix)] ].rename(columns=stripsuf)
    genes=list({ stripsuf(e) for e in events})
    etype_frame_raw=etype_frame.reindex(columns=genes).transpose()
    etype_frame_raw.index.name='entrez'
    etype_melt=etype_frame_raw.reset_index().melt(id_vars='entrez',var_name='Tumor_Sample_Barcode',value_name='Event').assign(Event_type=etypeval)
    return etype_melt.dropna()

def legacy_make_megamelt(events,omics) : 
    megamelt=pd.concat([ make_melt_frame( omics,events, suffix=suf, etypeval=etv)
                       for suf,etv in zip(['_mut','_fus','_up','_dn'],['mut','fus','up','dn'])],axis=0)
    megamelt=megamelt.query('Event > 0 or (Event_type == "mut") ').copy()
    return megamelt

def legacy_make_megapiv(megamelt,gene_order=None) : 
    if gene_order is None : 
        gene_order=megamelt.groupby('entrez').Event.sum().sort_values(ascending=False).index
    from itertools import product
    rico=list(product(gene_order,['up','dn','mut','fus']))
    ppiv=megamelt.pivot_table(
        index='Tumor_Sample_Barcode',columns=['entrez','Event_type'],values='Event',aggfunc='sum').fillna(0).reindex(columns=rico)
    ppiv=ppiv[ ppiv.columns[~ppiv.isnull().all() | (ppiv.columns.get_level_values(1) == 'mut') ]]
    ppiv=ppiv.sort_values(by=ppiv.columns.to_list(),ascending=False,axis=0)
    return ppiv
