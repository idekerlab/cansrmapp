from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np

import kidgloves as kg

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
        'reactome_prelim' : 'reactome' , 
        'go_unity' : 'GO',
        'go_cc' : 'GO CC' , 
        'go_bp' : 'GO BP' , 
        'go_mf' : 'GO MF' , 
        'kamber' : 'WK',
        'kuenzi_census' : 'Kuenzi Cen.',
        'kuenzi_consensus' : 'Kuenzi',
        'pcawg' : 'ActPwy',
        'webster' : 'WEBSTER' ,
        'H_IAS_clixo_hidef_Nov17.edges' : 'NeSTv0',
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
small_patch_height=0.25*cell_block_height
small_patch_upshift=full_patch_height/2-small_patch_height/2

cell_bg_color='#dddddd'


def invert_hue(ctf) : 
    import colorsys
    h,s,v=colorsys.rgb_to_hsv(*ctf)
    newh=h+0.5 % 1.0
    return colorsys.hsv_to_rgb(newh,s,v)

def ct2h(ctup) : 
    return '#'+''.join([ hex(int(ct*255))[2:] for ct in ctup[:3] ]) 

def h2cti(h) : 
    return tuple([ int(h[2*x-1:2*x+1],16) for x in range(1,4) ])

def h2ctf(h) : 
    return tuple([ int(h[2*x-1:2*x+1],16)/255 for x in range(1,4) ])


#a.set_xscale('symlog')
def stripsuf(s) : 
    return s.split('_')[0]

def strippref(s) : 
    return s.split('_')[1]

def make_melt_frame(omics,events,suffix,etypeval) : 
    etype_frame=omics[ omics.columns[omics.columns.str.endswith(suffix)] ].rename(columns=stripsuf)
    genes=list({ stripsuf(e) for e in events})
    etype_frame_raw=etype_frame.reindex(columns=genes).transpose()
    etype_frame_raw.index.name='entrez'
    etype_melt=etype_frame_raw.reset_index().melt(id_vars='entrez',var_name='Tumor_Sample_Barcode',value_name='Event').assign(Event_type=etypeval)
    return etype_melt.dropna()

def make_megamelt(events,omics) : 
    megamelt=pd.concat([ make_melt_frame( omics,events, suffix=suf, etypeval=etv)
                       for suf,etv in zip(['_mut','_fus','_up','_dn'],['mut','fus','up','dn'])],axis=0)
    megamelt=megamelt.query('Event > 0 or (Event_type == "mut") ').copy()
    return megamelt

def make_megapiv(megamelt,gene_order=None) : 
    if gene_order is None : 
        gene_order=megamelt.groupby('entrez').Event.sum().sort_values(ascending=False).index
    from itertools import product
    rico=list(product(gene_order,['up','dn','mut','fus']))
    ppiv=megamelt.pivot_table(
        index='Tumor_Sample_Barcode',columns=['entrez','Event_type'],values='Event',aggfunc='sum').fillna(0).reindex(columns=rico)
    ppiv=ppiv[ ppiv.columns[~ppiv.isnull().all() | (ppiv.columns.get_level_values(1) == 'mut') ]]
    ppiv=ppiv.sort_values(by=ppiv.columns.to_list(),ascending=False,axis=0)
    return ppiv
    

def make_patchcollection(exes,whys,**kwargs) : 
    olist=list()
    
    if 'upshift' in kwargs: 
        upshift=kwargs.pop('upshift')
    else : 
        upshift=0
    
    for x,y in zip(exes,whys) : 
        olist.append(Rectangle((x*cell_block_width,y*cell_block_height+upshift),**kwargs))
    return olist

def get_indices_of_etype(megamelt,etype) :
    slc=megamelt.query('Event_type == @etype and Event > 0')[['colind','rowind']].astype(float)
    
    retval=slc.values.transpose()
    return retval

def oncogrid(events,omics,patients=None,axes=None,gene_order=None) : 

    if patients is None : 
        patients=omics.index

    if kg._e2s is None : 
        kg._get_geneinfo() 

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

    #background_cells=make_patchcollection(cbg,rbg,**full_patch_kwargs)
    fus_cells=make_patchcollection(*get_indices_of_etype(megamelt,'fus'),**full_patch_kwargs)
    up_cells=make_patchcollection(*get_indices_of_etype(megamelt,'up'),**full_patch_kwargs)
    dn_cells=make_patchcollection(*get_indices_of_etype(megamelt,'dn'),**full_patch_kwargs)
    mut_cells=make_patchcollection(*get_indices_of_etype(megamelt,'mut'),**half_patch_kwargs)

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

