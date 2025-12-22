#!/usr/bin/sh


python3 << fin
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
df_test=pd.read_csv('summary/feature_summary.csv',index_col=0).set_index('feat')
df_canon=pd.read_csv('../data/canon_feature_data.csv',index_col=0).set_index('feat')

print('Feature weight agreement with publication (pearson)')
print(pearsonr(df_test.map_estimate,df_canon.map_estimate))

print('Feature identification agreement with publication (jaccard,differences)')
intest=np.array(df_test.query('map_estimate >0').index)
incanon=np.array(df_canon.query('map_estimate >0').index)

print('{: ^30}|{: ^30}'.format('Local run','Publication'))
print('-'*61)

testonly=np.setdiff1d(intest,incanon)
canononly=np.setdiff1d(incanon,intest)
both=np.intersect1d(incanon,intest)

print('{: ^19}|{: ^20}|{: ^19}'.format('only','common','only'))
print('{: ^19}|{: ^20}|{: ^19}'.format(len(testonly),len(both),len(canononly)))

print()
print(60*'=')
import torch
from cansrmapp import cmbuilder as cmb

def omics_syncer(omics_fp,valid_gene_ids) :
    omics=cmb.read_omics(omics_fp)
    master_gene_index,nzomics,subomics=cmb.omics_to_nonzeros(omics,valid_gene_ids)
    sparse_omics=cmb.create_sparse_omics(nzomics,osize=len(master_gene_index))
    
    ofreq=sparse_omics.sum(axis=2)/sparse_omics.shape[2]

    return master_gene_index,ofreq.to_dense().numpy().ravel()

canon_genes=df_canon[ df_canon.feattype.eq('gene') ].index
tcga_genes,tcga_freq=omics_syncer('../data/omics_tcga_luad.csv.gz',canon_genes)
cptac_genes,cptac_freq=omics_syncer('../data/omics_cptac_luad.csv.gz',canon_genes)
pred_freq=torch.load('summary/model_pred_frequencies.pt',weights_only=True)

cptac_gene_indices=np.argwhere(canon_genes.isin(cptac_genes))
cptac_freq_indices=np.concatenate(
[ cptac_gene_indices+x*len(canon_genes) for x in range(4) ],
axis=0,
).ravel()
pred_cptac_freq=pred_freq.numpy()[ cptac_freq_indices ]



print('TCGA-LUAD [training] frequency agreement (pearson) :')
print( pearsonr(
        tcga_freq.ravel(),
        pred_freq.numpy(),
        ))

print('TCGA-CPTAC [evaluation] frequency agreement (pearson) :')
print( pearsonr(
        cptac_freq.ravel(),
        pred_cptac_freq,
        ))

fin

