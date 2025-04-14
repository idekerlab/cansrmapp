#!/usr/bin/sh


python3 << fin
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
df_test=pd.read_csv('summary/feature_summary.csv',index_col=0).set_index('feat')
df_canon=pd.read_csv('../data/canon_feature_data.csv',index_col=0).set_index('feat')

print('Agreement with publication (pearson)')
print(pearsonr(df_test.map_estimate,df_canon.map_estimate))

print('Agreement with publication (jaccard,differences)')
intest=np.array(df_test.query('map_estimate >0').index)
incanon=np.array(df_canon.query('map_estimate >0').index)

print('{: ^30}|{: ^30}'.format('Local run','Publication'))
print('-'*61)

testonly=np.setdiff1d(intest,incanon)
canononly=np.setdiff1d(incanon,intest)
both=np.intersect1d(incanon,intest)

print('{: ^19}|{: ^20}|{: ^19}'.format(len(testonly),len(both),len(canononly)))
fin

