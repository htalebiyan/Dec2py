import numpy as np
import pandas as pd
import pickle

folder = '/home/hesam/Desktop/Files/Game_synthetic/v4/postprocess/'
df1 = pd.read_pickle(folder+'postprocess_dicts.pkl')
df2 = pd.read_pickle(folder+'postprocess_dicts_BAYESGAME_bgCCUU_EDM20_OPTIMALandUNIFORM.pkl')

for i in [2, 4, 5, 6, 7, 9, 10]:
    # df1[i].loc[:, 'topology'] = 'general'
    # df1[i].loc[df1[i]['decision_type']=='ng', 'rationality'] = 'unbounded'
    # df1[i].loc[df1[i]['decision_type']=='indp', 'rationality'] = 'optimal'
    df2[i].loc[:, 'topology'] = 'general'
    df2[i].loc[df2[i]['decision_type']=='bgCCUU', 'rationality'] = 'bounded'
    df2[i].loc[df2[i]['decision_type']=='indp', 'rationality'] = 'optimal'
for i in [0, 1, 3, 8]:
    for x in df2[i]:
        if x not in df1[i]:
            df1[i].append(x)

for i in [2, 4, 5, 6, 7, 10]:
    df1[i] = pd.concat([df1[i],df2[i]]).drop_duplicates().reset_index(drop=True)

df1[9] = pd.concat([df1[9],df2[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)

with open(folder+'postprocess_dicts_temp.pkl', 'wb') as f:
    pickle.dump(df1, f)

