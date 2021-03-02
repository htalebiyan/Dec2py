import numpy as np
import pandas as pd
import pickle

root = '/home/hesam/Desktop/Files/Game_Shelby_County/postprocess_BNE_90perc/'
df1 = pd.read_pickle(root+'postprocess_dicts_CCCCUUUU.pkl')
df2 = pd.read_pickle(root+'postprocess_dicts_NNNNUUUU.pkl')
df3 = pd.read_pickle(root+'postprocess_dicts_CCNCUUUU.pkl')

for i in [0, 1, 3, 8]:
    for x in df2[i]:
        if x not in df1[i]:
            df1[i].append(x)
    for x in df3[i]:
        if x not in df1[i]:
            df1[i].append(x)
for i in [2, 4, 5, 6, 7]:
    df1[i] = pd.concat([df1[i],df2[i]]).drop_duplicates().reset_index(drop=True)
    df1[i] = pd.concat([df1[i],df3[i]]).drop_duplicates().reset_index(drop=True)
df1[9] = pd.concat([df1[9],df2[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)
df1[9] = pd.concat([df1[9],df3[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)

with open(root+'postprocess_dicts.pkl', 'wb') as f:
    pickle.dump(df1, f)

