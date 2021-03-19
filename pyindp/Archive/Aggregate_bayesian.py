import numpy as np
import pandas as pd
import pickle

folder = '/home/hesam/Desktop/Files/Game_Shelby_County/'
root = folder+'postprocess_BNE_90perc/'
df1 = pd.read_pickle(root+'postprocess_dicts_CCCCUUUU.pkl')
df2 = pd.read_pickle(root+'postprocess_dicts_NNNNUUUU.pkl')
df3 = pd.read_pickle(root+'postprocess_dicts_CCNCUUUU.pkl')
df4 = pd.read_pickle(root+'postprocess_dicts_CCCCFFFF.pkl')
df5 = pd.read_pickle(root+'postprocess_dicts_NNNNFFFF.pkl')
df6 = pd.read_pickle(root+'postprocess_dicts_CCNCFFFF.pkl')
df7 = pd.read_pickle(root+'postprocess_dicts_CCCCIIII.pkl')
df8 = pd.read_pickle(root+'postprocess_dicts_NNNNIIII.pkl')
df9 = pd.read_pickle(root+'postprocess_dicts_CCNCIIII.pkl')
df10 = pd.read_pickle(folder+'postprocess_NE_90perc/postprocess_dicts.pkl')
for i in [0, 1, 3, 8]:
    for x in df2[i]:
        if x not in df1[i]:
            df1[i].append(x)
    for x in df3[i]:
        if x not in df1[i]:
            df1[i].append(x)
    for x in df4[i]:
        if x not in df1[i]:
            df1[i].append(x)
    for x in df5[i]:
        if x not in df1[i]:
            df1[i].append(x)
    for x in df6[i]:
        if x not in df1[i]:
            df1[i].append(x)
    for x in df7[i]:
        if x not in df1[i]:
            df1[i].append(x)
    for x in df8[i]:
        if x not in df1[i]:
            df1[i].append(x)
    for x in df9[i]:
        if x not in df1[i]:
            df1[i].append(x)
    for x in df10[i]:
        if x not in df1[i]:
            df1[i].append(x)
for i in [2, 4, 5, 6, 7]:
    df1[i] = pd.concat([df1[i],df2[i]]).drop_duplicates().reset_index(drop=True)
    df1[i] = pd.concat([df1[i],df3[i]]).drop_duplicates().reset_index(drop=True)
    df1[i] = pd.concat([df1[i],df4[i]]).drop_duplicates().reset_index(drop=True)
    df1[i] = pd.concat([df1[i],df5[i]]).drop_duplicates().reset_index(drop=True)
    df1[i] = pd.concat([df1[i],df6[i]]).drop_duplicates().reset_index(drop=True)
    df1[i] = pd.concat([df1[i],df7[i]]).drop_duplicates().reset_index(drop=True)
    df1[i] = pd.concat([df1[i],df8[i]]).drop_duplicates().reset_index(drop=True)
    df1[i] = pd.concat([df1[i],df9[i]]).drop_duplicates().reset_index(drop=True)
    df1[i] = pd.concat([df1[i],df10[i]]).drop_duplicates().reset_index(drop=True)
df1[9] = pd.concat([df1[9],df2[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)
df1[9] = pd.concat([df1[9],df3[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)
df1[9] = pd.concat([df1[9],df4[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)
df1[9] = pd.concat([df1[9],df5[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)
df1[9] = pd.concat([df1[9],df6[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)
df1[9] = pd.concat([df1[9],df7[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)
df1[9] = pd.concat([df1[9],df8[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)
df1[9] = pd.concat([df1[9],df9[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)
df1[9] = pd.concat([df1[9],df10[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)

with open(root+'postprocess_dicts.pkl', 'wb') as f:
    pickle.dump(df1, f)

