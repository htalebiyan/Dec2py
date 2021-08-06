import numpy as np
import pandas as pd
import pickle

root1 = 'C:/Users/ht20/Documents/Files/Game_Shelby_County/postprocess/'
df1 = pd.read_pickle(root1+'postprocess_dicts.pkl')
root2 = 'C:/Users/ht20/Documents/GitHub/Dec2py/results/'
df2 = pd.read_pickle(root2+'postprocess_dicts.pkl')

for i in [0, 1, 3]:
    for x in df2[i]:
        if x not in df1[i]:
            df1[i].append(x)
for i in [2, 4]:
    df1[i] = pd.concat([df1[i],df2[i]]).drop_duplicates().reset_index(drop=True)
    df1[i]['no_resources'] = pd.to_numeric(df1[i]['no_resources'])
with open(root2+'temp.pkl', 'wb') as f:
    pickle.dump(df1, f)

