import numpy as np
import pandas as pd
import pickle

folder = '/home/hesam/Desktop/Files/Game_synthetic/v4/postprocess/'
df1 = pd.read_pickle(folder+'postprocess_dicts.pkl')
df2 = pd.read_pickle(folder+'postprocess_dicts_BAYESGAME_bgNNUU_EDM20_OPTIMALandUNIFORM.pkl')

for i in [2, 4, 5, 6, 7, 9, 10]:
    # df1[i].loc[:, 'topology'] = 'general'
    # df1[i].loc[df1[i]['decision_type']=='ng', 'rationality'] = 'unbounded'
    # df1[i].loc[df1[i]['decision_type']=='indp', 'rationality'] = 'optimal'
    df2[i].loc[:, 'topology'] = 'general'
    df2[i].loc[df2[i]['decision_type']=='bgNNUU', 'rationality'] = 'bounded'
    df2[i].loc[df2[i]['decision_type']=='indp', 'rationality'] = 'optimal'
for i in [0, 1, 3, 8]:
    for x in df2[i]:
        if x not in df1[i]:
            df1[i].append(x)

for i in [2, 4, 5, 6, 7, 10]:
    df1[i] = pd.concat([df1[i],df2[i]]).drop_duplicates().reset_index(drop=True)

df1[9] = pd.concat([df1[9],df2[9]]).drop_duplicates(subset=df1[9].columns.difference(['no_payoffs'])).reset_index(drop=True)

with open(folder+'postprocess_dicts.pkl', 'wb') as f:
    pickle.dump(df1, f)

########-----------------------------------------------------------------------
# config_list_folder = '/home/hesam/Desktop/Files/Generated_Network_Dataset_v4/GeneralNetworks/'
# data = pd.read_csv(config_list_folder+'List_of_Configurations.txt',
#                  header=0, sep="\t")
# data = data.assign(topology='general')

# folder = '/home/hesam/Desktop/Files/Game_synthetic/v4/postprocess/'
# dfs = pd.read_pickle(folder+'postprocess_dicts.pkl')

# comp_df=pd.merge(dfs[4], data, left_on=['Magnitude'], right_on=['Config Number'])

# comp_df['lambda_U'] = pd.to_numeric(comp_df['lambda_U'], errors='coerce')
# comp_df['T1'] = 0
# comp_df['T2'] = 0
# for idx, row in comp_df.iterrows():
#     comp_df.loc[idx, 'T1'] = comp_df.loc[idx, ' Net Types'][2]
#     comp_df.loc[idx, 'T2'] = comp_df.loc[idx, ' Net Types'][-3]

# import seaborn as sns

# fig_df = comp_df
# #[(comp_df['decision_type']=='bgNNUU')]#&((comp_df['T1']=='s')|(comp_df['T2']=='s'))]
# g = sns.lmplot(data=fig_df, x=" Interconnection Prob", y="lambda_U",
#                      hue='decision_type',legend_out=False)
# g.ax.set_ylim([-2,0.05])
# # g.ax.set_xlim([0,100])
# # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# # comp_df.columns

