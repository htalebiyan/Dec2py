import seaborn as sns
import os.path
import networkx as nx
import numpy as np
import pandas as pd
from SALib.analyze import delta
import plot

'''Compute sensitivity'''
# comp_res = pd.read_pickle('temp_synthetic_v3_1')

# df1 = pd.read_csv('C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\GridNetworks\\List_of_Configurations.txt',
#                   header=0, sep="\t")
# df1 = df1.assign(topology='Grid')
# df2 = pd.read_csv("C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\ScaleFreeNetworks\\List_of_Configurations.txt",
#                   header=0, sep="\t")
# df2 = df2.assign(topology='ScaleFree')
# df3 = pd.read_csv("C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\RandomNetworks\\List_of_Configurations.txt",
#                   header=0, sep="\t")
# df3 = df3.assign(topology='Random')
# config_info = pd.concat([df1,df2,df3])

# comp_res=pd.merge(comp_res, config_info,
#               left_on=['Magnitude','topology'],
#               right_on=['Config Number','topology']) 

# topo = 'Grid' #'ScaleFree' #'Random' #Grid
# auc = 'MAA'
# gamma_dict={'ScaleFree':[2, 3], 'Random':[0.02, 0.62], 'Grid':[3, 10]}
# problem = {
#     'num_vars': 6,
#     'names': ['L', 'N', 'gamma', 'pi', 'pd', 'Rc'],
#     'bounds': [[2, 4], [10, 50], gamma_dict[topo], [0.001, 0.1], [0.05, 0.5], [2, 400]]}
# comp_res['lambda_U'] = comp_res['lambda_U'].astype(float)
# # comp_res = comp_res[(comp_res['topology']==topo)&(comp_res['auction_type']==auc)]
# comp_res = comp_res[(comp_res['topology']==topo)&(comp_res['auction_type']!='')]
# # id_vars = [x for x in comp_res.columns if x not in ['lambda_U','lambda_P','lambda_TC',
# #                                                     'Area_TC','Area_P']]
# # comp_res = comp_res.groupby(id_vars, as_index=False)['lambda_U'].mean()
# X = pd.concat([comp_res[' No. Layers'], comp_res[' No. Nodes'], comp_res[' Topology Parameter'],
#                 comp_res[' Interconnection Prob'], comp_res[' Damage Prob'], comp_res[' Resource Cap']], 
#               axis=1, keys=['L', 'N', 'gamma', 'pi', 'pd', 'Rc']).to_numpy()
# Y = pd.concat([comp_res['lambda_U']], axis=1, keys=['lambda_U']).to_numpy().ravel()
# delta = delta.analyze(problem, X, Y, num_resamples=100, conf_level=0.95, print_to_console=True, seed=None)

# np.where(np.isnan(Y))
# np.where(np.isinf(Y))
# np.where(np.isneginf(Y))
# X.max(axis=0)
# X.min(axis=0)

'''Plot results'''

results = pd.read_csv('Results_perf.csv', header=0)
results = results[results['Resource_allocation']!='All']
results["Topology/Resource_allocation"] = results["Resource_allocation"]+'\n'+results["Topology"]
df = results.pivot_table(values='Rank', index='Parameter', columns='Topology/Resource_allocation')
df.reset_index()

plot.plot_radar(df,df.index.values,df.columns)
print(df.mean(axis=1))

# plt.figure()
# ax = sns.lineplot(x="Topology/Resource_allocation", y="Rank",
#                   hue="Parameter", style="Parameter", lw=5, ms=12,
#                   markers=True, dashes=False, data=results)
# ax.invert_yaxis()
# g = sns.catplot(x='Parameter',y='Delta',hue="Topology", col="Resource_allocation",
#                 data=results, kind="bar", height=4, aspect=.7);

