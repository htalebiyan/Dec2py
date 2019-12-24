import plots
import seaborn as sns
import os.path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import plotly

plt.close('all')
sns.set(context='notebook',style='darkgrid')

plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


#run_time_df = run_time_df.assign(topology='Grid',interdependency='full')
#comp_res = pd.DataFrame(run_time_df)

comp_res = pd.read_pickle('temp_run')
#comp_res = pd.concat([comp_res,run_time_df])

#comp_res.to_pickle('temp_run') 

df1 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3.1\GridNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df1 = df1.assign(topology='Grid')
df2 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3.1\ScaleFreeNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df2 = df2.assign(topology='ScaleFree')
df3 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3.1\RandomNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df3 = df3.assign(topology='Random')
config_info = pd.concat([df1,df2,df3])

comp_res=pd.merge(comp_res, config_info,
             left_on=['Magnitude','topology'],
             right_on=['Config Number','topology']) 
#
""" Plot results """    
selected_df = comp_res[(comp_res['decision_time']!='Uniform')]
selected_df["decision_time"] = pd.to_numeric(selected_df["decision_time"])

selected_df = selected_df.rename(columns={"auction_type": "Auction Type",
                                    "valuation_time": "Valuation Time",
                                    'topology':'Topology'})
g = sns.barplot(x=' No. Layers', y='Valuation Time', hue='Topology',data=selected_df,
                 palette="Blues",linewidth=0.5,edgecolor=[.25,.25,.25],
                 capsize=.05,errcolor=[.25,.25,.25],errwidth=1,)
handles, labels = g.get_legend_handles_labels()   
lgd = g.legend(handles, labels, title='Topology',loc='upper left', bbox_to_anchor=(0, 1),
           frameon =True,framealpha=0.5, ncol=1)  
g.set_ylabel(r'Valuation Time (sec)')
g.set_xlabel(r'Number of Layers') 
plt.savefig('valuation_time.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')    #, bbox_inches='tight'

#g = sns.catplot(x=' No. Layers', y='auction_time', hue='auction_type',data=comp_res,
#                 kind='bar',palette="Reds",
#                 linewidth=0.5,edgecolor=[.25,.25,.25],
#                 capsize=.05,errcolor=[.25,.25,.25],errwidth=1,)#,row='layer'

#selected_df.loc[selected_df['decision_type']=='indp','auction_type']='iINDP'
#g = sns.barplot(x='auction_type', y='decision_time',
#                hue='topology',data=selected_df,palette="Reds",
#                 linewidth=0.5,edgecolor=[.25,.25,.25],
#                 capsize=.05,errcolor=[.25,.25,.25],errwidth=1,zorder=0.75)#,row='layer'
#clrs=['b','g','k']
#for nol in [2,3,4]:
#    sns.set_palette(sns.color_palette([clrs[nol-2],clrs[nol-2],clrs[nol-2]]))
#    g = sns.pointplot(x="auction_type", y="decision_time",hue="topology",
#                      data=selected_df[selected_df[' No. Layers']==nol],
#                      dodge=.532,join=False,markers=["+",'o','s'], ci=None,zorder=nol*10)
#handles, labels = g.get_legend_handles_labels()   
#lgd = g.legend(handles, labels,loc='lower left', bbox_to_anchor=(-.1, 1),
#           frameon =True,framealpha=0.5, ncol=4)   
#g.set_ylabel(r'Decision Time (sec)')
#g.set_xlabel(r'Auction Type')   
#plt.savefig('Decide_time.png', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')    #, bbox_inches='tight'  
""" Parallel Axes"""
#cols=list(selected_df.columns.values)
#selected_df = comp_res[(comp_res['decision_time']!='nan')]
#width = 0.5  
#cat_columns = list(selected_df.select_dtypes(['object']).columns)
#cat_columns.append(' No. Layers')
#mapping = {}
#for cc in cat_columns:
#    mapping[cc] = dict(enumerate(selected_df[cc].astype('category').cat.categories))
#    if cc!= ' No. Layers':
#        selected_df[cc] = selected_df[cc].astype('category').cat.codes   
#selected_df[cat_columns]=selected_df[cat_columns].add(np.random.rand(selected_df[cat_columns].shape[0], selected_df[cat_columns].shape[1])*width)
#
#dimensions = list([
#    dict(range = [0,0.025],
#         label = ' Interconnection Prob', values = selected_df[' Interconnection Prob']),
#    dict(range = [5,50],
#         label = ' No. Nodes', values = selected_df[' No. Nodes']),
#    dict(range = [0,1+width],
#         label = 'decision_type', values = selected_df['decision_type']),
#    dict(range = [0,10],
#         label = ' Topology Parameter', values = selected_df[' Topology Parameter']),
#    dict(range = [0,2+width],
#         label = 'topology', values = selected_df['topology']),
#    dict(range = [0,400],
#         label = ' Resource Cap', values = selected_df[' Resource Cap']),
#    dict(range = [0,1+width],
#         label = 'valuation_type', values = selected_df['valuation_type']),
#    dict(range = [0,3+width],
#         label = 'auction_type', values = selected_df['auction_type']),
#    dict(range = [2,4+width],
#         label = ' No. Layers', values = selected_df[' No. Layers']),
#    dict(range = [0.05,0.5],
#         label = ' Damage Prob', values = selected_df[' Damage Prob']),
#    dict(range = [0,10],
#         label = 't', values = selected_df['t'])
##            dict(range = [0,2+width],
##                 label = 'interdependency', values = selected_df['interdependency']),
#    ])
#fig = go.Figure(data=
#    go.Parcoords(
#        line = dict(color = selected_df['decision_time'],
#                   colorscale = 'Electric',
#                   showscale = True,cmin=0,cmax=25),
#        dimensions = dimensions
#    )
#)
#
#plotly.offline.plot(fig)
##

"""Other plots"""
#selected_df = comp_res[(comp_res['decision_type']!='indp')]
#selected_df["decision_time"] = pd.to_numeric(selected_df["decision_time"])

#f, ax = plt.subplots()
#sns.despine(bottom=True, left=True)
#g=sns.stripplot(x="auction_time", y="auction_type", hue=" No. Layers",
#              data=selected_df, dodge=True, jitter=True,
#              alpha=.25, zorder=1)
#g.set(xlim=(-.5, 5))
## Show the conditional means
#g=sns.pointplot(x="auction_time", y="auction_type", hue=" No. Layers",
#              data=selected_df, dodge=.532, join=False, palette="dark",
#              markers="d", scale=.75, ci=None)

#f, ax = plt.subplots()
#sns.scatterplot(y="distance_to_optimal", x=" Resource Cap",
#                hue="auction_type", size=" No. Layers",
#                sizes=(3, 8), linewidth=0,
#                data=selected_df)

##Draw the two density plots
#ax = sns.kdeplot(selected_df[selected_df[' No. Layers']==4][" Resource Cap"],
#                 selected_df[selected_df[' No. Layers']==4]["auction_type"],
#                 cmap="Reds", shade=True, shade_lowest=False, bw='scott',cbar=True)
#ax = sns.kdeplot(selected_df[selected_df[' No. Layers']==2][" Resource Cap"], 
#                 selected_df[selected_df[' No. Layers']==2]["auction_type"],
#                 cmap="Blues", shade=True, shade_lowest=False, bw='scott',cbar=True)


'''Scatter'''
#selected_df = selected_df.rename(columns={"norm_distance_to_optimal": "norm distance to optimal",
#                                          "auction_type": "Auction Type",
#                                          "topology":"Topology"})
 
#sns.set(font_scale=1.5) 
#with sns.xkcd_palette(['black',"windows blue",'red',"green"]): #sns.color_palette("muted"):
#    g=sns.relplot(x=" Resource Cap", y="decision_time",
#            hue="topology", size='topology', col="auction_type",
#            legend="full", data=selected_df)
#    g.set(xlim=(-5, 100))
#    g.set(ylim=(-5, 1000))
#    g.set_xlabels(r'$R_c$')
#    g.set_ylabels(r'Mean $\omega^k(r^k_d,r^k_c)$ over layers')
#    g.set_titles(col_template = 'Auction Type: {col_name}')
#    g._legend.set_bbox_to_anchor([0.89, 0.75])    
    
#plt.savefig('OmegaVsRc.pdf', dpi=600)    #, bbox_inches='tight'
'''End Scatter'''

#g = sns.catplot(x="auction_type", y="valuation_time",
#                 hue="valuation_type", col="topology",
#                 data=selected_df, kind="violin", split=False, inner="quartile",
#                 height=4, aspect=.7);

#cols = ['distance_to_optimal', ' No. Layers', ' Interconnection Prob', ' Damage Prob', ' Resource Cap', ' No. Nodes']
#pp = sns.pairplot(selected_df[cols], size=1.8, aspect=1.8, 
#                  palette={"red": "#FF9999", "white": "#FFE888"},
#                  plot_kws=dict(edgecolor="black", linewidth=0.5))

""" PCA"""
#from sklearn.preprocessing import StandardScaler
#features = ['auction_time','decision_time','valuation_time',' No. Layers',' Interconnection Prob',' Damage Prob',
#            ' Resource Cap',' No. Nodes',' Topology Parameter']
## Separating out the features
#x = selected_df.loc[:, features].values
## Separating out the target
#y = selected_df.loc[:,['topology']].values
## Standardizing the features
#x = StandardScaler().fit_transform(x)
#
#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#finalDf = pd.concat([principalDf, selected_df[['topology']]], axis = 1)
#
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component PCA', fontsize = 20)
#targets = ['Random', 'ScaleFree','Grid']
#colors = ['r', 'g', 'b']
#for target, color in zip(targets,colors):
#    indicesToKeep = finalDf['topology'] == target
#    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#               , finalDf.loc[indicesToKeep, 'principal component 2']
#               , c = color
#               , s = 50)
#ax.legend(targets)
#ax.grid()
#
#pca.explained_variance_ratio_
