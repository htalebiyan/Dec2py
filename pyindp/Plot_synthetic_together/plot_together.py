import plots
import seaborn as sns
import os.path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

plt.close('all')
sns.set(context='notebook',style='whitegrid')

plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


#lambda_df = lambda_df.assign(topology='Grid',interdependency='full')
#comp_lambda_df = pd.DataFrame(lambda_df)
#
comp_lambda_df = pd.read_pickle('temp_synthetic_v3_1_MDDN')
#comp_lambda_df = pd.concat([comp_lambda_df,lambda_df])
#
#comp_lambda_df.to_pickle('temp_synthetic_v3_1_MDDN') 

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

comp_lambda_df=pd.merge(comp_lambda_df, config_info,
             left_on=['Magnitude','topology'],
             right_on=['Config Number','topology']) 

""" Plot results """    
selected_df = comp_lambda_df[(comp_lambda_df['lambda_U']!='nan')]
selected_df["lambda_TC"] = pd.to_numeric(selected_df["lambda_U"])

selected_df = selected_df.rename(columns={"lambda_U": "lambda U",
                                          "auction_type": "Auction Type",
                                          "topology":"Topology"})
sns.set(font_scale=1.5) 
g = sns.catplot(x=' No. Layers', y='lambda U', hue='Auction Type',
                 col='Topology',data=selected_df,
                 kind='bar',palette="Reds",
                 linewidth=0.5,edgecolor=[.25,.25,.25],
                 capsize=.05,errcolor=[.25,.25,.25],errwidth=1,
                 height=6, aspect=0.5)

#g.set(ylim=(-0.65, 0))
g.axes[0,0].set_ylabel(r'$E[\lambda_U]$')
g.axes[0,0].set_xlabel(r'Number of Layers')
g.axes[0,1].set_xlabel(r'Number of Layers')
g.axes[0,2].set_xlabel(r'Number of Layers')
g.axes[0,0].spines['bottom'].set_visible(False)
g.axes[0,0].spines['left'].set_visible(False)
g.axes[0,1].spines['bottom'].set_visible(False)
g.axes[0,1].spines['left'].set_visible(False)
g.axes[0,2].spines['bottom'].set_visible(False)
g.axes[0,2].spines['left'].set_visible(False)
g.set_titles(col_template = 'Topology: {col_name}')
handles, labels = g.axes[0,2].get_legend_handles_labels()   
labels[0]='MCA+MDDN'
lgd = g.axes[0,2].legend(handles, labels,loc='lower right', bbox_to_anchor=(1.025, 0),
           frameon =True,framealpha=0.9, ncol=1, title='Auction Type') 
plt.savefig('Relative Performance.pdf', dpi=600, bbox_extra_artists=(lgd,))    #, bbox_inches='tight'
""" Parallel Axes"""
#cols=list(selected_df.columns.values)
#selected_df = comp_lambda_df[(comp_lambda_df['lambda_TC']!='nan')&
#                             (comp_lambda_df['lambda_TC']<-0.1)]
#selected_df["lambda_TC"] = pd.to_numeric(selected_df["lambda_TC"])
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
#import plotly.graph_objects as go
#import plotly
#fig = go.Figure(data=
#    go.Parcoords(
#        line = dict(color = selected_df['lambda_TC'],
#                   colorscale = 'Electric',
#                   showscale = True,cmin=-0.5,cmax=0),
#        dimensions = list([
#            dict(range = [0,0.025],
#                 label = ' Interconnection Prob', values = selected_df[' Interconnection Prob']),
#            dict(range = [5,50],
#                 label = ' No. Nodes', values = selected_df[' No. Nodes']),
#            dict(range = [0,1+width],
#                 label = 'decision_type', values = selected_df['decision_type']),
#            dict(range = [0,10],
#                 label = ' Topology Parameter', values = selected_df[' Topology Parameter']),
#            dict(range = [0,2+width],
#                 label = 'topology', values = selected_df['topology']),
#            dict(range = [0,400],
#                 label = ' Resource Cap', values = selected_df[' Resource Cap']),
#            dict(range = [0,1+width],
#                 label = 'valuation_type', values = selected_df['valuation_type']),
#            dict(range = [0,3+width],
#                 label = 'auction_type', values = selected_df['auction_type']),
#            dict(range = [2,4+width],
#                 label = ' No. Layers', values = selected_df[' No. Layers']),
#            dict(range = [0.05,0.5],
#                 label = ' Damage Prob', values = selected_df[' Damage Prob']),
##            dict(range = [0,2+width],
##                 label = 'interdependency', values = selected_df['interdependency']),
#            ])
#    )
#)
#
#plotly.offline.plot(fig)


"""Other plots"""
#selected_df = comp_lambda_df[(comp_lambda_df['lambda_U']!='nan')]
#selected_df["lambda_U"] = pd.to_numeric(selected_df["lambda_U"])
##
#f, ax = plt.subplots()
#sns.despine(bottom=True, left=True)
#sns.stripplot(x="lambda_U", y="auction_type", hue=" No. Layers",
#              data=selected_df, dodge=True, jitter=True,
#              alpha=.25, zorder=1)
#
## Show the conditional means
#sns.pointplot(x="lambda_U", y="auction_type", hue=" No. Layers",
#              data=selected_df, dodge=.532, join=False, palette="dark",
#              markers="d", scale=.75, ci=None)
#
#selected_df = selected_df.rename(columns={"norm_distance_to_optimal": "norm distance to optimal",
#                                          "auction_type": "Auction Type",
#                                          "topology":"Topology"})
 
#sns.set(font_scale=1.5) 
#with sns.xkcd_palette(['black',"windows blue",'red',"green"]): #sns.color_palette("muted"):
#    g=sns.relplot(x=" Resource Cap", y="lambda_U",
#            hue="topology", size='topology', col="auction_type",
#            legend="full", data=selected_df)
#    g.set(ylim=(-2, .1))
#    g.set(xlim=(-5, 150))
#    g.set_xlabels(r'$R_c$')
#    g.set_ylabels(r'Mean $\lambda_U$ over layers')
#    g.set_titles(col_template = 'Auction Type: {col_name}')
#    g._legend.set_bbox_to_anchor([0.89, 0.75])    
    
#plt.savefig('OmegaVsRc.pdf', dpi=600) 

## Draw the two density plots
#ax = sns.kdeplot(selected_df[selected_df[' No. Layers']==2][" Damage Prob"],
#                 selected_df[selected_df[' No. Layers']==2]["lambda_U"],
#                 cmap="Reds", shade=True, shade_lowest=False)
#ax = sns.kdeplot(selected_df[selected_df[' No. Layers']==4][" Damage Prob"], 
#                 selected_df[selected_df[' No. Layers']==4]["lambda_U"],
#                 cmap="Blues", shade=True, shade_lowest=False)

#sns.relplot(x=" Damage Prob", y="lambda_U",
#            hue="auction_type", size="topology", col=" No. Layers",
#            height=5, aspect=.75, legend="full", data=selected_df)

#g = sns.catplot(x="auction_type", y="lambda_U",
#                 hue="topology", col=" No. Layers",
#                 data=selected_df, kind="violin", split=True, inner="quartile",
#                 height=4, aspect=.7);

#cols = ['lambda_U', ' No. Layers', ' Interconnection Prob', ' Damage Prob', ' Resource Cap', ' No. Nodes']
#pp = sns.pairplot(selected_df[cols], size=1.8, aspect=1.8, 
#                  palette={"red": "#FF9999", "white": "#FFE888"},
#                  plot_kws=dict(edgecolor="black", linewidth=0.5))

""" PCA"""
#from sklearn.preprocessing import StandardScaler
#features = ['lambda_U',' No. Layers',' Interconnection Prob',' Damage Prob',
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
#targets = ['Random', 'ScaleFree']
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
