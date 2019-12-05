import plots
import seaborn as sns
import os.path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

plt.close('all')
sns.set(context='notebook',style='darkgrid')

#plt.rc('text', usetex=True)
#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


#lambda_df = lambda_df.assign(topology='ScaleFree',interdependency='quarter')
#comp_lambda_df = pd.DataFrame(lambda_df)
##
comp_lambda_df = pd.read_pickle('temp')
#comp_lambda_df = pd.concat([comp_lambda_df,lambda_df])
#
#comp_lambda_df.to_pickle('temp') 

df1 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3\GridNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df1 = df1.assign(topology='Grid')
df2 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3\ScaleFreeNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df2 = df2.assign(topology='ScaleFree')
df3 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3\RandomNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df3 = df3.assign(topology='Random')
config_info = pd.concat([df1,df2,df3])

comp_lambda_df=pd.merge(comp_lambda_df, config_info,
             left_on=['Magnitude','topology'],
             right_on=['Config Number','topology']) 

""" Plot results """    
#selected_df = comp_lambda_df[(comp_lambda_df['lambda_TC']!='nan')]
#selected_df["lambda_TC"] = pd.to_numeric(selected_df["lambda_TC"])
#g = sns.catplot(x='auction_type', y='lambda_TC', hue='interdependency',
#                 col=' No. Layers', row='topology',data=selected_df,
#                 kind='bar',palette="Blues",
#                 linewidth=0.5,edgecolor=[.25,.25,.25],
#                 capsize=.05,errcolor=[.25,.25,.25],errwidth=1,)
#g.set(ylim=(-1, 0))
#cols=list(selected_df.columns.values)


selected_df = comp_lambda_df[(comp_lambda_df['lambda_TC']!='nan')&
                             (comp_lambda_df['lambda_TC']<-0.1)]
selected_df["lambda_TC"] = pd.to_numeric(selected_df["lambda_TC"])
width = 0.5  
cat_columns = list(selected_df.select_dtypes(['object']).columns)
cat_columns.append(' No. Layers')
mapping = {}
for cc in cat_columns:
    mapping[cc] = dict(enumerate(selected_df[cc].astype('category').cat.categories))
    if cc!= ' No. Layers':
        selected_df[cc] = selected_df[cc].astype('category').cat.codes   
selected_df[cat_columns]=selected_df[cat_columns].add(np.random.rand(selected_df[cat_columns].shape[0], selected_df[cat_columns].shape[1])*width)
    
import plotly.graph_objects as go
import plotly
fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = selected_df['lambda_TC'],
                   colorscale = 'Electric',
                   showscale = True,cmin=-0.5,cmax=0),
        dimensions = list([
            dict(range = [0,0.025],
                 label = ' Interconnection Prob', values = selected_df[' Interconnection Prob']),
            dict(range = [5,50],
                 label = ' No. Nodes', values = selected_df[' No. Nodes']),
            dict(range = [0,1+width],
                 label = 'decision_type', values = selected_df['decision_type']),
            dict(range = [0,10],
                 label = ' Topology Parameter', values = selected_df[' Topology Parameter']),
            dict(range = [0,2+width],
                 label = 'topology', values = selected_df['topology']),
            dict(range = [0,120],
                 label = ' Resource Cap', values = selected_df[' Resource Cap']),
            dict(range = [0,1+width],
                 label = 'valuation_type', values = selected_df['valuation_type']),
            dict(range = [0,3+width],
                 label = 'auction_type', values = selected_df['auction_type']),
            dict(range = [2,4+width],
                 label = ' No. Layers', values = selected_df[' No. Layers']),
            dict(range = [0.05,0.25],
                 label = ' Damage Prob', values = selected_df[' Damage Prob']),
            dict(range = [0,2+width],
                 label = 'interdependency', values = selected_df['interdependency']),
            ])
    )
)

plotly.offline.plot(fig)