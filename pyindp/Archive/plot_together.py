import plots
import seaborn as sns
import os.path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
sns.set(context='notebook',style='darkgrid')

#plt.rc('text', usetex=True)
#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


#lambda_df = lambda_df.assign(topology='ScaleFree',interdependency='full')
###comp_lambda_df = pd.DataFrame(lambda_df)
#
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
#    plot_performance_curves_shelby(df,cost_type='Total',decision_names=method_name,ci=None,normalize=True)
#    plot_relative_performance_shelby(lambda_df)
#    plot_auction_allocation_shelby(resource_allocation,ci=None)
#    plot_relative_allocation_shelby(res_alloc_rel)

#plot_performance_curves_synthetic(df,ci=None,x='t',y='normalized_cost')    
#plots.plot_relative_performance_synthetic(comp_lambda_df)  
#plot_auction_allocation_synthetic(resource_allocation,ci=None,resource_type='normalized_resource')
#plot_relative_allocation_synthetic(res_alloc_rel)

selected_df = comp_lambda_df[(comp_lambda_df['lambda_TC']!='nan')] 
selected_df["lambda_TC"] = pd.to_numeric(selected_df["lambda_TC"])
g = sns.catplot(x='auction_type', y='lambda_TC', hue='decision_type',
                 col='interdependency', row='topology',data=selected_df,
                 kind='bar',palette="Blues",
                 linewidth=0.5,edgecolor=[.25,.25,.25],
                 capsize=.05,errcolor=[.25,.25,.25],errwidth=1,)
g.set(ylim=(-1, 0))
#g = sns.FacetGrid(selected_df, col="interdependency", hue="decision_type", height=4, aspect=1,palette="Blues",edgecolor = 'w')
#g.map(sns.barplot, "auction_type", "lambda_TC");