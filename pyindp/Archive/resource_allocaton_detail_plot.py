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
sns.despine()

#plt.rc('text', usetex=True)
#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

df2 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3.1\GridNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df2 = df2.assign(topology='Grid')

comp_res=pd.merge(resource_allocation, df2,
             left_on=['Magnitude'],
             right_on=['Config Number']) 

df_rees = comp_res[((comp_res['auction_type']=='MAA')|(comp_res['auction_type']==''))&(comp_res[' No. Layers']==4)]
plots.plot_auction_allocation_synthetic(df_rees,ci=0,resource_type='normalized_resource')