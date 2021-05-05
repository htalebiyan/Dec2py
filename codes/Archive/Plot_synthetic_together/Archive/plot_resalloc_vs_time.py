import plots
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.close('all')
sns.set(context='notebook',style='darkgrid')

# plt.rc('text', usetex=True)
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

dir_data="C:\Users\ht20\Documents\Files\Auction_synthetic_networks_v3.1\ScaleFree\objs2.pkl"
[combinations,optimal_combinations,df,method_name,lambda_df,resource_allocation,res_alloc_rel,cost_type,run_time_df] = pd.read_pickle(dir_data)
df1 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3.1\GridNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df1 = df1.assign(topology='Grid')
df2 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3.1\ScaleFreeNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df2 = df2.assign(topology='ScaleFree')
df3 = pd.read_csv("C:\Users\ht20\Documents\Files\Generated_Network_Dataset_v3.1\RandomNetworks\List_of_Configurations.txt",
                 header=0, sep="\t")
df3 = df3.assign(topology='Random')

comp_res=pd.merge(resource_allocation, df2,
             left_on=['Magnitude'],
             right_on=['Config Number']) 
""" Plot results """    
selected_df = comp_res[((comp_res['auction_type']=='MAA')|
                        (comp_res['auction_type']==''))&
                       (comp_res[' No. Layers']==4)]
# selected_df["distance_to_optimal"] = pd.to_numeric(selected_df["distance_to_optimal"])
plots.plot_auction_allocation_synthetic(selected_df,resource_type='resource', ci=None)

