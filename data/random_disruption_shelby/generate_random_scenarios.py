'''
Each node or arc is this database is damages with probability p where p increases 
from 1% to 99% with sample number. For each value, 10 samples are generated, 
which makes a total number of 990 samples. Samples are stacked in an 
increasing order in terms of p. For example, the first 10 samples have p=1%, 
the second 10 samples have p=2%, and so on and so forth.
'''
import numpy as np
import matplotlib.pyplot as plt

# p_values = np.linspace(0.01,0.99,99)
# node_samples=[]
# for p in p_values:
#     temp=np.random.choice([0,1],(167,10),p=[p,1-p])
#     if p==0.01:
#         node_samples=temp
#     else:
#         node_samples=np.hstack([node_samples,temp])

# arc_samples=[]
# for p in p_values:
#     temp=np.random.choice([0,1],(213,10),p=[p,1-p])
#     if p==0.01:
#         arc_samples=temp
#     else:
#         arc_samples=np.hstack([arc_samples,temp]) 

# initial=initial_net.G.edges()
# edges = []
# inter = []
# for i in initial:
# 	u=i[0]
# 	v=i[1]
# 	if u[1]!=v[1]:
#         inter.append(i)
#         if (v,u) in edges:
#             print((v,u))
#     elif (v,u) not in edges:       
#         edges.append(i)
        
# ax=plt.imshow(node_samples)      
# plt.xlabel('scenario')
# plt.ylabel('element')  
# plt.xlim(0,500)

import seaborn as sns
plt.close('all')
sns.set(context='notebook',style='darkgrid')
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
f,ax=plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [2,2]})#f,ax=plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [2,2,2.6]})
folder='C:/Users/ht20\Documents/GitHub/td-DINDP/data/random_disruption_shelby/'
node_data=np.genfromtxt(folder+'Initial_node.csv',delimiter=',')
arc_data=np.genfromtxt(folder+'Initial_links.csv',delimiter=',')
num_sce = 900

sns.heatmap(node_data[:,2:], xticklabels=100, yticklabels=False,ax=ax[1], cbar=False,  cmap="YlGnBu")
ax[1].set_ylabel('Nodes')
ax[1].set_xlabel('Scenario')

# sns.heatmap(arc_data[:,4:], xticklabels=25, yticklabels=False,ax=ax[2], cbar=False,  cmap="YlGnBu")
# ax[2].set_xlabel('Disruption Scenario')
# ax[2].set_ylabel('Arcs')

data = node_data[:,2:] #np.vstack((node_data[:,2:],arc_data[:,4:]))
means=1-data.mean(axis=0)
sns.lineplot(x=range(990),y=means,ax=ax[0],lw=1)
ax[0].set_ylabel(r'\% Nodes Damaged')

ax[1].set_xticklabels(np.arange(-50, -50+num_sce+1, 100))
plt.xlim(50,num_sce+50+1)
plt.savefig('Disruption_scenarios.png',dpi=600,bbox_inches='tight')