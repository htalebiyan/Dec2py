import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
from pymc3.glm import GLM
import os
import tkinter as tk
from tkinter import filedialog

print('Running on PyMC3 v{}'.format(pm.__version__))
plt.style.use('seaborn-darkgrid') 
import pickle # python3

"""Data"""
raw_df = pd.read_pickle('temp_synthetic_v3_1')

df1 = pd.read_csv("C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\GridNetworks\\List_of_Configurations.txt",
                 header=0, sep="\t")
df1 = df1.assign(topology='Grid')
df2 = pd.read_csv("C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\ScaleFreeNetworks\\List_of_Configurations.txt",
                 header=0, sep="\t")
df2 = df2.assign(topology='ScaleFree')
df3 = pd.read_csv("C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\RandomNetworks\\List_of_Configurations.txt",
                 header=0, sep="\t")
df3 = df3.assign(topology='Random')
config_info = pd.concat([df1,df2,df3])

raw_df=pd.merge(raw_df, config_info,
             left_on=['Magnitude','topology'],
             right_on=['Config Number','topology']) 

topo='Random'
auc = 'Uniform'
outputName = topo+'_'+auc


raw_df = raw_df[(raw_df['topology']==topo)& (raw_df['auction_type']==auc)]
df = pd.DataFrame({'nNodes': raw_df[' No. Nodes'], 'topoParam': raw_df[' Topology Parameter'],
                   'Pint':  raw_df[' Interconnection Prob'], 'Pdam': raw_df[' Damage Prob'],
                   'ResCap': raw_df[' Resource Cap'],'nLayers': raw_df[' No. Layers'],
                   'y': raw_df['lambda_U'].astype('float')})
df = df[df['y']!='nan']
normalized_df=(df-df.mean())/df.std()

# Modeling
priors = {"Intercept": pm.Normal.dist(mu=0, sd=10),
          "Regressor": pm.Normal.dist(mu=0, sd=10)}

with pm.Model() as model_glm:
    family = pm.glm.families.StudentT()
    GLM.from_formula('y ~ nNodes+Pint+Pdam+ResCap+topoParam+nLayers', normalized_df, family=family, priors=priors)
    trace = pm.sample(cores=1, draws = 2000, tune=1000)

# Save Output and model
pm.stats.summary(trace).to_csv('output/Output_'+outputName+'.csv')


# Write model and results to file
with open('output/model'+outputName+'.pkl', 'wb') as buff:
    pickle.dump({'model': model_glm, 'trace': trace}, buff)

with open('output/model'+outputName+'.pkl', 'rb') as buff:
    data = pickle.load(buff)  

basic_model, trace = data['model'], data['trace']

# Plots
plt.close('all')
#pm.traceplot(trace, combined=False)
#pm.plot_posterior(trace)
#pm.forestplot(trace)
#pm.autocorrplot(trace)

axInt = pm.densityplot(trace)
plt.figure(figsize=(5, 3))
#plt.rc('text', usetex=True)
#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
for i in range(1,len(axInt)-1):
    line = axInt[i].lines[0]
    xy = line.get_xydata()
    ax = plt.plot(xy[:,0], xy[:,1], linewidth=1)
    ax = plt.fill(xy[:,0], xy[:,1], alpha=0.75)
plt.legend([r'$N$',r'$P_i$',r'$P_d$',r'$R_c$',r'$\Upsilon$',r'$L$'], loc=1, frameon =True,framealpha=0.5)
plt.ylim(ymin=0)
plt.xlabel(r'$\beta_i$')
plt.ylabel(r'Probability')
plt.savefig('output/PosteriorDist_'+outputName+'.pdf', dpi=600)
plt.show()
