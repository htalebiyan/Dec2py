import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import Import_Data_Shelby
import pandas as pd
import os 

noSets = 5
noScenarios = 96
resCap = 3
T = 10      

samples = Import_Data_Shelby.importData(noScenarios,noSets,T,resCap)
          
names = list(samples.keys())
samplesNodes = {}
for i in samples:
    if i[0] == 'w' and i[-1]=='P':
        samplesNodes[i] = samples[i]
noSpa = len(samplesNodes.keys())
names = list(samplesNodes.keys())
noTmp = T+1
    
noSamples = 10
p0 = {}
sdy = {}
y_t = {names[x]:np.zeros((noTmp,noSamples)) for x in range(noSpa)} 
y_t_mean = {}
y_t_sd = {}
y_t_rounded = {}
y_cmpr = {}

noScenarios = samples[names[0]].shape[1]
for i in range(noSpa):
    key = names[i]
    if key[-1]=='P':
        keySplit = key.split('_')
        namefile = 'w_%d_%s'%(int(keySplit[1]),keySplit[2])
        datafile = 'Parameters\\model_parameters_%s.txt'%namefile
        if os.path.exists(datafile):
            param = pd.read_csv(datafile, delimiter = ' ', index_col=0)
            mean =  param['mean']['sdy_'+namefile]
            sd =  param['sd']['sdy_'+namefile]
            sdy[i] = (mean,sd)
            
            name = 'p(' + namefile + ')_' + str(0)
            mean =  param['mean'][name]
            sd =  param['sd'][name]
            p0[i] = (mean,sd)
        else:
            p0[i] = (1.0,0)

counter = 0
pRealized = {}
for sce in range(noScenarios):                         
    for s in range (noSamples):
        for i in range(noSpa):
            key = names[i]
            y_t[key][0,s] = samples[key][0,sce]
            if p0[i][1] != 0.0:
                p0Dist = pm.Beta.dist(mu=p0[i][0], sd=p0[i][1])
                pRealized[key,0] = p0Dist.random(size=1)
                
                for t in range(1,noTmp):
                    sdyRealized = sdy[i][0] + np.random.randn(1)*sdy[i][1]
                    y_t[key][t,s] = 0
                    if y_t[key][t-1,s] != 1.0:
                        pRealized[key,t] = (1/(1+np.exp(-pRealized[key,t-1]-sdyRealized)))[0]
                        trial = np.random.binomial(n=11, p=pRealized[key,t])
                        if trial/11>0.5:
                            y_t[key][t,s] = 1.0 
                    else:
                        y_t[key][t,s] = 1.0
            else:
                for t in range(1,noTmp):
                    y_t[key][t,s] = 1.0
                    
                        
    y_cmpr[sce] = np.zeros((noSpa,noTmp))                   
    for i in range(noSpa):
        key = names[i]
        if key[-1]=='P':
            y_t_mean[key] = np.mean(y_t[key],axis=1)
            y_t_sd[key] = np.std(y_t[key],axis=1)
            y_t_rounded[key] = np.zeros_like(y_t_mean[key]) 
            y_t_rounded[key][0] = samples[key][0,sce]
            for t in range(1,noTmp):
                if y_t_mean[key][t]>=0.5:
                    y_t_rounded[key][t] = 1.0
                else:
                    y_t_rounded[key][t] = 0.0
    #            if y_t_rounded[key][t-1] == 1.0:
    #                y_t_rounded[key][t] = 1.0
                
            for t in range(noTmp):
                if y_t_rounded[key][t]!=samples[key][t,sce]:
                    y_cmpr[sce][i,t] = 1.0
    print('Scenario %d' % sce)
                    
fig = plt.figure()
counter = 0
for sce in range(noScenarios):    
    counter += 1
    ax = fig.add_subplot(7,8,counter)
    ax.imshow(y_cmpr[sce], aspect=0.08)
    
count = 0
sum_y_cmpr = np.zeros((noSpa,noTmp))    
for key in y_cmpr:
    count += 1
    sum_y_cmpr += y_cmpr[key]
   
plt.figure()
plt.imshow(sum_y_cmpr/count, aspect=0.08)        
plt.colorbar()  
            
     


#''' Generate and plot samples from the trained model '''
#ppc = pm.sample_ppc(trace, samples=1000, model=model)
#varName = 'w_1_P'
#_, ax = plt.subplots(figsize=(12, 6))
#pred = [n.mean() for n in ppc[varName+'_0']]
#ax.hist(pred, bins=20, alpha=0.5, density=True)
#ax.axvline(samples[varName][0,:].mean())
#ax.set(title='Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency');
#
