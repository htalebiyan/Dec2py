import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import Import_Data_Shelby
import STAR_Net_Nodes_
import pandas as pd
import os 
import random
import copy

noSets = 50
noScenarios = 96
resCap = 3
T = 10      
v = 83

#samples = Import_Data_Shelby.importData(noScenarios,noSets,T,resCap)
samplesDiff = copy.deepcopy(samples)
for key, val in samples.items(): 
    for s in range(val.shape[1]):
        for t in range(1,T+1):
            if val[t,s]==1:
                if val[t-1,s] == 0:
                    samplesDiff[key][t,s] = 1
                elif val[t-1,s] == 1:
                    samplesDiff[key][t,s] = 2
                else:
                    print('???')
            elif  val[t,s]==0 and val[t-1,s]==0: 
                samplesDiff[key][t,s] = 0 
            else:
                print('???')

noScenarios = samples['w_0_W'].shape[1]
setAll = np.array(range(noScenarios))
noTrainData = int(noScenarios*0.85)

meanDiffCV = np.zeros((10,T+1))
for cv in range(10):   
    random.shuffle(setAll)
    setTrain = setAll[0:noTrainData]
    setTest= setAll[noTrainData:noScenarios]
    STAR_Net_Nodes_.train_model(samples, samplesDiff, T, resCap, setTrain, v)
    
    names = list(samples.keys())
    noSpa = 1
    names = list(samples.keys())
    noTmp = T+1
        
    noSamples = 20
    p0 = {}
    sdy = {}
    y_t = {names[v]:np.zeros((noTmp,noSamples))} 
    y_t_mean = {}
    y_t_sd = {}
    y_t_rounded = {}
    y_cmpr = {}
    
    noScenariosTest = len(setTest)
    i=v
    key = names[i]
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
    for sce in setTest:                         
        for s in range (noSamples):
            i = v
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
                        
                            
        y_cmpr[sce] = np.zeros(noTmp)                   
        i = v
        key = names[i]
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
                y_cmpr[sce][t] = 1.0
        print('Scenario %d' % sce)
                        
#    fig = plt.figure()
    sum_y_cmpr = np.zeros((noScenariosTest,noTmp)) 
    counter = 0
    for sce in setTest:    
        for t in range(noTmp):
            sum_y_cmpr[counter,t] = y_cmpr[sce][t]
        counter += 1
#    plt.imshow(sum_y_cmpr, aspect=.1)
    
    meanDiffCV[cv,:] = np.mean(sum_y_cmpr,axis=0)
                
     


plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.imshow(meanDiffCV)
plt.xlabel('Time')
plt.ylabel('Iteration')
cbar = plt.colorbar()
cbar.set_label('Mean Error')

plt.savefig('MeanCVError_w_%d_P.pdf'%v, dpi=300, bbox_inches='tight')
