import pymc3 as pm
import pandas as pd
import Import_Data_Shelby
print('Running with PyMC3 version v.{}'.format(pm.__version__))
import copy
import numpy as np
import matplotlib.pyplot as plt

def logistic(l):
    return 1 / (1 + pm.math.exp(-l))

def train_model(samples, samplesDiff, T, resCap, trainSet, v):
    names = list(samples.keys())
    samplesNodes = {}
    noSpa = len(samples.keys())
    names = list(samples.keys())
    noTmp = T+1
    
    p = {}
    sdy = {}
    y_t = {}
    
    conv = pd.read_csv('filteredVarsFromCrudeModel.txt', delimiter = ' ')
    for i in range(v,v+1):
        estParamAll = pd.DataFrame()
        key = names[i]
        
        y_t[i,0] = samplesDiff[key][0,:]
        ind = conv.index[(conv['name'] == key)].tolist()
        noTimeSteps = len(ind)
        if noTimeSteps != 0:
            with pm.Model() as model:
                p[i,0] = pm.Beta('p(%s)_%d'%(key,0), alpha=2, beta=2)
                sdy[i] = pm.Normal('sdy_%s'%(key), mu=0, sd=1)

                for j in ind:
                    if (i==75 and j==31) or (i==83 and j==50) or \
                    (i==83 and j==52) or (i==86 and j==61) or \
                    (i==86 and j==62) or (i==89 and j==68) or (i==95 and j==71)\
                    or (i==107 and j==75) or (i==108 and j==82):
                            if (i==75 and j==31) or (i==95 and j==71):
                                t = 2
                            elif (i==83 and j==50) or (i==107 and j==75):
                                t = 1
                            elif (i==89 and j==68) or (i==108 and j==82):
                                t = 3
                            elif (i==83 and j==52) or (i==86 and j==61):
                                t = 4
                            elif (i==86 and j==62):
                                t = 6
                            index = np.argwhere(samplesDiff[key][t,:]==2)
                            trainData = np.delete(samplesDiff[key][t,:],index)
                            p[i,t] = pm.Deterministic('p(%s)_%d'%(key,t),
                                                 logistic(p[i,t-1]+sdy[i]))
                            y_t[i,t] = pm.Bernoulli('%s_%d'%(key,t),
                                                       p = p[i,t],
                               observed=trainData)          
                            
                    t = conv['time'][j]
                    index = np.argwhere(samplesDiff[key][t,trainSet]==2)
                    trainData = np.delete(samplesDiff[key][t,trainSet],index)
                    p[i,t] = pm.Deterministic('p(%s)_%d'%(key,t),
                                         logistic(p[i,t-1]+sdy[i]))
                    y_t[i,t] = pm.Bernoulli('%s_%d'%(key,t),
                                               p = p[i,t],
                                               observed=trainData)
                trace = pm.sample(2000, tune=500, cores=2, njobs=1)
                estParam = pm.summary(trace).round(4)
                print(estParam)
                estParam['Name'] = key 
               
                estParam.to_csv('Parameters\\model_parameters_%s.txt' % names[i],
                                header=True, index=True, sep=' ', mode='w')