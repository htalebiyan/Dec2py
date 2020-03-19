import pymc3 as pm
import pandas as pd
import Import_Data_Shelby
print('Running with PyMC3 version v.{}'.format(pm.__version__))
import copy
import numpy as np
import matplotlib.pyplot as plt
def logistic(l):
    return 1 / (1 + pm.math.exp(-l))

def correct() :
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
noSets = 50
noScenarios = 96
resCap = 3
T = 10      

samples = Import_Data_Shelby.importData(noScenarios,noSets,T,resCap)
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

      
names = list(samples.keys())
samplesNodes = {}
for i in samples:
    if i[0] == 'w':
        samplesNodes[i] = samples[i]
noSpa = len(samplesNodes.keys())
names = list(samplesNodes.keys())
noTmp = T+1

p = {}
sdy = {}
y_t = {}

conv = pd.read_csv('filteredVarsFromCrudeModel.txt', delimiter = ' ')
for i in range(noSpa):
    estParamAll = pd.DataFrame()
    key = names[i]
    if key[-1]=='P':
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
                        correct()                    
                    t = conv['time'][j]
                    index = np.argwhere(samplesDiff[key][t,:]==2)
                    trainData = np.delete(samplesDiff[key][t,:],index)
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

      
#pm.traceplot(trace, combined=True) # ,varnames=['p_w_0_W_6']
#pm.model_to_graphviz(model).render()
#pp(y_t[1,1])

''' Generate and plot samples from the trained model '''
ppc = pm.sample_ppc(trace, samples=1000, model=model)


plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
times = [1,3,5]
v = 83
f, ax = plt.subplots(1, len(times), sharex=False, sharey=True,figsize=(12,4))
f.text(0.5, 0.01, 'State Transition Value', ha='center', va='center')
f.text(0.1, 0.5, 'Density', ha='center', va='center', rotation='vertical')
#f.tight_layout()

for i in range(len(times)):
    t = times[i]
    name = 'w_%d_P_%d'% (v,t)
    idx = np.argwhere(samplesDiff[key][t,:]==2)
    tdata = np.delete(samplesDiff[key][t,:],idx)
    pred = [n.mean() for n in ppc[name]]
    
    ax[i].hist(pred, bins=20, alpha=0.5, density=True, color="r")
    ax[i].axvline(tdata.mean())
    ax[i].set_title('$w_%d^{%d}$' % (t,v))
    f.legend(['Mean of training data','Predicted distribution of Mean'],
        loc=10, frameon=True, framealpha=0.05, ncol=2, bbox_to_anchor=(0.4, 1.05))
plt.savefig('Prediction_w_%d_P.pdf'%v, dpi=300, bbox_inches='tight')