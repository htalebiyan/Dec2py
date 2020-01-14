import numpy as np
import pandas as pd
from indp import *
import os
import sys
import string 
import pymc3 as pm
print('Running with PyMC3 version v.{}'.format(pm.__version__))
import copy
import matplotlib.pyplot as plt

def importData(params,failSce_param,layers):  
    # Set root directories
    if failSce_param['type']=='Andres':
        base_dir = failSce_param['Base_dir']+"/data/INDP_7-20-2015/"
        ext_interdependency = "../data/INDP_4-12-2016"
        topology = None
        shelby_data = True
    elif failSce_param['type']=='WU':
        if failSce_param['filtered_List']!=None:
            listHD = pd.read_csv(failSce_param['filtered_List'])
        base_dir = failSce_param['Base_dir']+"data/Extended_Shelby_County/"
        damage_dir = failSce_param['Base_dir']+"/data/Wu_Damage_scenarios/"
        ext_interdependency = None
        topology = None
        shelby_data = True
    elif failSce_param['type']=='synthetic':  
        base_dir = failSce_param['Base_dir']
        ext_interdependency = None
        shelby_data = False  
        topology = failSce_param['topology']
    
    samples={}
    network_objects={}
    initial_net = 0
    for m in failSce_param['mags']:    
        for i in failSce_param['sample_range']:
            if failSce_param['filtered_List']==None or len(listHD.loc[(listHD.set == i) & (listHD.sce == m)].index):
#                print '\n---Running Magnitude '+`m`+' sample '+`i`+'...'
            
                print("Initializing network...")
                if not shelby_data:  
                    InterdepNet,noResource,layers=initialize_network(BASE_DIR=base_dir,external_interdependency_dir=ext_interdependency,magnitude=m,sample=i,shelby_data=shelby_data,topology=topology) 
                    params["V"]=noResource
                else:  
                    InterdepNet,_,_=initialize_network(BASE_DIR=base_dir,external_interdependency_dir=ext_interdependency,sim_number=0,magnitude=6,sample=0,v=params["V"],shelby_data=shelby_data)                    
                params["N"]=InterdepNet
                params["SIM_NUMBER"]=i
                params["MAGNITUDE"]=m
                if not initial_net:
                    initial_net = InterdepNet
                
                if failSce_param['type']=='WU':
                    add_Wu_failure_scenario(InterdepNet,BASE_DIR=damage_dir,noSet=i,noSce=m)
                elif failSce_param['type']=='ANDRES':
                    add_failure_scenario(InterdepNet,BASE_DIR=base_dir,magnitude=m,v=params["V"],sim_number=i)
                elif failSce_param['type']=='synthetic':
                    add_synthetic_failure_scenario(InterdepNet,BASE_DIR=base_dir,topology=topology,config=m,sample=i)
                
                
                samples = initialize_matrix(InterdepNet,samples,m,i,10)
                results_dir=params['OUTPUT_DIR']+'_L'+`len(layers)`+'_m'+`m`+'_v'+`params['V']`
                samples = read_restoration_plans(samples, m, i,results_dir,suffix='')
                network_objects[m,i]=InterdepNet
    print('Data Imported')
    return samples,network_objects,initial_net,noResource,layers    


def initialize_matrix(N, sample, m, i, time_steps):
    for v in N.G.nodes():
        if v not in sample.keys():
            sample[v]= np.ones((time_steps+1,1))
        else:
            sample[v] = np.append(sample[v], np.ones((time_steps+1,1)), axis=1)
            
        if N.G.node[v]['data']['inf_data'].functionality==0.0:
            sample[v][:,-1] = 0.0
    return sample

def read_restoration_plans(sample, m, i, results_dir,suffix=''):
    action_file=results_dir+"/actions_"+`i`+"_"+suffix+".csv" 
    if os.path.isfile(action_file):
        with open(action_file) as f:
            lines=f.readlines()[1:]
            for line in lines:
                data=string.split(str.strip(line),",")
                t=int(data[0])
                action=str.strip(data[1])
                k = int(action[-1])
                if '/' in action:
                    pass
                else:
                    act_data=string.split(str.strip(action),".")
                    node_id=int(act_data[0])
                    sample[(node_id,k)][t:,-1] = 1.0
    return sample

def collect_feature_data(samples,network_objects):
    return True
    
def train_model(samples):
    samplesDiff = copy.deepcopy(samples)
    noSets = 50
    noScenarios = 96
    resCap = 3
    T = 10     
    
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
        if True:#i[0] == 'w':
            samplesNodes[i] = samples[i]
    noSpa = len(samplesNodes.keys())
    names = list(samplesNodes.keys())
    noTmp = T+1
    
    p = {}
    sdy = {}
    y_t = {}
#    trace=[]
#    model=[]
#    conv = pd.read_csv('filteredVarsFromCrudeModel.txt', delimiter = ' ')
    for i in range(noSpa):
        estParamAll = pd.DataFrame()
        key = names[i]
        if True:#key[-1]=='P':
            y_t[i,0] = samplesDiff[key][0,:]
#            ind = conv.index[(conv['name'] == key)].tolist()
            noTimeSteps =10# len(ind)
            if noTimeSteps != 0:
                with pm.Model() as model:
                    p[i,0] = pm.Beta('p(%s)_%d'%(key,0), alpha=2, beta=2)
                    sdy[i] = pm.Normal('sdy_%s'%(key,), mu=0, sd=1)
    
#                    for j in ind: 
#                        t = conv['time'][j]
#                        index = np.argwhere(samplesDiff[key][t,:]==2)
#                        trainData = np.delete(samplesDiff[key][t,:],index)
                    for t in range(1,noTimeSteps+1):
                        index = np.argwhere(samplesDiff[key][t,:]==2)
                        trainData = np.delete(samplesDiff[key][t,:],index)
                        p[i,t] = pm.Deterministic('p(%s)_%d'%(key,t),
                                             logistic(p[i,t-1]+sdy[i]))
                        y_t[i,t] = pm.Bernoulli('%s_%d'%(key,t),
                                                   p = p[i,t],
                                                   observed=trainData)
                    trace = pm.sample(1500, tune=500, cores=4, njobs=4)
                    estParam = pm.summary(trace).round(4)
                    print(estParam)
#                    estParam['Name'] = key 
                   
                    estParam.to_csv('Parameters\\model_parameters_%s.txt' % (names[i],),
                                    header=True, index=True, sep=' ', mode='w')
    
          
#    pm.traceplot(trace, combined=True) # ,varnames=['p_w_0_W_6']
#    pm.model_to_graphviz(model).render()
#    pp(y_t[1,1])
#    
#    ''' Generate and plot samples from the trained model '''
#    ppc = pm.sample_ppc(trace, samples=1000, model=model)
#    
#    
##    plt.rc('text', usetex=True)
##    plt.rcParams.update({'font.size': 14})
##    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#    times = [1,3,5]
#    v = 83
#    f, ax = plt.subplots(1, len(times), sharex=False, sharey=True,figsize=(12,4))
#    f.text(0.5, 0.01, 'State Transition Value', ha='center', va='center')
#    f.text(0.1, 0.5, 'Density', ha='center', va='center', rotation='vertical')
#    #f.tight_layout()
#    
#    for i in range(len(times)):
#        t = times[i]
#        name = 'w_%d_P_%d'% (v,t)
#        idx = np.argwhere(samplesDiff[key][t,:]==2)
#        tdata = np.delete(samplesDiff[key][t,:],idx)
#        pred = [n.mean() for n in ppc[name]]
#        
#        ax[i].hist(pred, bins=20, alpha=0.5, density=True, color="r")
#        ax[i].axvline(tdata.mean())
#        ax[i].set_title('$w_%d^{%d}$' % (t,v))
#        f.legend(['Mean of training data','Predicted distribution of Mean'],
#            loc=10, frameon=True, framealpha=0.05, ncol=2, bbox_to_anchor=(0.4, 1.05))
#    plt.savefig('Prediction_w_%d_P.pdf'%v, dpi=300, bbox_inches='tight')
        
def logistic(l):
    return 1 / (1 + pm.math.exp(-l))