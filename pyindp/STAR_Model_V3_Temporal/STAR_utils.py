import numpy as np
import pandas as pd
from indp import *
import os
import sys
import string 
import pymc3 as pm
print('Running with PyMC3 version v.{}'.format(pm.__version__))
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
    return samples,network_objects,initial_net,params["V"],layers    

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
    else:
        sys.exit('No results dir: '+action_file)
    return sample

def collect_feature_data(samples,network_objects):
    return True
    
def train_model(samples,resCap,initial_net):
    samplesDiff = {}
    node_names = samples.keys()
    noSamples = int(samples[node_names[0]].shape[1])
    T = int(samples[node_names[0]].shape[0])     
    
    for key, val in samples.items(): 
        if key not in samplesDiff.keys():
            samplesDiff[key]=np.zeros((T-1,noSamples))
        for s in range(noSamples):
            for t in range(1,T):
                if val[t,s]==1:
                    if val[t-1,s] == 0:
                        samplesDiff[key][t-1,s] = 1
                    elif val[t-1,s] == 1:
                        samplesDiff[key][t-1,s] = 2 #!!!Assumptiom
                    else:
                        print('AYFKM????')
                elif  val[t,s]==0 and val[t-1,s]==0: 
                    samplesDiff[key][t-1,s] = 0 
                else:
                    print('AYFKM????')
              
    selected_nodes = {}
    for key, val in samplesDiff.items():
        if key[1]==2:
            selected_nodes[key] = val
    noSpa = len(selected_nodes.keys())
    node_names = list(selected_nodes.keys())
    noTmp = T+1
    A = nx.adjacency_matrix(initial_net.G.subgraph(node_names)).todense()
    
    p = {}
    a = {}
    sdy = {}
    y = {}
    trainData={}
#    trace=[]
#    model=[]
#    conv = pd.read_csv('filteredVarsFromCrudeModel.txt', delimiter = ' ')
    with pm.Model() as model:
        priors = {}
        for i in range(noSpa):
            # estParamAll = pd.DataFrame()
            key = node_names[i]
            priors['p_%s'%(key,)] = pm.Beta('p_%s'%(key,), alpha=2, beta=2)
            priors['sdy_%s'%(key,)] = pm.Normal('sdy_%s'%(key,), mu=0, sd=1)
            trainData[key] = samplesDiff[key].transpose().reshape(-1)

        for i in range(noSpa):
            key = node_names[i]
            formula = priors['p_%s'%(key,)]+priors['sdy_%s'%(key,)]
            for j in range(noSpa):
                if A[i,j]:
                    key_neighbor = node_names[j]
                    formula +=  priors['p_%s'%(key_neighbor,)]
            p[key] = pm.Deterministic('p%s'%(key,),logistic(formula))
            y[key] = pm.Bernoulli('y_%s'%(key,), p = p[key],
                                       observed=trainData[key])
        trace = pm.sample(1500, tune=500, cores=4, njobs=4)
        estParam = pm.summary(trace).round(4)
        print(estParam)     
        estParam.to_csv('Parameters\\model_parameters.txt',
                        header=True, index=True, sep=' ', mode='w')
    
          
    # pm.traceplot(trace, combined=True) # ,varnames=['p_w_0_W_6']
    pm.model_to_graphviz(model).render()
    
    ''' Generate and plot samples from the trained model '''
    ppc = pm.sample_posterior_predictive(trace, samples=1000, model=model)
    
    return samplesDiff, trainData, ppc
    
def test_model(samples, samplesDiff,trainData, ppc):
#    plt.rc('text', usetex=True)
#    plt.rcParams.update({'font.size': 14})
#    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    ''' Prediction & mean data '''
    nodes = [1,10,12]
    f, ax = plt.subplots(1, len(nodes), sharex=False, sharey=True,figsize=(12,4))
    f.text(0.5, 0.01, 'State Transition Value', ha='center', va='center')
    f.text(0.1, 0.5, 'Density', ha='center', va='center', rotation='vertical')
    #f.tight_layout()
    for i in range(len(nodes)):
        key= (nodes[i],2)
        name = 'y_%s'% (key,)
        data = trainData[key]
        pred = [n.mean() for n in ppc['y_%s'% (key,)]]
        
        ax[i].hist(pred, bins=20, alpha=0.5, density=True, color="r")
        ax[i].axvline(data.mean())
        # print data.mean()
        ax[i].set_title('$y_%s$' % (key,))
        f.legend(['Mean of training data','Predicted distribution of Mean'],
            loc=10, frameon=True, framealpha=0.05, ncol=2, bbox_to_anchor=(0.4, 1.05))
    # plt.savefig('Prediction_w_%d_P.pdf'%v, dpi=300, bbox_inches='tight')  
    ''' Prediction error '''
def logistic(l):
    return 1 / (1 + pm.math.exp(-l))