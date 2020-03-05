import numpy as np
import pandas as pd
import seaborn as sns
from indp import *
import os
import sys
import string 
import pymc3 as pm
print('Running with PyMC3 version v.{}'.format(pm.__version__))
import matplotlib.pyplot as plt

def importData(params,failSce_param,layers):  
    # Set root directories
    base_dir = failSce_param['Base_dir']
    damage_dir = failSce_param['Damage_dir'] 
    topology = None
    shelby_data = True
    ext_interdependency = None
    if failSce_param['type']=='Andres':
        ext_interdependency = "../data/INDP_4-12-2016"
    elif failSce_param['type']=='WU':
        if failSce_param['filtered_List']!=None:
            listHD = pd.read_csv(failSce_param['filtered_List'])
    elif failSce_param['type']=='random':
        pass
    elif failSce_param['type']=='synthetic':  
        shelby_data = False  
        topology = failSce_param['topology']
    
    samples={}
    network_objects={}
    initial_net = 0
    for m in failSce_param['mags']:    
        for i in failSce_param['sample_range']:
            if failSce_param['filtered_List']==None or len(listHD.loc[(listHD.set == i) & (listHD.sce == m)].index):
#                print '\n---Running Magnitude '+`m`+' sample '+`i`+'...'
            
                # print("Initializing network...")
                if i%50==0:
                    print ".", 
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
                    add_Wu_failure_scenario(InterdepNet,DAM_DIR=damage_dir,noSet=i,noSce=m)
                elif failSce_param['type']=='ANDRES':
                    add_failure_scenario(InterdepNet,DAM_DIR=damage_dir,magnitude=m,v=params["V"],sim_number=i)
                elif failSce_param['type']=='random':
                    add_random_failure_scenario(InterdepNet,DAM_DIR=damage_dir,sample=i) 
                elif failSce_param['type']=='synthetic':
                    add_synthetic_failure_scenario(InterdepNet,DAM_DIR=damage_dir,topology=topology,config=m,sample=i)
                
                samples = initialize_matrix(InterdepNet,samples,m,i,10)
                results_dir=params['OUTPUT_DIR']+'_L'+`len(layers)`+'_m'+`m`+'_v'+`params['V']`
                samples = read_restoration_plans(samples, m, i,results_dir,suffix='')
                network_objects[m,i]=InterdepNet
    print('Data Imported')
    return samples,network_objects,initial_net,params["V"],layers    

def initialize_matrix(N, sample, m, i, time_steps):
    for v in N.G.nodes():
        name = 'w_'+`v`
        if name not in sample.keys():
            sample[name]= np.ones((time_steps+1,1))
        else:
            sample[name] = np.append(sample[name], np.ones((time_steps+1,1)), axis=1)
        if N.G.node[v]['data']['inf_data'].functionality==0.0:
            sample[name][:,-1] = 0.0
           
    for u,v,a in N.G.edges(data=True):
        if not a['data']['inf_data'].is_interdep:  
            name = 'y_'+`u`+','+`v`
            if name not in sample.keys():
                sample[name]= np.ones((time_steps+1,1))
            else:
                sample[name] = np.append(sample[name], np.ones((time_steps+1,1)), axis=1)
            if N.G[u][v]['data']['inf_data'].functionality==0.0:
                sample[name][:,-1] = 0.0    
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
                    act_data=string.split(str.strip(action),"/")
                    u = string.split(str.strip(act_data[0]),".")
                    v = string.split(str.strip(act_data[1]),".")
                    if u[1]!=v[1]:
                        sys.exit('Interdepndency '+act_data+' is counted as an arc')
                    arc_id = `(int(u[0]),k)`+','+`(int(v[0]),k)`
                    sample['y_'+arc_id][t:,-1] = 1.0
                else:
                    act_data=string.split(str.strip(action),".")
                    node_id=int(act_data[0])
                    sample['w_'+`(node_id,k)`][t:,-1] = 1.0
    else:
        sys.exit('No results dir: '+action_file)
    return sample

def collect_feature_data(samples,network_objects):
    return True
    
def train_data(samples,resCap,initial_net):
    names = samples.keys()
    noSamples = int(samples[names[0]].shape[1])
    T = int(samples[names[0]].shape[0])                
              
    selected_nodes = {}
    selected_ids = []
    selected_arcs = {}
    for key, val in samples.items():
        if key[0]=='w':
            selected_nodes[key] = val
            node_id = key[2:].strip(' )(').split(',') 
            selected_ids.append((int(node_id[0]),int(node_id[1])))
        elif key[0]=='y':
            selected_arcs[key] = val
    noSpa = len(selected_nodes.keys())+len(selected_arcs.keys())
    node_names = list(selected_nodes.keys())
    arc_names = list(selected_arcs.keys())
    selected_g = initial_net.G.subgraph(selected_ids)
    # A = nx.adjacency_matrix(initial_net.G.subgraph(node_names)).todense()
    
    w_t={}
    w_t_1={}   
    w_n_t_1={}  #neighbor nodes
    w_a_t_1={}  # connected arcs
    y_t={}
    y_t_1={}
    for key, val in selected_nodes.items():
        w_t[key]=selected_nodes[key][1:,:]
        w_t_1[key]=selected_nodes[key][:-1,:]  
    for key, val in selected_arcs.items():
        y_t[key]=selected_arcs[key][1:,:]
        y_t_1[key]=selected_arcs[key][:-1,:]  
    for key, val in selected_nodes.items():
        w_n_t_1[key] = np.zeros((T-1,noSamples))
        w_a_t_1[key] = np.zeros((T-1,noSamples))
        for u,v,a in selected_g.edges_iter(data=True):
            if not a['data']['inf_data'].is_interdep: 
                if 'w_'+`u`==key :
                    w_n_t_1[key]+=w_t_1['w_'+`v`]/2.0
                    w_a_t_1[key]+=y_t_1['y_'+`v`+','+`u`]/2.0
                if 'w_'+`v`==key:
                    w_n_t_1[key]+=w_t_1['w_'+`u`]/2.0
                    w_a_t_1[key]+=y_t_1['y_'+`v`+','+`u`]/2.0
    cols=['w_t','w_t_1','w_n_t_1','w_a_t_1','sample','time']
    train_data={}
    for key, val in selected_nodes.items():
        train_df = pd.DataFrame(columns=cols)
        for s in range(noSamples):
            for t in range(T-1):
                if w_t[key][t,s]==1 and w_t_1[key][t,s]==1:
                    pass
                else:
                    row = np.array([w_t[key][t,s],w_t_1[key][t,s],w_n_t_1[key][t,s],
                                    w_a_t_1[key][t,s],s,t])
                    temp = pd.Series(row,index=cols)
                    train_df=train_df.append(temp,ignore_index=True)
        train_data[key]=train_df 
    
    train_data_all = pd.DataFrame(columns=cols)
    for key, val in train_data.items():   
        train_data_all=pd.concat([train_data_all,val])
    train_data_all=train_data_all.reset_index(drop=True)   
    return train_data_all,train_data

def train_model(train_data_all,train_data):
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula('w_t ~ w_n_t_1+w_a_t_1',
                                train_data_all,
                                family=pm.glm.families.Binomial()) #x_t_1 + 
        trace = pm.sample(500, tune=1000,cores=1)
        estParam = pm.summary(trace).round(4)
        print(estParam)     
        estParam.to_csv('Parameters\\model_parameters.txt',
                        header=True, index=True, sep=' ', mode='w')
        pm.model_to_graphviz(logistic_model).render()
    return trace,logistic_model
    
def test_model(train_data_all,trace,model):
#    plt.rc('text', usetex=True)
#    plt.rcParams.update({'font.size': 14})
#    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    ''' Traceplot '''
    pm.traceplot(trace, combined=False) # ,varnames=['p_w_0_W_6']
    
    ''' Generate and plot samples from the trained model '''
    ppc = pm.sample_posterior_predictive(trace, samples=1000, model=model)
         
    
    ''' Acceptance rate '''
    f, ax = plt.subplots(1, 2)
    accept = trace.get_sampler_stats('mean_tree_accept', burn=0)
    sns.distplot(accept, kde=False, ax=ax[0])
    ax[0].set_title('Distribution of Tree Acceptance Rate')
    print 'Mean Acceptance rate: '+ `accept.mean()`
    print '\nIndex of all diverging transitions: '+ `trace['diverging'].nonzero()`

    ''' Energy level '''
    energy = trace['energy']
    energy_diff = np.diff(energy)
    sns.distplot(energy - energy.mean(), label='energy',ax=ax[1])
    sns.distplot(energy_diff, label='energy diff',ax=ax[1])
    ax[1].set_title('Distribution of energy level vs. change of energy between successive samples')
    ax[1].legend()    

    
    ''' Prediction & data '''
    f, ax = plt.subplots(1, 2)
    x=np.array(train_data_all['w_t'])
    y=ppc['y'].T.mean(axis=1)
    ax[0].scatter(x,y,alpha=0.01)
    ax[0].plot([0,1],[0,1],'r')
    ax[0].set_title('Data vs. Prediction: training data ')    

    test_data=train_data_all.iloc[100:500,:]
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula('w_t ~ w_n_t_1+ w_a_t_1',
                                test_data,
                                family=pm.glm.families.Binomial()) #x_t_1 + 3
        ppc_test = pm.sample_posterior_predictive(trace)
        x=np.array(test_data['w_t'])
        y=ppc_test['y'].T.mean(axis=1)     
        ax[1].scatter(x,y,alpha=0.1)
        ax[1].plot([0,1],[0,1],'r')
        ax[1].set_title('Data vs. Prediction: test data ')   
    # diff_pred_train_data = np.subtract(ppc['y'], np.array(train_data_all['x_t']).T).T
        
    ''' Check the performance of predicted plans '''

            
    return ppc
def logistic(l):
    return 1 / (1 + pm.math.exp(-l))