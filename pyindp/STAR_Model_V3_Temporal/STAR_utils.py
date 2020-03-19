import numpy as np
import pandas as pd
import seaborn as sns
from indp import *
import os
import sys
import string 
import pymc3 as pm
import random
import sklearn.metrics
import flow 
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
    
def prepare_data(samples,resCap,initial_net,keys):
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
    node_names = list(selected_nodes.keys())
    arc_names = list(selected_arcs.keys())
    selected_g = initial_net.G.subgraph(selected_ids)
    
    w_t={}
    w_t_1={}   
    w_n_t_1={}  # neighbor nodes
    w_d_t_1={}  # dependee nodes
    w_a_t_1={}  # connected arcs
    y_t={}
    y_t_1={}
    y_n_t_1={}  # connected nodes
    for key, val in selected_nodes.items():
        w_t[key]=selected_nodes[key][1:,:]
        w_t_1[key]=selected_nodes[key][:-1,:]  
    for key, val in selected_arcs.items():
        y_t[key]=selected_arcs[key][1:,:]
        y_t_1[key]=selected_arcs[key][:-1,:] 
        
    for key, val in selected_nodes.items():
        w_n_t_1[key] = np.zeros((T-1,noSamples))
        w_a_t_1[key] = np.zeros((T-1,noSamples))
        w_d_t_1[key] = np.zeros((T-1,noSamples))
        for u,v,a in selected_g.edges_iter(data=True):
            if 'w_'+`v` in keys or 'w_'+`u` in keys:
                if not a['data']['inf_data'].is_interdep: 
                    if 'w_'+`u`==key :
                        w_n_t_1[key]+=w_t_1['w_'+`v`]/2.0 # Because each arc happens twice
                        w_a_t_1[key]+=y_t_1['y_'+`v`+','+`u`]/2.0
                    if 'w_'+`v`==key:
                        w_n_t_1[key]+=w_t_1['w_'+`u`]/2.0
                        w_a_t_1[key]+=y_t_1['y_'+`v`+','+`u`]/2.0
                else:
                    if 'w_'+`v`==key:
                        w_d_t_1[key]+=1-w_t_1['w_'+`u`] # So that for non-dependent and those whose depndee nodes are functional, we get the same number
                        
    for u,v,a in selected_g.edges_iter(data=True):
        key = 'y_'+`u`+','+`v`
        if key in keys:
            if not a['data']['inf_data'].is_interdep:
                y_n_t_1[key] = np.zeros((T-1,noSamples))
                y_n_t_1[key]+=w_t_1['w_'+`v`]
                y_n_t_1[key]+=w_t_1['w_'+`u`]
                
    cols=['w_t','w_t_1','w_n_t_1','w_a_t_1','w_d_t_1','sample','time']
    node_data={}
    for key, val in selected_nodes.items():
        if key in keys:
            train_df = pd.DataFrame(columns=cols)
            for s in range(noSamples):
                for t in range(T-1):
                    # if w_t[key][t,s]==1 and w_t_1[key][t,s]==1:
                    #     pass
                    # else:
                    row = np.array([w_t[key][t,s],w_t_1[key][t,s],w_n_t_1[key][t,s],
                                    w_a_t_1[key][t,s],w_d_t_1[key][t,s],s,t+1])
                    temp = pd.Series(row,index=cols)
                    train_df=train_df.append(temp,ignore_index=True)
            node_data[key]=train_df 
    
    cols=['y_t','y_t_1','y_n_t_1','sample','time']
    arc_data={}
    for key, val in selected_arcs.items():
        if key in keys:
            train_df = pd.DataFrame(columns=cols)
            for s in range(noSamples):
                for t in range(T-1):
                    # if w_t[key][t,s]==1 and w_t_1[key][t,s]==1:
                    #     pass
                    # else:
                    row = np.array([y_t[key][t,s],y_t_1[key][t,s],y_n_t_1[key][t,s],
                                   s,t+1])
                    temp = pd.Series(row,index=cols)
                    train_df=train_df.append(temp,ignore_index=True)
            arc_data[key]=train_df 
            
    # data_all = pd.DataFrame(columns=cols)
    # # for key, val in data.items():   
    # #     data_all=pd.concat([data_all,val])
    # # data_all=data_all.reset_index(drop=True)   
    return node_data,arc_data

def train_model(train_data):
    logistic_model={}
    trace={}
    for key, val in train_data.items():
        if key[0]=='w':#[-2]==`2`:
            with pm.Model() as logistic_model[key]:
                pm.glm.GLM.from_formula('w_t ~ w_t_1 + w_n_t_1 + w_a_t_1 + w_d_t_1 + time',
                                        val,
                                        family=pm.glm.families.Binomial()) 
                trace[key] = pm.sample(1500, tune=750,chains=4, cores=4)
                estParam = pm.summary(trace[key]).round(4)
                print(estParam)     
                estParam.to_csv('Parameters\\model_parameters_%s.txt'%(key,),
                                header=True, index=True, sep=' ', mode='w')
                # pm.model_to_graphviz(logistic_model[key]).render(filename='Parameters\\model_graph_%s'%(key,))
        if key[0]=='y':#[-2]==`2`:
            with pm.Model() as logistic_model[key]:
                pm.glm.GLM.from_formula('y_t ~ y_t_1 + y_n_t_1 + time',
                                        val,
                                        family=pm.glm.families.Binomial()) 
                trace[key] = pm.sample(1500, tune=750,chains=4, cores=4)
                estParam = pm.summary(trace[key]).round(4)
                print(estParam)     
                estParam.to_csv('Parameters\\model_parameters_%s.txt'%(key,),
                                header=True, index=True, sep=' ', mode='w')
                # pm.model_to_graphviz(logistic_model[key]).render(filename='Parameters\\model_graph_%s'%(key,))

    return trace,logistic_model
    
def test_model(train_data,test_data,trace_all,model_all):
#    plt.rc('text', usetex=True)
#    plt.rcParams.update({'font.size': 14})
#    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    # plt.close('all')
    for key, val in train_data.items():
        trace = trace_all[key]
        model = model_all[key]
        ''' Traceplot '''
        # pm.traceplot(trace, combined=False) 
        
        ''' Generate and plot samples from the trained model '''
        ppc = pm.sample_posterior_predictive(trace, samples=1000, model=model)
             
        
        ''' Acceptance rate '''
        # f, ax = plt.subplots(2, 2)
        accept = trace.get_sampler_stats('mean_tree_accept', burn=0)
        # sns.distplot(accept, kde=False, ax=ax[0,0])
        # ax[0,0].set_title('Distribution of Tree Acceptance Rate')
        print 'Mean Acceptance rate: '+ `accept.mean()`
        # print 'Index of all diverging transitions: '+ `trace['diverging'].nonzero()`
        # print 'No of diverging transitions: '+ `len(trace['diverging'].nonzero()[0])`
    
        ''' Energy level '''
        # energy = trace['energy']
        # energy_diff = np.diff(energy)
        # sns.distplot(energy - energy.mean(), label='energy',ax=ax[0,1])
        # sns.distplot(energy_diff, label='energy diff',ax=ax[0,1])
        # ax[0,1].set_title('Distribution of energy level vs. change of energy between successive samples')
        # ax[0,1].legend()    
    
        
        ''' Prediction & data '''
        if key[0]=='w':
            x=np.array(train_data[key]['w_t'])
            y=ppc['y'].T.mean(axis=1)
            # ax[1,0].scatter(x,y,alpha=0.1)
            # ax[1,0].plot([0,1],[0,1],'r')
            # ax[1,0].set_title('Data vs. Prediction: training data ')    
            # ax[1,0].set_xlabel('data')
            # ax[1,0].set_ylabel('Mean Prediction')
            with open('Parameters\\R2.txt', mode='a') as f:
                print('\n'+key+' Training data R2: '+`sklearn.metrics.r2_score(x,y)`)
                f.write('\n'+key+' Training data R2: '+`sklearn.metrics.r2_score(x,y)`)
                f.close()
                
            # test_data=train_data[key].iloc[[random.randint(0, train_data[key].shape[0]/2) for p in range(0, train_data[key].shape[0]/5)],:]
            with pm.Model() as logistic_model_test:
                pm.glm.GLM.from_formula('w_t ~ w_t_1 + w_n_t_1+ w_a_t_1+w_d_t_1+time',
                                        test_data[key],
                                        family=pm.glm.families.Binomial())
                ppc_test = pm.sample_posterior_predictive(trace)
                x=np.array(test_data[key]['w_t'])
                y=ppc_test['y'].T.mean(axis=1)     
                # ax[1,1].scatter(x,y,alpha=0.2)
                # ax[1,1].plot([0,1],[0,1],'r')
                # ax[1,1].set_title('Data vs. Prediction: test data ')  
                # ax[1,1].set_xlabel('data')
                # ax[1,1].set_ylabel('Mean Prediction')
                with open('Parameters\\R2.txt', mode='a') as f:
                    print('\n'+key+' Test data R2: '+`sklearn.metrics.r2_score(x,y)`)
                    f.write('\n'+key+' Test data R2: '+`sklearn.metrics.r2_score(x,y)`)
                    f.close()
        # diff_pred_train_data = np.subtract(ppc['y'], np.array(train_data[key]['x_t']).T).T
        if key[0]=='y':
            x=np.array(train_data[key]['y_t'])
            y=ppc['y'].T.mean(axis=1)
            # ax[1,0].scatter(x,y,alpha=0.1)
            # ax[1,0].plot([0,1],[0,1],'r')
            # ax[1,0].set_title('Data vs. Prediction: training data ')    
            # ax[1,0].set_xlabel('data')
            # ax[1,0].set_ylabel('Mean Prediction')
            with open('Parameters\\R2.txt', mode='a') as f:
                print('\n'+key+' Training data R2: '+`sklearn.metrics.r2_score(x,y)`)
                f.write('\n'+key+' Training data R2: '+`sklearn.metrics.r2_score(x,y)`)
                f.close()
                
            # test_data=train_data[key].iloc[[random.randint(0, train_data[key].shape[0]/2) for p in range(0, train_data[key].shape[0]/5)],:]
            with pm.Model() as logistic_model_test:
                pm.glm.GLM.from_formula('y_t ~ y_t_1 + y_n_t_1+time',
                                        test_data[key],
                                        family=pm.glm.families.Binomial())
                ppc_test = pm.sample_posterior_predictive(trace)
                x=np.array(test_data[key]['y_t'])
                y=ppc_test['y'].T.mean(axis=1)     
                # ax[1,1].scatter(x,y,alpha=0.2)
                # ax[1,1].plot([0,1],[0,1],'r')
                # ax[1,1].set_title('Data vs. Prediction: test data ')  
                # ax[1,1].set_xlabel('data')
                # ax[1,1].set_ylabel('Mean Prediction')
                with open('Parameters\\R2.txt', mode='a') as f:
                    print('\n'+key+' Test data R2: '+`sklearn.metrics.r2_score(x,y)`)
                    f.write('\n'+key+' Test data R2: '+`sklearn.metrics.r2_score(x,y)`)
                    f.close()
        # diff_pred_train_data = np.subtract(ppc['y'], np.array(train_data[key]['x_t']).T).T        
        ''' Check the performance of predicted plans '''

            
    return ppc,ppc_test

def save_trace_and_data(train_data,test_data,trace,initial_net,samples,network_objects):
    import pickle
    
    pickle.dump([samples,network_objects,initial_net.G], open( "initial_data.pkl", "wb" ))
    pickle.dump([train_data,test_data], open( "train_test_data.pkl", "wb" ))
    
    # save traces
    fname={}
    for key,val in trace.items():
        fname[key]=pm.save_trace(trace[key],directory='./traces/'+key,overwrite=True)  

def compare_resotration(samples,network_objects,failSce_param,initial_net,layers,v_r,real_results_dir,no_prediction_samples=2):
    no_samples = samples[samples.keys()[0]].shape[1]
    no_time_steps = samples[samples.keys()[0]].shape[0]
    cols_results=['sample','time','resource_cap','pred_sample','result_type','cost',
                  'run_time','performance']
    result_df = pd.DataFrame(columns=cols_results)
    for res in v_r:
        pred_state={}
        for key,val in samples.iteritems():
            pred_state[key] = np.ones((no_time_steps,no_samples))
        for s in failSce_param['sample_range']:
            s_itr = s - failSce_param['sample_range'][0]
            for pred_s in range(no_prediction_samples):
                print '\nPrediction '+`pred_s+1`+': time step ',
                for t in range(no_time_steps):
                    '''compute total cost for the predictions'''
                    print `t`+'.',
                    decision_vars={0:{}} #0 becasue iINDP
                    for key,val in samples.iteritems():
                        if t==0:
                            pred_state[key][t,s_itr] = samples[key][0,s_itr] 
                            decision_vars[0][key]=pred_state[key][t,s_itr]
                        elif pred_state[key][t-1,s_itr]==1.0:
                            pred_state[key][t,s_itr] = 1.0
                            decision_vars[0][key]=1.0
                            if key[0]=='y':
                                arc_id = key[2:].split('),(') 
                                u = arc_id[0]+')'
                                v = '('+arc_id[1]
                                decision_vars[0]['y_'+v+','+u]=1.0
                        else:                        
                            if key[0]=='w':
                                w_t_1=pred_state[key][t-1,s_itr]  
                                w_n_t_1=0
                                w_a_t_1=0
                                w_d_t_1=0
                                for u,v,a in initial_net.G.edges_iter(data=True):
                                    if not a['data']['inf_data'].is_interdep: 
                                        if 'w_'+`u`==key :
                                            w_n_t_1+=pred_state['w_'+`v`][t-1,s_itr]/2.0 # Because each arc happens twice
                                            w_a_t_1+=pred_state['y_'+`v`+','+`u`][t-1,s_itr]/2.0
                                        if 'w_'+`v`==key:
                                            w_n_t_1+=pred_state['w_'+`u`][t-1,s_itr]/2.0
                                            w_a_t_1+=pred_state['y_'+`v`+','+`u`][t-1,s_itr]/2.0
                                    else:
                                        if 'w_'+`v`==key:
                                            w_d_t_1+=1-pred_state['w_'+`u`][t-1,s_itr] # So that for non-dependent and those whose depndee nodes are functional, we get the same number
                                cols=['w_t','w_t_1','w_n_t_1','w_a_t_1','w_d_t_1','time']
                                row = np.array([0,w_t_1,w_n_t_1,w_a_t_1,w_d_t_1,t])
                                with pm.Model() as logistic_model_test:
                                    pm.glm.GLM.from_formula('w_t ~ w_t_1 + w_n_t_1+ w_a_t_1+w_d_t_1+time',
                                                            pd.Series(row,index=cols),
                                                            family=pm.glm.families.Binomial())
                                    trace = pm.load_trace('./traces/'+key) 
                                    ppc_test = pm.sample_posterior_predictive(trace,
                                                                              samples=1,
                                                                              progressbar=False)
                                    pred_state[key][t,s_itr]=ppc_test['y']
                                decision_vars[0][key]=pred_state[key][t,s_itr] #0 becasue iINDP
                                    
                            if key[0]=='y': 
                                if key not in decision_vars[0].keys():
                                    y_n_t_1=0
                                    y_t_1=pred_state[key][t-1,s_itr] 
                                    
                                    u,v=arc_to_node(key)
                                    y_n_t_1+=pred_state['w_'+u][t-1,s_itr]
                                    y_n_t_1+=pred_state['w_'+v][t-1,s_itr]
                                    cols=['y_t','y_t_1','y_n_t_1','time']
                                    row = np.array([0,y_t_1,y_n_t_1,t+1])
                                    with pm.Model() as logistic_model_pred:
                                        pm.glm.GLM.from_formula('y_t ~ y_t_1 + y_n_t_1+time',
                                                                        pd.Series(row,index=cols),
                                                                        family=pm.glm.families.Binomial())
                                        trace = pm.load_trace('./traces/'+key) 
                                        ppc_test = pm.sample_posterior_predictive(trace,
                                                                                  samples=1,
                                                                                  progressbar=False)
                                        pred_state[key][t,s_itr]=ppc_test['y']
                                     #0 becasue iINDP
                                    decision_vars[0][key]=pred_state[key][t,s_itr]
                                    decision_vars[0]['y_'+v+','+u]=pred_state[key][t,s_itr] #0 becasue iINDP
                        
                    flow_results=flow.flow_problem(network_objects[0,s],v_r=0,
                                    layers=layers,controlled_layers=layers,
                                  decision_vars=decision_vars,print_cmd=True, time_limit=None)
                    row = np.array([s,t,res,pred_s,'predicted',
                                    flow_results[1][0]['costs']['Total'],
                                    flow_results[1][0]['run_time'],
                                    flow_results[1][0]['costs']['Under Supply Perc']])
                    temp = pd.Series(row,index=cols_results)
                    result_df=result_df.append(temp,ignore_index=True)     
                    if pred_s==0:
                        ''' read cost from the actual data computed by INDP '''
                        folder_name = real_results_dir+'results/indp_results_L'+`len(layers)`+'_m0_v'+`res`
                        real_results=indputils.INDPResults()
                        real_results=real_results.from_csv(folder_name,s,suffix="")
                        row = np.array([s,t,res,-1,'data',
                                        real_results[t]['costs']['Total'],
                                        real_results[t]['run_time'],
                                        real_results[t]['costs']['Under Supply Perc']])
                        temp = pd.Series(row,index=cols_results)
                        result_df=result_df.append(temp,ignore_index=True)
                    ### Write models to file  
                    # indp.save_INDP_model_to_file(flow_results[0],'./models',t,l=0)   
    return result_df

def train_test_split(node_data,arc_data,keys):
    train_data={}
    test_data={}
    data=0
    duplicates=[]
    for key in keys:
        duplicate=False
        if key[0]=='w':
            data=node_data[key]
        elif key[0]=='y':
            data=arc_data[key]
            duplicates.append(key)
            u,v=STAR_utils.arc_to_node(key)
            if 'y_'+v+','+u in duplicates:
                duplicate=True
        else:
            sys.exit('Wrong variable name: ' + key)
    
        if not duplicate:
            msk = np.random.rand(len(data)) < 0.8
            test_data[key]=data[~msk]
            train_data[key]=data[msk]
    return train_data,test_data

def logistic(l):
    return 1 / (1 + pm.math.exp(-l))

def arc_to_node(arc_key):
    arc_id = arc_key[2:].split('),(') 
    u = arc_id[0]+')'
    v = '('+arc_id[1] 
    return u,v