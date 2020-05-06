import numpy as np
import pandas as pd
from indp import *
import os
import sys
import string 
import pymc3 as pm
import sklearn.metrics
import flow 
import time
import _pickle as pickle
from infrastructure import *
from operator import itemgetter 
import seaborn as sns
import copy
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
print('Running with PyMC3 version v.{}'.format(pm.__version__))

def importData(params,failSce_param,layers,suffix=''):  
    print('\nNumber of resources: '+str(params["V"]))
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
    costs = {}
    initial_net = {}
    
    t_suf = time.strftime("%Y%m%d")
    folder_name = 'data'+t_suf
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  
        
    for m in failSce_param['mags']:    
        for i in failSce_param['sample_range']:
            results_dir=params['OUTPUT_DIR']+'_L'+str(len(layers))+'_m'+str(m)+'_v'+str(params['V'])
            action_file=results_dir+"/actions_"+str(i)+"_"+suffix+".csv" 
            cost_file=results_dir+"/costs_"+str(i)+"_"+suffix+".csv" 
            if os.path.isfile(action_file) and os.path.isfile(cost_file):
                if failSce_param['filtered_List']==None or len(listHD.loc[(listHD.set == i) & (listHD.sce == m)].index):
    #                print '\n---Running Magnitude '+str(m)+' sample '+str(i)+'...'
                
                    # print("Initializing network...")
                    if (i-failSce_param['sample_range'][0]+1)%2==0:
                        update_progress(i-failSce_param['sample_range'][0]+1,len(failSce_param['sample_range']))
    
                    if not shelby_data:  
                        InterdepNet,noResource,layers=initialize_network(BASE_DIR=base_dir,external_interdependency_dir=ext_interdependency,magnitude=m,sample=i,shelby_data=shelby_data,topology=topology) 
                        params["V"]=noResource
                    else:  
                        InterdepNet,_,_=initialize_network(BASE_DIR=base_dir,external_interdependency_dir=ext_interdependency,sim_number=0,magnitude=6,sample=0,v=params["V"],shelby_data=shelby_data)                    
                    params["N"]=InterdepNet
                    params["SIM_NUMBER"]=i
                    params["MAGNITUDE"]=m
                    if not initial_net.keys():
                        initial_net[0] = copy.deepcopy(InterdepNet)
                    
                    if failSce_param['type']=='WU':
                        add_Wu_failure_scenario(InterdepNet,DAM_DIR=damage_dir,noSet=i,noSce=m)
                    elif failSce_param['type']=='ANDRES':
                        add_failure_scenario(InterdepNet,DAM_DIR=damage_dir,magnitude=m,v=params["V"],sim_number=i)
                    elif failSce_param['type']=='random':
                        add_random_failure_scenario(InterdepNet,DAM_DIR=damage_dir,sample=i) 
                    elif failSce_param['type']=='synthetic':
                        add_synthetic_failure_scenario(InterdepNet,DAM_DIR=damage_dir,topology=topology,config=m,sample=i)
                    
                    samples = initialize_matrix(InterdepNet,samples,m,i,10)
                    samples,costs = read_restoration_plans(samples, costs, InterdepNet, m, i,results_dir,suffix='')
            else:
                with open(folder_name+'/missing_scenarios.txt', 'a') as filehandle:
                    filehandle.write(str(params["V"])+'\t'+str(m)+'\t'+str(i)+'\n')   
                    filehandle.close()                  
    update_progress(i-failSce_param['sample_range'][0]+1,len(failSce_param['sample_range']))
    return samples,costs,initial_net,params["V"],layers    

def initialize_matrix(N, sample, m, i, time_steps):
    for v in N.G.nodes():
        name = 'w_'+str(v)
        if name not in sample.keys():
            sample[name]= np.ones((time_steps+1,1))
        else:
            sample[name] = np.append(sample[name], np.ones((time_steps+1,1)), axis=1)
        if N.G.node[v]['data']['inf_data'].functionality==0.0:
            sample[name][:,-1] = 0.0
           
    for u,v,a in N.G.edges(data=True):
        if not a['data']['inf_data'].is_interdep:  
            name = 'y_'+str(u)+','+str(v)
            if name not in sample.keys():
                sample[name]= np.ones((time_steps+1,1))
            else:
                sample[name] = np.append(sample[name], np.ones((time_steps+1,1)), axis=1)
            if N.G[u][v]['data']['inf_data'].functionality==0.0:
                sample[name][:,-1] = 0.0
    return sample

def read_restoration_plans(sample,costs,InterdepNet,m,i,results_dir,suffix=''):
    # Read elemnts' states
    action_file=results_dir+"/actions_"+str(i)+"_"+suffix+".csv" 
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
                    arc_id = str((int(u[0]),k))+','+str((int(v[0]),k))
                    sample['y_'+arc_id][t:,-1] = 1.0
                else:
                    act_data=string.split(str.strip(action),".")
                    node_id=int(act_data[0])
                    sample['w_'+str((node_id,k))][t:,-1] = 1.0
    else:
        sys.exit('No results dir: '+action_file)
        
    # Read cost for the entire interdepndent network   
    cost_file=results_dir+"/costs_"+str(i)+"_"+suffix+".csv" 
    T=sample[sample.keys()[0]].shape[0]
    if os.path.isfile(cost_file):
        with open(cost_file) as f:
            lines=f.readlines()
            for line in lines:
                if line[0]=='t':
                    headers=string.split(str.strip(line),",")
                    headers = [c.replace(' ', '_') for c in headers]
                    for h in headers[1:]:
                        if h not in costs.keys():
                            costs[h]= np.zeros((T,1))
                        else:
                            costs[h] = np.append(costs[h], np.zeros((T,1)), axis=1)
                    continue
                data=string.split(str.strip(line),",")
                t=int(data[0])
                for ii in data[1:]:
                    cost_name = headers[data.index(ii)]
                    costs[cost_name][t,-1]=float(ii)
    else:
        sys.exit('No results dir: '+cost_file)
                
    return sample,costs

def prepare_data(samples,costs,initial_net,res,keys):
    print('Preparing data:')
    names = samples.keys()
    noSamples = int(samples[names[0]].shape[1])
    T = int(samples[names[0]].shape[0])                
      
    w_n_t_1,w_d_t_1,w_a_t_1,w_h_t_1,w_c_t_1,y_n_t_1,y_c_t_1=extract_features(samples,initial_net,keys)
    
    print('\nBuilding Dataframes:') 
    cost_names=costs.keys()
    costs_normed={}
    for i in cost_names: 
        costs_normed[i] = normalize_costs(costs[i],i,costs['Total'][0,:])
        
    node_cols=['w_t','w_t_1','sample','time','Rc','w_n_t_1','w_a_t_1','w_d_t_1','w_h_t_1','w_c_t_1','y_c_t_1']
    arc_cols=['y_t','y_t_1','sample','time','Rc','y_n_t_1','w_c_t_1','y_c_t_1']
    node_data={} 
    arc_data={}
    for key in keys:
        if keys.index(key)%10==0:
            update_progress(keys.index(key),len(keys))
        
        cols=node_cols+cost_names
        if key[0]=='y':
            cols=arc_cols+cost_names
        train_df = pd.DataFrame(columns=cols)
        for s in range(noSamples):
            for t in range(T-1):
                if samples[key][t+1,s]==1 and samples[key][t,s]>0:
                    pass
                else:
                    norm_cost_values=[costs_normed[x][t,s] for x in cost_names]
                    l = int(key[-2])
                    basic_features=[samples[key][t+1,s],samples[key][t,s],s,(t+1)/float(T-1),res/100.0]
                    if key[0]=='w':
                        special_feature=[w_n_t_1[key][t,s],w_a_t_1[key][t,s],
                                        w_d_t_1[key][t,s],w_h_t_1[l][t,s],
                                        w_c_t_1[l][t,s],y_c_t_1[l][t,s]]
                    elif key[0]=='y':
                        special_feature=[y_n_t_1[key][t,s],w_c_t_1[l][t,s],y_c_t_1[l][t,s]]
                    row = np.array(basic_features+special_feature+norm_cost_values)
                    temp = pd.Series(row,index=cols)
                    train_df=train_df.append(temp,ignore_index=True)
        if key[0]=='w':
            node_data[key]=train_df
        elif key[0]=='y':
            arc_data[key]=train_df 
            
    update_progress(len(keys),len(keys))  
    return node_data,arc_data

def extract_features(samples,initial_net,keys,prog_bar=True):
    keys = samples.keys()
    noSamples = int(samples[keys[0]].shape[1])
    T = int(samples[keys[0]].shape[0]) 

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
        else:
            sys.exit()
    node_keys = list(selected_nodes.keys())
    arc_keys = list(selected_arcs.keys())
    selected_g = initial_net[0].G.subgraph(selected_ids)
    
    w_t={}
    w_t_1={}   
    w_n_t_1={}  # neighbor nodes
    w_d_t_1={}  # dependee nodes
    w_a_t_1={}  # connected arcs
    w_h_t_1={}  # highest demand nodes
    w_c_t_1={}  # all nodes
    y_c_t_1={}  # all arcs
    no_w_n={}  
    no_w_d={}   
    no_w_c={} 
    no_y_c={}
    y_t={}
    y_t_1={}
    y_n_t_1={}  # connected nodes
    for key, val in selected_nodes.items():
        w_t[key]=selected_nodes[key][1:,:]
        w_t_1[key]=selected_nodes[key][:-1,:]   
    for key, val in selected_arcs.items():
        y_t[key]=selected_arcs[key][1:,:]
        y_t_1[key]=selected_arcs[key][:-1,:]
        
    for key in node_keys:           
        w_n_t_1[key] = np.zeros((T-1,noSamples))
        w_a_t_1[key] = np.zeros((T-1,noSamples))
        w_d_t_1[key] = np.zeros((T-1,noSamples))
        no_w_n[key]=0
        no_w_d[key]=0
    for u,v,a in selected_g.edges_iter(data=True):
        if 'w_'+str(v) in keys or 'w_'+str(u) in keys:
            if not a['data']['inf_data'].is_interdep: 
                key = 'w_'+str(u)
                if key in keys:
                    w_n_t_1[key]+=w_t_1['w_'+str(v)]/2.0 # Because each arc happens twice
                    w_a_t_1[key]+=y_t_1['y_'+str(v)+','+str(u)]/2.0
                    no_w_n[key]+=0.5
                key = 'w_'+str(v)
                if key in keys:
                    w_n_t_1[key]+=w_t_1['w_'+str(u)]/2.0
                    w_a_t_1[key]+=y_t_1['y_'+str(v)+','+str(u)]/2.0
                    no_w_n[key]+=0.5
            else:
                if 'w_'+str(v)in keys:
                    w_d_t_1['w_'+str(v)]+=1-w_t_1['w_'+str(u)] # o that for non-dependent and those whose depndee nodes are functional, we get the same number
                    no_w_d['w_'+str(v)]+=1    
    for key in node_keys:
        w_n_t_1[key]/=no_w_n[key]
        w_a_t_1[key]/=no_w_n[key] 
        if no_w_d[key]!=0.0:
            w_d_t_1[key]/=no_w_d[key]
    if prog_bar:
        update_progress(3.0,7.0)      
        
    node_demand={}      
    no_high_nodes=5                                   
    for n,d in selected_g.nodes_iter(data=True):
        if n[1] not in node_demand.keys():
            node_demand[n[1]]={}
            w_c_t_1[n[1]] = np.zeros((T-1,noSamples))
            no_w_c[n[1]]=0 
        node_demand[n[1]]['w_'+str(n)]=abs(d['data']['inf_data'].demand)
        w_c_t_1[n[1]]+=w_t_1['w_'+str(n)]
        no_w_c[n[1]]+=1
    
    for u,v,a in selected_g.edges_iter(data=True):
        if not a['data']['inf_data'].is_interdep:
            layer = a['data']['inf_data'].layer
            if layer not in y_c_t_1.keys():
                y_c_t_1[layer] = np.zeros((T-1,noSamples))
                no_y_c[layer]=0 
            y_c_t_1[layer]+=y_t_1['y_'+str(u)+','+str(v)]/2.0
            no_y_c[layer]+=0.5
        
    for nn in node_demand.keys():
        node_demand_highest = dict(sorted(node_demand[nn].items(), key = itemgetter(1),
                                          reverse = True)[:no_high_nodes]) 
        w_h_t_1[nn] = np.zeros((T-1,noSamples)) 
        for nhd in node_demand_highest.keys():
             w_h_t_1[nn]+=w_t_1[nhd]/no_high_nodes
        
        w_c_t_1[nn]/=no_w_c[nn]
        y_c_t_1[nn]/=no_y_c[nn]    
    if prog_bar:
        update_progress(6.0,7.0)
                        
    for u,v,a in selected_g.edges_iter(data=True):
        if not a['data']['inf_data'].is_interdep:
            key = 'y_'+str(u)+','+str(v)               
            if key in arc_keys:
                y_n_t_1[key] = np.zeros((T-1,noSamples))
                y_n_t_1[key]+=w_t_1['w_'+str(v)]/2.0
                y_n_t_1[key]+=w_t_1['w_'+str(u)]/2.0
    if prog_bar:
        update_progress(7.0,7.0) 
    
    return w_n_t_1,w_d_t_1,w_a_t_1,w_h_t_1,w_c_t_1,y_n_t_1,y_c_t_1

def train_model(train_data,exclusions):
    print('\nTraining models:')
    logistic_model={}
    trace={}
    t_suf = time.strftime("%Y%m%d")
    folder_name = 'parameters'+t_suf
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for key, val in train_data.items():
        # update_progress(train_data.keys().index(key),len(train_data.keys()))
        variables = list(val.columns)
        exclusions+=['index','sample']
        if key[0]=='w':
            dependent =['w_t']
        elif key[0]=='y':
            dependent =['y_t']
        formula = make_formula(variables,exclusions,dependent)

        with pm.Model() as logistic_model[key]:
            pm.glm.GLM.from_formula(formula,val,family=pm.glm.families.Binomial()) 
            trace[key] = pm.sample(1500, tune=1000,chains=2, cores=2,init='adapt_diag')
            estParam = pm.summary(trace[key]).round(4)
            print(estParam)     
            estParam.to_csv(folder_name+'/model_parameters_'+key+'.txt',
                            header=True, index=True, sep=' ', mode='w')
            # pm.model_to_graphviz(logistic_model[key]).render(filename='Parameters\\model_graph_%s'%(key,))
 
    return trace,logistic_model
    
def test_model(train_data,test_data,trace_all,model_all,exclusions,plot=True):
#    plt.rc('text', usetex=True)
#    plt.rcParams.update({'font.size': 14})
#    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    plt.close('all')
    t_suf = time.strftime("%Y%m%d")
    folder_name = 'parameters'+t_suf
    for key, val in train_data.items():
        trace = trace_all[key]
        model = model_all[key]
        
        ''' Traceplot '''
        if plot:
            pm.traceplot(trace, combined=False) 
        
        ''' Generate and plot samples from the trained model '''
        ppc = pm.sample_posterior_predictive(trace, samples=1000, model=model)
             
        
        ''' Acceptance rate '''
        accept = trace.get_sampler_stats('mean_tree_accept', burn=0)
        if plot:
            f, ax = plt.subplots(2, 2)
            sns.distplot(accept, kde=False, ax=ax[0,0])
            ax[0,0].set_title('Distribution of Tree Acceptance Rate')
            print( 'Mean Acceptance rate: '+ str(accept.mean()))
            print( 'Index of all diverging transitions: '+ str(trace['diverging'].nonzero()))
            print( 'No of diverging transitions: '+ str(len(trace['diverging'].nonzero()[0])))
    
            ''' Energy level '''
            energy = trace['energy']
            energy_diff = np.diff(energy)
            sns.distplot(energy - energy.mean(), label='energy',ax=ax[0,1])
            sns.distplot(energy_diff, label='energy diff',ax=ax[0,1])
            ax[0,1].set_title('Distribution of energy level vs. change of energy between successive samples')
            ax[0,1].legend()    
    
        
        ''' Prediction & data '''
        variables = list(val.columns)
        exclusions+=['index','sample']
        if key[0]=='w':
            dependent =['w_t']
            x=np.array(train_data[key]['w_t'])
        elif key[0]=='y':
            dependent =['y_t']
            x=np.array(train_data[key]['y_t'])
        formula = make_formula(variables,exclusions,dependent)
        y=ppc['y'].T.mean(axis=1)
        if plot:
            ax[1,0].scatter(x,y,alpha=0.1)
            ax[1,0].plot([0,1],[0,1],'r')
            ax[1,0].set_title('Data vs. Prediction: training data ')    
            ax[1,0].set_xlabel('data')
            ax[1,0].set_ylabel('Mean Prediction')
        with open(folder_name+'/R2.txt', mode='a') as f:
            print('\n'+key+' Training data R2: '+str(sklearn.metrics.r2_score(x,y)))
            f.write('\n'+key+' Training data R2: '+str(sklearn.metrics.r2_score(x,y)))
            f.close()
                
        with pm.Model() as logistic_model_test:
            pm.glm.GLM.from_formula(formula,test_data[key],family=pm.glm.families.Binomial())
            ppc_test = pm.sample_posterior_predictive(trace, samples=1000, model=logistic_model_test)
            if key[0]=='w':
                x=np.array(test_data[key]['w_t'])
            elif key[0]=='y':
                x=np.array(test_data[key]['y_t'])           
            y=ppc_test['y'].T.mean(axis=1)
            if plot:
                ax[1,1].scatter(x,y,alpha=0.1)
                ax[1,1].plot([0,1],[0,1],'r')
                ax[1,1].set_title('Data vs. Prediction: test data ')  
                ax[1,1].set_xlabel('data')
                ax[1,1].set_ylabel('Mean Prediction')
            with open(folder_name+'/R2.txt', mode='a') as f:
                print('\n'+key+' Test data R2: '+str(sklearn.metrics.r2_score(x,y)))
                f.write('\n'+key+' Test data R2: '+str(sklearn.metrics.r2_score(x,y)))
                f.close()
                
    return ppc,ppc_test

def compare_resotration(pred_s,samples,costs,s,s_itr,res,network_object,failSce_param,
                        initial_net,layers,real_results_dir,model_folder,param_folder):
    print( '\nPrediction '+str(pred_s)+': time step ',)
    
    # Define a few vars and lists
    no_samples = samples[samples.keys()[0]].shape[1]
    T = samples[samples.keys()[0]].shape[0]
    keys=samples.keys()
    cost_names=costs.keys()
    node_cols=['w_t','w_t_1','sample','time','Rc','w_n_t_1','w_a_t_1','w_d_t_1','w_h_t_1','w_c_t_1','y_c_t_1']
    arc_cols=['y_t','y_t_1','sample','time','Rc','y_n_t_1','y_c_t_1']
    
    '''Isolate input data for the current sample'''
    costs_iso={}
    for i in cost_names: 
        costs_iso[i]=np.asarray([x  for x in costs[i][:,s_itr]]).reshape((T, -1))
    samples_iso={}
    for key in keys:
        samples_iso[key]=np.asarray([x  for x in samples[key][:,s_itr]]).reshape((T, -1))

    ''' read cost from the actual data computed by INDP '''
    if pred_s==0:
        real_results=indputils.INDPResults()
        real_results=real_results.from_csv(real_results_dir,s,suffix="")
        for t in range(T):
            row = np.array([s,t,res,-1,'data',costs_iso[key]['Total'][t,0],
                real_results[t]['run_time'],costs_iso[key]['Under_Supply_Perc'][t,0]])
            write_results_to_file(row)
            
    '''Writ initial costs to file'''
    print('0.',)
    row = np.array([s,0,res,pred_s,'predicted',costs_iso['Total'][0,0],0.0,costs_iso['Under_Supply_Perc'][0,0]])
    write_results_to_file(row)  
    
    '''Predict restoration plans'''
    run_times=[]
    for t in range(T-1):
        '''Predict a full scenario'''
        print( str(t+1)+'.',)
        start_time = time.time()
        
        w_n_t_1,w_d_t_1,w_a_t_1,w_h_t_1,w_c_t_1,y_n_t_1,y_c_t_1=extract_features(samples_iso,initial_net,keys,prog_bar=False)
        
        costs_normed={}
        for i in cost_names: 
            costs_normed[i] = normalize_costs(costs_iso[i],i,costs_iso['Total'][0])
        
        decision_vars={0:{}} #0 becasue iINDP
        for key in keys:
            if key[0]=='w':                   
                decision_vars[0][key]=predict_next_step(key,s,t,samples_iso,param_folder,
                            model_folder,node_cols,arc_cols,costs_normed,res,
                            w_n_t_1,w_d_t_1,w_a_t_1,w_h_t_1,w_c_t_1,y_n_t_1,y_c_t_1)
            if key[0]=='y' and key not in decision_vars[0].keys():
                decision_vars[0][key]=samples_iso[key][t+1,0]
                u,v=arc_to_node(key)
                decision_vars[0]['y_'+v+','+u]=samples_iso[key][t+1,0]
            samples_iso[key][t+1,0]=decision_vars[0][key]
                           
        '''Calculate the cost of scenario'''
        flow_results=flow.flow_problem(network_object,v_r=0,
                        layers=layers,controlled_layers=layers,
                      decision_vars=decision_vars,print_cmd=True, time_limit=None)
        apply_recovery(network_object,flow_results[1],0)
        run_times.append(time.time()-start_time)
        row = np.array([s,t+1,res,pred_s,'predicted',flow_results[1][0]['costs']['Total'],
                        run_times[-1],flow_results[1][0]['costs']['Under Supply Perc']])
        write_results_to_file(row)    
        
        '''Update the cost dict for the next time step'''
        for h in cost_names:
            costs_iso[h][t+1]=flow_results[1][0]['costs'][h.replace('_', ' ')]
        
        #'''Write models to file'''  
        # indp.save_INDP_model_to_file(flow_results[0],'./models',t,l=0)           
    
    sum_real_rep_perc=np.zeros((T)) 
    sum_pred_rep_perc=np.zeros((T))       
    for key in keys:
        if samples_iso[key][0,0]!=1.0:
            real_rep_time = T-sum(samples[key][:,s_itr])
            pred_rep_time = T-sum(samples_iso[key][:,0])
            row = np.array([s,key,res,pred_s,real_rep_time,pred_rep_time,
                            real_rep_time-pred_rep_time])
            write_results_to_file(row,filename='pred_error')
        sum_real_rep_perc+=samples[key][:,s_itr]
        sum_pred_rep_perc+=samples_iso[key][:,0]

    for t in range(T):
        no_elements = len(keys)
        row = np.array([s,t,res,pred_s,sum_real_rep_perc[t]/no_elements,
                        sum_pred_rep_perc[t]/no_elements])   
        write_results_to_file(row,filename='rep_prec')

    return run_times

def predict_next_step(key,s,t,samples_iso,param_folder,model_folder,
                      node_cols,arc_cols,costs_normed,res,
                      w_n_t_1,w_d_t_1,w_a_t_1,w_h_t_1,w_c_t_1,y_n_t_1,y_c_t_1):
    l = int(key[-2])
    T = samples_iso[samples_iso.keys()[0]].shape[0]
    cost_names=costs_normed.keys()
    pred_decision=0.0
    if samples_iso[key][t,0]==1:
        pred_decision=1.0
    else:
        # Making model formula                 
        parameters=pd.read_csv(param_folder+'/model_parameters_'+key+'.txt',delimiter=' ')
        varibales = list(parameters['Unnamed: 0'])
        varibales.remove('Intercept')

        if key[0]=='w':
            dependent =['w_t'] 
            cols=node_cols+cost_names
            special_feature=[w_n_t_1[key][t,0],w_a_t_1[key][t,0],w_d_t_1[key][t,0],
                      w_h_t_1[l][t,0],w_c_t_1[l][t,0],y_c_t_1[l][t,0]]
        elif key[0]=='y':
            dependent =['y_t']
            cols=arc_cols+cost_names
            special_feature=[y_n_t_1[key][t,0],y_c_t_1[l][t,0]] 
        formula = make_formula(varibales,[],dependent)
        norm_cost_values=[costs_normed[x][t,0] for x in cost_names]
        basic_features=[samples_iso[key][t+1,0],samples_iso[key][t,0],s,(t+1)/float(T-1),res/100.0]
        row = np.array(basic_features+special_feature+norm_cost_values)
        
        with pm.Model() as logistic_model_pred:
            pm.glm.GLM.from_formula(formula,
                                    pd.Series(row,index=cols),
                                    family=pm.glm.families.Binomial())
            trace = pm.load_trace(model_folder+'/'+key)
            ppc_test = pm.sample_posterior_predictive(trace,samples=1,
                                                      progressbar=False)
            pred_decision=ppc_test['y'][0][0]     
            # print `samples_iso[key][t,0]`+'->'+`pred_decision`
    return pred_decision

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
            u,v=arc_to_node(key)
            if 'y_'+v+','+u in duplicates:
                duplicate=True
        else:
            sys.exit('Wrong variable name: ' + key)
    
        if not duplicate:
            msk = np.random.rand(len(data)) < 0.8
            test_data[key]=data[~msk]
            train_data[key]=data[msk]
            test_data[key]=test_data[key].reset_index()
            train_data[key]=train_data[key].reset_index()
    return train_data,test_data

def save_initial_data(initial_net,samples,costs):
    t_suf = time.strftime("%Y%m%d")
    folder_name = 'data'+t_suf
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    pickle.dump([samples,costs,initial_net], open(folder_name+'/initial_data.pkl', "wb" ))

def save_prepared_data(train_data,test_data):
    t_suf = time.strftime("%Y%m%d")
    folder_name = 'data'+t_suf
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    pickle.dump([train_data,test_data], open(folder_name+'/train_test_data.pkl', "wb" ))

def save_traces(trace):
    t_suf = time.strftime("%Y%m%d")
    fname={}
    folder_name = 'traces'+t_suf
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for key,val in trace.items():
        fname[key]=pm.save_trace(trace[key],directory=folder_name+'/'+key,overwrite=True)  

def make_formula(variables,exclusions,dependent):
    predictors= [i for i in variables if i not in exclusions+dependent]
    formula=dependent[0]+'~'
    for i in predictors:
        formula+=i+'+'
    return formula[:-1]

def discretize(x):
    x_dis=[]
    for i in x:
        if i>0.6:
           x_dis.append(1.0)
        else:
           x_dis.append(0.0)
    return x_dis

def logistic(l):
    return 1 / (1 + pm.math.exp(-l))

def arc_to_node(arc_key):
    arc_id = arc_key[2:].split('),(') 
    u = arc_id[0]+')'
    v = '('+arc_id[1] 
    return u,v

def key_to_tuple(key):
    raw=re.findall(r'\b\d+\b', key)
    if key[0]=='w':
        return (int(raw[0]),int(raw[1]))
    elif key[0]=='y':
        return (int(raw[0]),int(raw[1])),(int(raw[2]),int(raw[3]))
    
def write_results_to_file(row,filename='results'):
    t_suf = time.strftime("%Y%m%d")
    folder_name = './results'+t_suf
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open('./results'+t_suf+'/'+filename+t_suf+'.txt', 'a') as filehandle:
        for i in row:
            filehandle.write('%s\t' % i)
        filehandle.write('\n')   
        filehandle.close()

''' Data should be normailzied so that when using models such normalization is reproducible
at each time step including the first one. Therefore, the normalization should be based
on information that we have before starting the restoration'''
def normalize_costs(costs, i):
    T = int(costs.shape[0])
    S = int(costs.shape[1])
    
    costs_normed=np.zeros((T,S))
    if i in ['Total','Over_Supply','Under_Supply',
             'Under_Supply_layer','Total_layer','Over_Supply_layer']:
        for s in range(S):
            costs_normed[:,s] = costs[:,s]/costs[0,s]
    elif i in ['Under_Supply_Perc','Under_Supply_Perc_layer']:
        for s in range(S):
            costs_normed[:,s] = costs[:,s]       
    elif i in ['Node','Arc','Space_Prep','Node_layer','Arc_layer','Space_Prep_layer','Flow','Flow_layer']:
        for s in range(S):
            for t in range(T):
                history_vec=costs[:t+1,s]
                norm_factor=np.linalg.norm(history_vec)
                if norm_factor!=0:
                    costs_normed[t,s] = costs[t,s]/norm_factor
                else:
                    costs_normed[t,s] = costs[t,s]
    elif i=='Total_no_disconnection':
        pass
    else:
        sys.exit('Wrong cost name: '+i)
    return costs_normed

def find_sce_index(s, res, mis_sce_file):   
    if os.path.isfile(mis_sce_file):
        mis_sce=np.loadtxt(mis_sce_file)
        mis_sce_filtered_by_res=mis_sce[mis_sce[:,0]==res,:]
        if s in mis_sce_filtered_by_res[:,2]:
            return -1
        else:
            mis_sce_before_s=[x for x in mis_sce_filtered_by_res[:,2] if x<s]
            return s-len(mis_sce_before_s)
    else:
        sys.exit('No nissing scenario file: '+mis_sce_file)    
def update_progress(progress,total):
    print( '\r[%s] %1.1f%%' % ('#'*int(progress/float(total)*20), (progress/float(total)*100)),)
    sys.stdout.flush()