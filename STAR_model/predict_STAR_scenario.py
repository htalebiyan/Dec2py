import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import cPickle as pickle
from functools import partial
import indp 
import copy
import sys
from indp import *
import flow
from operator import itemgetter 
import STAR_utils
import pymc3 as pm
import theano
    
def predict_resotration(initial_states,initial_costs,initial_run_time,pred_dict,failSce_param,params):
    print '\nPredicting sample '+`pred_dict['sample']`+': time step ',
    
    # Define a few vars and lists
    keys=initial_states.keys()
    node_cols=['w_t','w_t_1','time','Rc','w_n_t_1','w_a_t_1','w_d_t_1','w_h_t_1','w_c_t_1','y_c_t_1']
    arc_cols=['y_t','y_t_1','time','Rc','y_n_t_1','w_c_t_1','y_c_t_1']
    num_pred = pred_dict['num_pred']
    T = params['NUM_ITERATIONS']
    net_obj = {x:copy.deepcopy(params['N']) for x in range(num_pred)}
    #initialize prediction and cost vectors
    predictions={}
    for key in keys:
        predictions[key]=np.ones((T+1,num_pred))
        predictions[key][0,:]=initial_states[key]
    costs={}
    for c in initial_costs.keys():
        costs[c]=np.zeros((T+1,num_pred))
        costs[c][0,:]=initial_costs[c]     
        
    # Writ initial costs to file
    print `0`+'.',
    for pred_s in range(num_pred):
        row = np.array([pred_dict['sample'],0,params['V'],pred_s,'predicted',initial_costs['Total'],
                        initial_run_time,initial_costs['Under_Supply_Perc']])
        STAR_utils.write_results_to_file(row)  
    
    #Predict restoration plans
    run_times=[]
    for t in range(T): # t is the time index for previous time step
        # Predict one step of the scenario'''
        print `t+1`+'.',
        start_time = time.time()
        
        feature_dict,layer_dict=extract_features(predictions,net_obj,t,params['L'],prog_bar=False)
        
        costs_normed={}
        for c in initial_costs.keys(): 
            costs_normed[c] = STAR_utils.normalize_costs(costs[c],c)
        
        # decision_vars={0:{}} #0 becasue iINDP
        # for key in keys:    
        #     if key[0]=='w':            
        predictions=predict_next_step(t,predictions,pred_dict,node_cols,arc_cols,
                                      costs_normed,params['V'],feature_dict,layer_dict)
            #     predictions[key][t+1,:]=decision_vars[0][key]
            # if key[0]=='y' and key not in decision_vars[0].keys():
            #     pass
                # decision_vars[0][key]=predic[key][t+1,0]
                # u,v=arc_to_node(key)
                # decision_vars[0]['y_'+v+','+u]=samples_iso[key][t+1,0]
            
                           
    #     '''Calculate the cost of scenario'''
    #     flow_results=flow.flow_problem(network_object,v_r=0,
    #                     layers=layers,controlled_layers=layers,
    #                   decision_vars=decision_vars,print_cmd=True, time_limit=None)
    #     apply_recovery(network_object,flow_results[1],0)
    #     run_times.append(time.time()-start_time)
    #     row = np.array([s,t+1,res,pred_s,'predicted',flow_results[1][0]['costs']['Total'],
    #                     run_times[-1],flow_results[1][0]['costs']['Under Supply Perc']])
    #     write_results_to_file(row)    
        
    #     '''Update the cost dict for the next time step'''
    #     for h in cost_names:
    #         costs_iso[h][t+1]=flow_results[1][0]['costs'][h.replace('_', ' ')]
        
        #'''Write models to file'''  
        # indp.save_INDP_model_to_file(flow_results[0],'./models',t,l=0)           
    
    return predictions

def predict_next_step(t,predictions,pred_dict,node_cols,arc_cols,
                      costs_normed,res,feature_dict,layer_dict):
    # define vars and lists
    keys = predictions.keys()
    T = predictions[predictions.keys()[0]].shape[0]
    cost_names=costs_normed.keys()
    num_pred = pred_dict['num_pred']

    # Making model formula
    formula={'w':'','y':''}
    logi_model={'w':[],'y':[]}
    x_shared={'w':0,'y':0}
    for ke in ['w']: #,'y'
        key=[i for i in keys if i[0]==ke][0]
        parameters=pd.read_csv(pred_dict['param_folder']+'/model_parameters_'+key+'.txt',delimiter=' ')
        varibales = list(parameters['Unnamed: 0'])
        varibales.remove('Intercept')
        with pm.Model() as logi_model['w']:
            formula[ke] = STAR_utils.make_formula(varibales,[],[ke+'_t'])
            # x_shared[ke]=theano.shared()
            dummy=pd.DataFrame(np.zeros((num_pred,len(varibales)+1)),columns=varibales+[ke+'_t'])
            pm.glm.GLM.from_formula(formula[ke],dummy,
                                    family=pm.glm.families.Binomial())    
            
    for key in keys:      
        l = int(key[-2])     
        # initialize new predictions
        pred_decision=np.zeros(num_pred)
        for pred_s in range(num_pred):
            if predictions[key][t,pred_s]==1:
                pred_decision[pred_s]=1.0
                
        if pred_decision.any()==0: #if any pred_s needs prediction
            # Make input datframe
            data=pd.DataFrame()
            pred_idx = []
            for pred_s in range(num_pred):
                if pred_decision[pred_s]==0:
                    pred_idx.append(pred_s)
                    if key[0]=='w':
                        cols=node_cols+cost_names
                        special_feature=[feature_dict[key]['w_n_t_1'][pred_s],
                                         feature_dict[key]['w_a_t_1'][pred_s],
                                         feature_dict[key]['w_d_t_1'][pred_s],
                                         layer_dict[l]['w_h_t_1'][pred_s],
                                         layer_dict[l]['w_c_t_1'][pred_s],
                                         layer_dict[l]['y_c_t_1'][pred_s]]
                    elif key[0]=='y':
                        cols=arc_cols+cost_names
                        special_feature=[feature_dict[key]['y_n_t_1'][pred_s],
                                         layer_dict[l]['w_c_t_1'][pred_s],
                                         layer_dict[l]['y_c_t_1'][pred_s]]
                norm_cost_values=[costs_normed[x][t,pred_s] for x in cost_names]
                basic_features=[predictions[key][t+1,pred_s],predictions[key][t,pred_s],
                                (t+1)/float(T-1),res/100.0]
                row = np.array(basic_features+special_feature+norm_cost_values)
                data=data.append(pd.Series(row,index=cols),ignore_index=True)
            # x_shared[key[0]].set_value(theano.shared(data.to_numpy()))
            
            # Run models
            no_pred_itr=100
            with logi_model['w']:
                pm.set_data({'dummy': data})
                trace = pm.load_trace(pred_dict['model_dir']+'/'+key)
                ppc_test = pm.sample_posterior_predictive(trace,samples=no_pred_itr,
                                                          progressbar=False)
                for idx in pred_idx:
                    sum_state=0
                    for sr in range(no_pred_itr):
                        sum_state+=ppc_test['y'][sr][pred_idx.index(idx)]
                    if sum_state/float(no_pred_itr)<0.5:
                        pred_decision[idx]=0.0
                    else:
                        pred_decision[idx]=1.0
                # print `samples_iso[key][t,0]`+'->'+`pred_decision
        predictions[key][t+1,:]=pred_decision
    return predictions
             
def extract_features(predictions,net_obj,t,layers,prog_bar=True):
    # define vars and lists
    keys = predictions.keys()
    num_pred = int(predictions[keys[0]].shape[1])
    node_features=['w_n_t_1','w_a_t_1','w_d_t_1','no_w_n','no_w_d']
    
    # initialize feature dictionary
    feature_dict = {}
    for pred_s in range(num_pred):
        if pred_s==0:
            for key in keys:
                feature_dict[key]={}
                if key[0]=='w':
                    feature_dict[key]['w_t']=predictions[key][t+1,:]
                    feature_dict[key]['w_t_1']=predictions[key][t,:]
                    for feat in node_features:
                        feature_dict[key][feat]=np.zeros(num_pred)
                elif key[0]=='y':
                    feature_dict[key]['y_t']=predictions[key][t+1,:]
                    feature_dict[key]['y_t_1']=predictions[key][t,:]
                    feature_dict[key]['y_n_t_1']=np.zeros(num_pred)           
                else:
                    sys.exit('Wrong key.')

        for u,v,a in net_obj[pred_s].G.edges_iter(data=True):
            if not a['data']['inf_data'].is_interdep:
                feature_dict['w_'+`u`]['w_n_t_1'][pred_s]+=predictions['w_'+`v`][t,pred_s]/2.0
                feature_dict['w_'+`u`]['w_a_t_1'][pred_s]+=predictions['y_'+`u`+','+`v`][t,pred_s]/2.0
                feature_dict['w_'+`u`]['no_w_n'][pred_s]+=0.5
                feature_dict['w_'+`v`]['w_n_t_1'][pred_s]+=predictions['w_'+`u`][t,pred_s]/2.0
                feature_dict['w_'+`u`]['w_a_t_1'][pred_s]+=predictions['y_'+`u`+','+`v`][t,pred_s]/2.0
                feature_dict['w_'+`v`]['no_w_n'][pred_s]+=0.5 
                feature_dict['y_'+`u`+','+`v`]['y_n_t_1'][pred_s]+=(predictions['w_'+`u`][t,pred_s]+predictions['w_'+`v`][t,pred_s])/2.0
            else:
                feature_dict['w_'+`v`]['w_d_t_1'][pred_s]+=1-predictions['w_'+`u`][t,pred_s] # so that for non-dependent and those whose depndee nodes are functional, we get the same number
                feature_dict['w_'+`v`]['no_w_d'][pred_s]+=1.0
                
        for key in keys:
            if key[0]=='w':
                feature_dict[key]['w_n_t_1'][pred_s]/=feature_dict[key]['no_w_n'][pred_s]
                feature_dict[key]['w_a_t_1'][pred_s]/=feature_dict[key]['no_w_n'][pred_s] 
                if feature_dict[key]['no_w_d'][pred_s]!=0.0:
                    feature_dict[key]['w_d_t_1'][pred_s]/=feature_dict[key]['no_w_d'][pred_s]
  
    layer_dict={x:{} for x in layers}     
    node_demand={}      
    no_high_nodes=5
    for n,d in net_obj[pred_s].G.nodes_iter(data=True):
        if n[1] not in node_demand.keys():
            node_demand[n[1]]={}
        node_demand[n[1]]['w_'+`n`]=abs(d['data']['inf_data'].demand) 
        
    for pred_s in range(num_pred):                               
        for n,d in net_obj[pred_s].G.nodes_iter(data=True):
            if 'w_c_t_1' not in layer_dict[n[1]].keys():
                layer_dict[n[1]]['w_c_t_1'] = np.zeros(num_pred)
                layer_dict[n[1]]['no_w_c'] = np.zeros(num_pred) 
            layer_dict[n[1]]['w_c_t_1'][pred_s]+=predictions['w_'+`n`][t,pred_s]
            layer_dict[n[1]]['no_w_c'][pred_s]+=1.0
        
        for u,v,a in net_obj[pred_s].G.edges_iter(data=True):
            if not a['data']['inf_data'].is_interdep:
                l = a['data']['inf_data'].layer
                if 'y_c_t_1' not in layer_dict[l].keys():
                    layer_dict[l]['y_c_t_1'] = np.zeros(num_pred)
                    layer_dict[l]['no_y_c'] = np.zeros(num_pred) 
                layer_dict[l]['y_c_t_1'][pred_s]+=predictions['y_'+`u`+','+`v`][t,pred_s]/2.0
                layer_dict[l]['no_y_c'][pred_s]+=0.5
        
        for l in node_demand.keys():
            node_demand_highest = dict(sorted(node_demand[l].items(), key = itemgetter(1),
                                              reverse = True)[:no_high_nodes]) 
            if 'w_h_t_1' not in layer_dict[l].keys():
                layer_dict[l]['w_h_t_1'] = np.zeros(num_pred)
            for nhd in node_demand_highest.keys():
                  layer_dict[l]['w_h_t_1'][pred_s]+=predictions[nhd][t,pred_s]/no_high_nodes
            layer_dict[l]['w_c_t_1'][pred_s]/=layer_dict[l]['no_w_c'][pred_s]
            layer_dict[l]['y_c_t_1'][pred_s]/=layer_dict[l]['no_y_c'][pred_s] 
                        
    return feature_dict,layer_dict

def import_initial_data(params,failSce_param,suffix=''):  
    print('\nImport Data...')
    # Set root directories and params
    base_dir = failSce_param['Base_dir']
    damage_dir = failSce_param['Damage_dir'] 
    sample = failSce_param['sample']
    mag = failSce_param['mag']
    topology = None
    shelby_data = True
    ext_interdependency = None
    if failSce_param['type']=='Andres':
        ext_interdependency = '../data/INDP_4-12-2016'
    elif failSce_param['type']=='synthetic':  
        shelby_data = False  
        topology = failSce_param['topology']

    if not shelby_data:  
        InterdepNet,noResource,layers=initialize_network(BASE_DIR=base_dir,
                external_interdependency_dir=ext_interdependency,magnitude=mag,
                sample=sample,shelby_data=shelby_data,topology=topology) 
        params["V"]=noResource
        params["L"]=layers
    else:  
        InterdepNet,_,_=initialize_network(BASE_DIR=base_dir,
                external_interdependency_dir=ext_interdependency,sim_number=0,
                magnitude=6,sample=0,v=params["V"],shelby_data=shelby_data)  
          
    params["N"]=InterdepNet
    params["SIM_NUMBER"]=sample
    params["MAGNITUDE"]=mag
    
    if failSce_param['type']=='WU':
        add_Wu_failure_scenario(InterdepNet,DAM_DIR=damage_dir,noSet=sample,noSce=mag)
    elif failSce_param['type']=='ANDRES':
        add_failure_scenario(InterdepNet,DAM_DIR=damage_dir,magnitude=mag,v=params["V"],sim_number=sample)
    elif failSce_param['type']=='random':
        add_random_failure_scenario(InterdepNet,DAM_DIR=damage_dir,sample=mag) 
    elif failSce_param['type']=='synthetic':
        add_synthetic_failure_scenario(InterdepNet,DAM_DIR=damage_dir,topology=topology,config=mag,sample=sample)              
        
    states={}
    costs = {}
    for v in InterdepNet.G.nodes():
        states['w_'+`v`]=InterdepNet.G.node[v]['data']['inf_data'].functionality         
    for u,v,a in InterdepNet.G.edges(data=True):
        if not a['data']['inf_data'].is_interdep:  
            states['y_'+`u`+','+`v`]=InterdepNet.G[u][v]['data']['inf_data'].functionality

    indp_results=indp(params['N'],v_r=0,T=1,layers=params['L'],controlled_layers=params['L'],
                      print_cmd=False,time_limit=None)
    for h in indp_results[1][0]['costs'].keys():
        costs[h.replace(' ', '_')]=indp_results[1][0]['costs'][h]
    return states,costs,indp_results[1][0]['run_time']


t_suf = '20200428'
dirrrr='C:/Users/ht20/Documents/Files/STAR_models/Node_only_final_all_Rc'
failSce_param = {"type":"WU","sample":6,"mag":2,'Base_dir':"../data/Extended_Shelby_County/",
                 'Damage_dir':"../data/Wu_Damage_scenarios/" ,'topology':None}
pred_dict={"sample":0,'sample_index':0,'num_pred':4,
           'model_dir':dirrrr+'/traces'+t_suf,'param_folder':dirrrr+'/parameters'+t_suf }
params={"NUM_ITERATIONS":1,"V":5,"ALGORITHM":"INDP",'L':[1,2,3,4]}

states,costs,run_time=import_initial_data(params,failSce_param,suffix='') 
pred=predict_resotration(states,costs,0,pred_dict,failSce_param,params)