import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import _pickle as pickle
from functools import partial
import indp 
import copy
import sys
from indp import *
import flow
from operator import itemgetter 
import STAR_utils
import pymc3 as pm
import os

class node_model(object):
    def __init__(self,name,net_id):
        self.name=name
        self.type='n'
        self.net_id=net_id
        self.initial_state=1.0
        self.state_hist=0
        self.model_status=0
    def set_initial_state(self,initial_state):
        self.initial_state=initial_state
    def set_initialize_state_hist(self,T,num_pred):
        self.state_hist=np.ones((T+1,num_pred))
        self.state_hist[0,:]=self.initial_state
    def check_model_exist(param_folder):
        param_file=param_folder+'/model_parameters_'+self.name+'.txt'
        if os.path.exists(param_file):
            self.model_status=1  

class arc_model(object):
    def __init__(self,name,net_id):
        self.name=name
        self.type='a'
        self.dupl_name=dupl_name()
        self.net_id=net_id
        self.initial_state=1.0
        self.state_hist=0
        self.model_status=0
    def set_initial_state(self,initial_state):
        self.initial_state=initial_state
    def set_initialize_state_hist(self,T,num_pred):
        self.state_hist=np.ones((T+1,num_pred))
        self.state_hist[0,:]=self.initial_state
    def dupl_name():
        arc_id = arc_key[2:].split('),(') 
        u = arc_id[0]+')'
        v = '('+arc_id[1] 
        return 'y_'+v+','+u
    def check_model_exist(param_folder):
        param_file=param_folder+'/model_parameters_'+self.name+'.txt'
        if os.path.exists(param_file):
            self.model_status=1
        else:
            param_file_new=param_folder+'/model_parameters_'+self.dupl_name+'.txt'
            if os.path.exists(param_file_new):
                self.model_status=1 
                temp = self.name
                self.name=name=self.dupl_name
                self.dupl_name=temp
                
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
        states['w_'+str(v)]=InterdepNet.G.node[v]['data']['inf_data'].functionality         
    for u,v,a in InterdepNet.G.edges(data=True):
        if not a['data']['inf_data'].is_interdep:  
            states['y_'+str(u)+','+str(v)]=InterdepNet.G[u][v]['data']['inf_data'].functionality

    indp_results=indp(params['N'],v_r=0,T=1,layers=params['L'],controlled_layers=params['L'],
                      print_cmd=False,time_limit=None)
    for h in list(indp_results[1][0]['costs'].keys()):
        costs[h.replace(' ', '_')]=indp_results[1][0]['costs'][h]
    return states,costs,indp_results[1][0]['run_time']

def writ_pred_to_file(predictions,num_pred,sample,mag):
    for pred_s in range(num_pred):
        for key in list(predictions.keys()):
            t_suf = time.strftime("%Y%m%d")
            folder_name = './results'+t_suf
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(folder_name+'/actions'+str(pred_s)+'.txt', 'a') as filehandle:
                for i in row:
                    filehandle.write('%s\t' % i)
                filehandle.write('\n')   
                filehandle.close()         
                
def predict_resotration(initial_states,initial_costs,initial_run_time,pred_dict,failSce_param,params):
    print('\nPredicting sample '+str(pred_dict['sample'])+': time step ',)
    
    ''' Define a few vars and lists '''
    keys=list(initial_states.keys())
    node_cols=['w_t','w_t_1','time','Rc','w_n_t_1','w_a_t_1','w_d_t_1','w_h_t_1','w_c_t_1','y_c_t_1']
    arc_cols=['y_t','y_t_1','time','Rc','y_n_t_1','w_c_t_1','y_c_t_1']
    num_pred = pred_dict['num_pred']
    T = params['NUM_ITERATIONS']
    net_obj = {x:copy.deepcopy(params['N']) for x in range(num_pred)}
    pred_results={x:INDPResults() for x in range(num_pred)} 
    ''' Initialize prediction and cost vectors '''
    predictions={}
    for key in keys:
        predictions[key]=np.ones((T+1,num_pred))
        predictions[key][0,:]=initial_states[key]
    costs={}
    for c in list(initial_costs.keys()):
        costs[c]=np.zeros((T+1,num_pred))
        costs[c][0,:]=initial_costs[c]     
        
    ''' Writ initial costs to file '''
    print('0.',)
    for pred_s in range(num_pred):
        row = np.array([pred_dict['sample'],0,params['V'],pred_s,'predicted',initial_costs['Total'],
                        initial_run_time,initial_costs['Under_Supply_Perc']])
        STAR_utils.write_results_to_file(row)  
    
    '''Predict restoration plans'''
    run_times=[]
    for t in range(T): # t is the time index for previous time step
        ''' Predict one step of the scenario'''
        print(str(t+1)+'.'),
        start_time = time.time()
        feature_dict,layer_dict=extract_features(predictions,net_obj,t,params['L'],prog_bar=False)
        costs_normed={}
        for c in list(initial_costs.keys()): 
            costs_normed[c] = STAR_utils.normalize_costs(costs[c],c)
        predictions=predict_next_step(t,predictions,pred_dict,node_cols,arc_cols,
                                      costs_normed,params['V'],feature_dict,layer_dict)
        
        ''' Calculate the cost of scenario '''
        for pred_s in range(num_pred):
            decision_vars={0:{}} #0 becasue iINDP 
            for key in keys:
                if key[0]=='w':
                    decision_vars[0][key]=predictions[key][t+1,pred_s]
                if key[0]=='y' and key not in list(decision_vars[0].keys()):
                    decision_vars[0][key]=predictions[key][t+1,pred_s]
                    u,v=STAR_utils.arc_to_node(key)
                    decision_vars[0]['y_'+v+','+u]=predictions[key][t+1,pred_s]
                    
            flow_results=flow.flow_problem(net_obj[pred_s],v_r=0,layers=params['L'],
                        controlled_layers=params['L'],decision_vars=decision_vars,
                        print_cmd=True, time_limit=None)
            pred_results[pred_s].extend(flow_results[1],t_offset=t+1)
            apply_recovery(net_obj[pred_s],flow_results[1],0)
            # run_times.append(time.time()-start_time)
            # row = np.array([s,t+1,res,pred_s,'predicted',flow_results[1][0]['costs']['Total'],
            #                 run_times[-1],flow_results[1][0]['costs']['Under Supply Perc']])
            # write_results_to_file(row)    
    
            ''' Update the cost dict for the next time step '''
            for h in list(initial_costs.keys()):
                costs[h][t+1,pred_s]=flow_results[1][0]['costs'][h.replace('_', ' ')]
                
                #'''Write models to file'''  
                # indp.save_INDP_model_to_file(flow_results[0],'./models',t,l=0)           
    
    return predictions,costs,pred_results

def predict_next_step(t,predictions,pred_dict,node_cols,arc_cols,
                      costs_normed,res,feature_dict,layer_dict):
    # define vars and lists
    keys = list(predictions.keys())
    T = predictions[list(predictions.keys())[0]].shape[0]
    cost_names=list(costs_normed.keys())
    num_pred = pred_dict['num_pred']
    no_pred_itr=10
      
    for key in keys:      
        l = int(key[-2])     
        # initialize new predictions
        pred_decision=np.zeros(num_pred)
        for pred_s in range(num_pred):
            if predictions[key][t,pred_s]==1:
                pred_decision[pred_s]=1.0
        mode_status = check_model_exist(key,param_folder)            
        if mode_status==1: #if any pred_s needs prediction
            param_file=pred_dict['param_folder']+'/model_parameters_'+key+'.txt'
            if os.path.exists(param_file):
                # Making model formula
                parameters=pd.read_csv(param_file,delimiter=' ')
                varibales = list(parameters['Unnamed: 0'])
                varibales.remove('Intercept')
                formula = STAR_utils.make_formula(varibales,[],[key[0]+'_t']) 
                
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
                    
                # Run models
                with pm.Model() as logi_model:
                    pm.glm.GLM.from_formula(formula,data,family=pm.glm.families.Binomial())  
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
            elif mode_status==2:
                
            else:
                pred_idx = []
                for pred_s in range(num_pred):
                    if pred_decision[pred_s]==0:
                        pred_decision[pred_s]=np.random.randint(2)
        predictions[key][t+1,:]=pred_decision
    return predictions
             
def extract_features(predictions,net_obj,t,layers,prog_bar=True):
    # define vars and lists
    keys = list(predictions.keys())
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

        for u,v,a in net_obj[pred_s].G.edges(data=True):
            if not a['data']['inf_data'].is_interdep:
                feature_dict['w_'+str(u)]['w_n_t_1'][pred_s]+=predictions['w_'+str(v)][t,pred_s]/2.0
                feature_dict['w_'+str(u)]['w_a_t_1'][pred_s]+=predictions['y_'+str(u)+','+str(v)][t,pred_s]/2.0
                feature_dict['w_'+str(u)]['no_w_n'][pred_s]+=0.5
                feature_dict['w_'+str(v)]['w_n_t_1'][pred_s]+=predictions['w_'+str(u)][t,pred_s]/2.0
                feature_dict['w_'+str(u)]['w_a_t_1'][pred_s]+=predictions['y_'+str(u)+','+str(v)][t,pred_s]/2.0
                feature_dict['w_'+str(v)]['no_w_n'][pred_s]+=0.5 
                feature_dict['y_'+str(u)+','+str(v)]['y_n_t_1'][pred_s]+=(predictions['w_'+str(u)][t,pred_s]+predictions['w_'+str(v)][t,pred_s])/2.0
            else:
                feature_dict['w_'+str(v)]['w_d_t_1'][pred_s]+=1-predictions['w_'+str(u)][t,pred_s] # so that for non-dependent and those whose depndee nodes are functional, we get the same number
                feature_dict['w_'+str(v)]['no_w_d'][pred_s]+=1.0
                
        for key in keys:
            if key[0]=='w':
                feature_dict[key]['w_n_t_1'][pred_s]/=feature_dict[key]['no_w_n'][pred_s]
                feature_dict[key]['w_a_t_1'][pred_s]/=feature_dict[key]['no_w_n'][pred_s] 
                if feature_dict[key]['no_w_d'][pred_s]!=0.0:
                    feature_dict[key]['w_d_t_1'][pred_s]/=feature_dict[key]['no_w_d'][pred_s]
  
    layer_dict={x:{} for x in layers}     
    node_demand={}      
    no_high_nodes=5
    for n,d in net_obj[pred_s].G.nodes(data=True):
        if n[1] not in list(node_demand.keys()):
            node_demand[n[1]]={}
        node_demand[n[1]]['w_'+str(n)]=abs(d['data']['inf_data'].demand) 
        
    for pred_s in range(num_pred):                               
        for n,d in net_obj[pred_s].G.nodes(data=True):
            if 'w_c_t_1' not in list(layer_dict[n[1]].keys()):
                layer_dict[n[1]]['w_c_t_1'] = np.zeros(num_pred)
                layer_dict[n[1]]['no_w_c'] = np.zeros(num_pred) 
            layer_dict[n[1]]['w_c_t_1'][pred_s]+=predictions['w_'+str(n)][t,pred_s]
            layer_dict[n[1]]['no_w_c'][pred_s]+=1.0
        
        for u,v,a in net_obj[pred_s].G.edges(data=True):
            if not a['data']['inf_data'].is_interdep:
                l = a['data']['inf_data'].layer
                if 'y_c_t_1' not in list(layer_dict[l].keys()):
                    layer_dict[l]['y_c_t_1'] = np.zeros(num_pred)
                    layer_dict[l]['no_y_c'] = np.zeros(num_pred) 
                layer_dict[l]['y_c_t_1'][pred_s]+=predictions['y_'+str(u)+','+str(v)][t,pred_s]/2.0
                layer_dict[l]['no_y_c'][pred_s]+=0.5
        
        for l in list(node_demand.keys()):
            node_demand_highest = dict(sorted(node_demand[l].items(), key = itemgetter(1),
                                              reverse = True)[:no_high_nodes]) 
            if 'w_h_t_1' not in list(layer_dict[l].keys()):
                layer_dict[l]['w_h_t_1'] = np.zeros(num_pred)
            for nhd in list(node_demand_highest.keys()):
                  layer_dict[l]['w_h_t_1'][pred_s]+=predictions[nhd][t,pred_s]/no_high_nodes
            layer_dict[l]['w_c_t_1'][pred_s]/=layer_dict[l]['no_w_c'][pred_s]
            layer_dict[l]['y_c_t_1'][pred_s]/=layer_dict[l]['no_y_c'][pred_s] 
                        
    return feature_dict,layer_dict




        
t_suf = ''
dirrrr='C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc'
failSce_param = {"type":"WU","sample":1,"mag":37,'Base_dir':"../data/Extended_Shelby_County/",
                 'Damage_dir':"../data/Wu_Damage_scenarios/" ,'topology':None}
pred_dict={"sample":0,'sample_index':0,'num_pred':4,
           'model_dir':dirrrr+'/traces'+t_suf,'param_folder':dirrrr+'/parameters'+t_suf }
params={"NUM_ITERATIONS":10,"V":5,"ALGORITHM":"INDP",'L':[1,2,3,4]}

states,costs,run_time=import_initial_data(params,failSce_param,suffix='') 
pred,cost,pred_results=predict_resotration(states,costs,0,pred_dict,failSce_param,params)
for key in list(pred.keys()):
    if pred[key].mean()==1:
        del pred[key]