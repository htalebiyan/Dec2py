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
        self.model_params=[]
        self.w_n_t_1=0
        self.w_a_t_1=0
        self.w_d_t_1=0
        self.degree=0
        self.neighbors=[]
        self.arcs=[]
        self.num_dependee=0
        self.dependees=[]
    def initialize_state_matrices(self,T,num_pred):
        self.state_hist=-1*np.ones((T+1,num_pred))
        self.state_hist[0,:]=self.initial_state
        self.w_n_t_1=np.zeros((T,num_pred))
        self.w_a_t_1=np.zeros((T,num_pred))
        self.w_d_t_1=np.zeros((T,num_pred))
    def add_neighbor(self,neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.arcs.append('y_'+self.name[2:]+','+neighbor[2:])
            self.arcs.append('y_'+neighbor[2:]+','+self.name[2:])
            self.degree+=1
    def add_dependee(self,dependee):
        if dependee not in self.dependees:
            self.dependees.append(dependee)
            self.num_dependee+=1
    def check_model_exist(self,param_folder):
        param_file=param_folder+'/model_parameters_'+self.name+'.txt'
        if os.path.exists(param_file):
            self.model_status=1  
            self.model_params=pd.read_csv(param_file,delimiter=' ')

class arc_model(object):
    def __init__(self,name,net_id):
        self.name=name
        self.type='a'
        self.net_id=net_id
        self.initial_state=1.0
        self.state_hist=0
        self.model_status=0
        self.model_params=[]
        self.y_n_t_1=0
        arc_id = name[2:].split('),(') 
        self.dupl_name='y_'+'('+arc_id[1]+','+arc_id[0]+')'
        self.end_nodes=['w_'+arc_id[0]+')','w_'+'('+arc_id[1]]
    def initialize_state_matrices(self,T,num_pred):
        self.state_hist=-1*np.ones((T+1,num_pred))
        self.state_hist[0,:]=self.initial_state
        self.y_n_t_1=np.zeros((T,num_pred))
    def check_model_exist(self,param_folder):
        param_file=param_folder+'/model_parameters_'+self.name+'.txt'
        if os.path.exists(param_file):
            self.model_status=1
            self.model_params=pd.read_csv(param_file,delimiter=' ')
        else:
            param_file_new=param_folder+'/model_parameters_'+self.dupl_name+'.txt'
            if os.path.exists(param_file_new):
                self.model_status=2 
                self.model_params=pd.read_csv(param_file_new,delimiter=' ')
                
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
     
    # initialize objects
    objs={}
    for v in InterdepNet.G.nodes():
        objs['w_'+str(v)]=node_model('w_'+str(v),v[1])
        objs['w_'+str(v)].initial_state=InterdepNet.G.node[v]['data']['inf_data'].functionality  
    for u,v,a in InterdepNet.G.edges(data=True):
        if not a['data']['inf_data'].is_interdep:
            if 'y_'+str(v)+','+str(u) not in list(objs.keys()):  
                objs['y_'+str(u)+','+str(v)]=arc_model('y_'+str(u)+','+str(v),v[1])
                objs['y_'+str(u)+','+str(v)].initial_state=InterdepNet.G[u][v]['data']['inf_data'].functionality
                objs['w_'+str(u)].add_neighbor('w_'+str(v))
                objs['w_'+str(v)].add_neighbor('w_'+str(u))
        else:
            objs['w_'+str(v)].add_dependee('w_'+str(u))
    
    # initial costs       
    indp_results=indp(params['N'],v_r=0,T=1,layers=params['L'],controlled_layers=params['L'],
                      print_cmd=False,time_limit=None)
    costs = {x:{} for x in params['L']+[0]}
    for h in list(indp_results[1][0]['costs'].keys()):
        costs[0][h.replace(' ', '_')]=indp_results[1][0]['costs'][h]
        for l in params['L']:
            costs[l][h.replace(' ', '_')]=indp_results[2][l][0]['costs'][h]
    return objs,costs,indp_results[1][0]['run_time']

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
                
def predict_resotration(objs,initial_costs,initial_run_time,pred_dict,failSce_param,params):
    print('\nPredicting sample '+str(pred_dict['sample'])+': time step ',)
    
    ''' Define a few vars and lists '''
    num_pred = pred_dict['num_pred']
    T = params['NUM_ITERATIONS']
    net_obj = {x:copy.deepcopy(params['N']) for x in range(num_pred)}
    pred_results={x:INDPResults() for x in range(num_pred)} 
    run_times={x:[0,0] for x in range(T+1)}
    run_times[0][0]=initial_run_time
    ''' Initialize prediction and cost vectors '''
    for key,val in objs.items(): 
        val.initialize_state_matrices(T,num_pred)
        val.check_model_exist(pred_dict['param_folder'])
    costs={}
    for l in list(initial_costs.keys()):
        costs[l]={}
        for h in list(initial_costs[l].keys()):
            costs[l][h]=np.zeros((T+1,num_pred))
            costs[l][h][0,:]=initial_costs[l][h]     
           
    '''Predict restoration plans'''

    for t in range(T): # t is the time index for previous time step
        print(str(t+1)+'.'),
        start_time = time.time()
        
        ''' Feature extraction'''
        layer_dict=extract_features(objs,net_obj,t,params['L'],num_pred,prog_bar=False)
        
        '''  Cost normalization '''
        costs_normed={}
        for c in list(initial_costs[0].keys()): 
            costs_normed[c] = STAR_utils.normalize_costs(costs[0][c],c)
        run_times[t+1][1]=predict_next_step(t,T,objs,pred_dict,costs_normed,params['V'],layer_dict,print_cmd=True)
        
        ''' Calculate the cost of scenario '''
        for pred_s in range(num_pred):
            decision_vars={0:{}} #0 becasue iINDP 
            for key,val in objs.items():
                decision_vars[0][val.name]=val.state_hist[t+1,pred_s]
                if val.type=='a':
                    decision_vars[0][val.dupl_name]=val.state_hist[t+1,pred_s]
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
            for h in list(initial_costs[0].keys()):
                costs[0][h][t+1,pred_s]=flow_results[1][0]['costs'][h.replace('_', ' ')]
                for l in params['L']:
                    costs[l][h][t+1,pred_s]=flow_results[2][l][0]['costs'][h.replace('_', ' ')]                
                #'''Write models to file'''  
                # indp.save_INDP_model_to_file(flow_results[0],'./models',t,l=0)           
        run_times[t+1][0]=time.time()-start_time
    return costs,pred_results,run_times

def predict_next_step(t,T,objs,pred_dict,costs_normed,res,layer_dict,print_cmd=True):
    '''define vars and lists'''
    cost_names=list(costs_normed.keys())
    num_pred = pred_dict['num_pred']
    no_pred_itr=10
    node_cols=['w_t','w_t_1','time','Rc','w_n_t_1','w_a_t_1','w_d_t_1','w_h_t_1','w_c_t_1','y_c_t_1']
    arc_cols=['y_t','y_t_1','time','Rc','y_n_t_1','w_c_t_1','y_c_t_1']    
    from_formula_calls=0
    for key,val in objs.items():      
        l = val.net_id  
        non_pred_idx = []
        '''initialize new predictions'''
        pred_decision=np.zeros(num_pred)
        for pred_s in range(num_pred):
            if val.state_hist[t,pred_s]==1:
                pred_decision[pred_s]=1
                non_pred_idx.append(pred_s)
        pred_idx=[x for x in range(num_pred) if x not in non_pred_idx] 
        
        if len(pred_idx): #if any pred_s needs prediction
            if val.model_status:
                '''Making model formula'''
                if print_cmd:
                    print('Predicting: '+key+'...')
                    if val.model_status==2:
                        print('A model with the duplicate name exists: '+key)
                varibales = list(val.model_params['Unnamed: 0'])
                varibales.remove('Intercept')
                formula = STAR_utils.make_formula(varibales,[],[key[0]+'_t']) 
                
                '''Make input datframe'''
                data=pd.DataFrame()
                for pred_s in pred_idx:
                    if val.type=='n':
                        cols=node_cols+cost_names
                        special_feature=[val.w_n_t_1[t,pred_s],
                                         val.w_a_t_1[t,pred_s],
                                         val.w_d_t_1[t,pred_s],
                                         layer_dict[l]['w_h_t_1'][pred_s],
                                         layer_dict[l]['w_c_t_1'][pred_s],
                                         layer_dict[l]['y_c_t_1'][pred_s]]
                    elif val.type=='a':
                        cols=arc_cols+cost_names
                        special_feature=[val.y_n_t_1[t,pred_s],
                                         layer_dict[l]['w_c_t_1'][pred_s],
                                         layer_dict[l]['y_c_t_1'][pred_s]]
                    norm_cost_values=[costs_normed[x][t,pred_s] for x in cost_names]
                    basic_features=[val.state_hist[t+1,pred_s],val.state_hist[t,pred_s],
                                    (t+1)/float(T-1),res/100.0]
                    row = np.array(basic_features+special_feature+norm_cost_values)
                    data=data.append(pd.Series(row,index=cols),ignore_index=True)
                    
                '''Run models'''
                with pm.Model() as logi_model:
                    pm.glm.GLM.from_formula(formula,data,family=pm.glm.families.Binomial()) 
                    from_formula_calls+=1
                    if val.model_status==1:
                        trace = pm.load_trace(pred_dict['model_dir']+'/'+val.name)
                    elif val.model_status==2:
                        trace = pm.load_trace(pred_dict['model_dir']+'/'+val.dupl_name)
                    ppc_test = pm.sample_posterior_predictive(trace,samples=no_pred_itr,
                                                              progressbar=False)
                    for idx in pred_idx:
                        sum_state=0
                        for sr in range(no_pred_itr):
                            sum_state+=ppc_test['y'][sr][pred_idx.index(idx)]
                        pred_decision[idx]=round(sum_state/float(no_pred_itr))

            elif val.model_status==0:
                for pred_s in pred_idx:
                    pred_decision[pred_s]=np.random.randint(2)
                    print('Prediction for '+key+' (prediction sample: '+pred_s+') was made randomly')
            else:
                sys.exit('Wrong model status: '+key)
                
        val.state_hist[t+1,:]=pred_decision
    return from_formula_calls

def extract_features(objs,net_obj,t,layers,num_pred,prog_bar=True):  
    # element feature extraction
    for pred_s in range(num_pred):
        for key,val in objs.items():
            if val.type=='n':
                for n in val.neighbors:
                    val.w_n_t_1[t,pred_s]+=objs[n].state_hist[t,pred_s]/val.degree
                for a in val.arcs:
                    try:
                        val.w_a_t_1[t,pred_s]+=objs[a].state_hist[t,pred_s]/val.degree
                    except:
                        pass
                for n in val.dependees:
                    val.w_d_t_1[t,pred_s]+=objs[n].state_hist[t,pred_s]/val.num_dependee
            elif val.type=='a':
                for n in val.end_nodes:
                    val.y_n_t_1[t,pred_s]+=objs[n].state_hist[t,pred_s]/2.0  
    # layer feature extraction
    layer_dict={x:{'w_c_t_1':np.zeros(num_pred),
                   'no_w_c':len([xx for xx in net_obj[0].G.nodes() if xx[1]==x]),
                   'y_c_t_1':np.zeros(num_pred),
                   'no_y_c':len([xx for xx in net_obj[0].G.edges() if xx[0][1]==x and xx[1][1]==x])//2,
                   'w_h_t_1':np.zeros(num_pred)} for x in layers}     
    node_demand={x:{} for x in layers}  
    node_demand_highest={}    
    no_high_nodes=5
    for n,d in net_obj[pred_s].G.nodes(data=True):
        node_demand[n[1]]['w_'+str(n)]=abs(d['data']['inf_data'].demand) 
    for l in list(node_demand.keys()):
        node_demand_highest[l] = dict(sorted(node_demand[l].items(), key = itemgetter(1),
                                          reverse = True)[:no_high_nodes])          
    for pred_s in range(num_pred):
        for key,val in objs.items():    
            if val.type=='n':
                layer_dict[val.net_id]['w_c_t_1'][pred_s]+=val.state_hist[t,pred_s]/layer_dict[val.net_id]['no_w_c']
            if val.type=='a':
                layer_dict[val.net_id]['y_c_t_1'][pred_s]+=val.state_hist[t,pred_s]/layer_dict[val.net_id]['no_y_c']
            if key in list(node_demand_highest[val.net_id].keys()):
                layer_dict[val.net_id]['w_h_t_1'][pred_s]+=val.state_hist[t,pred_s]/no_high_nodes
                        
    return layer_dict
  
t_suf = ''
dirrrr='C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc'
failSce_param = {"type":"WU","sample":1,"mag":37,'Base_dir':"../data/Extended_Shelby_County/",
                 'Damage_dir':"../data/Wu_Damage_scenarios/" ,'topology':None}
pred_dict={"sample":0,'sample_index':0,'num_pred':4,
           'model_dir':dirrrr+'/traces'+t_suf,'param_folder':dirrrr+'/parameters'+t_suf }
params={"NUM_ITERATIONS":10,"V":5,"ALGORITHM":"INDP",'L':[1,2,3,4]}

objs,costs,run_time=import_initial_data(params,failSce_param,suffix='') 
cost,pred_results,run_time=predict_resotration(objs,costs,run_time,pred_dict,failSce_param,params)
