"""Predicts restoration scenarios"""
import copy
import os
import sys
import time
import multiprocessing
from operator import itemgetter
import numpy as np
import pandas as pd
import pymc3 as pm
import indp
import indputils
import indpalt

class NodeModel():
    """Stores information for a node model """
    def __init__(self, name, net_id):
        self.name = name
        self.type = 'n'
        self.net_id = net_id
        self.initial_state = 1.0
        self.state_hist = 0
        self.model_status = 0
        self.model_params = []
        self.w_n_t_1 = 0
        self.w_a_t_1 = 0
        self.w_d_t_1 = 0
        self.degree = 0
        self.neighbors = []
        self.arcs = []
        self.num_dependee = 0
        self.dependees = []
    def initialize_state_matrices(self, time_step, num_pred):
        """Initializes state and predictor matrices """
        self.state_hist = np.ones((time_step+1, num_pred))
        self.state_hist[0, :] = self.initial_state
        self.w_n_t_1 = np.zeros((time_step, num_pred))
        self.w_a_t_1 = np.zeros((time_step, num_pred))
        self.w_d_t_1 = np.zeros((time_step, num_pred))
    def add_neighbor(self, neighbor):
        """Add a neighhbor node and updates connected arcs and degree"""
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.arcs.append('y_'+self.name[2:]+','+neighbor[2:])
            self.arcs.append('y_'+neighbor[2:]+','+self.name[2:])
            self.degree += 1
    def add_dependee(self, dependee):
        """Add a dependee node and updates the number of dependee nodes"""
        if dependee not in self.dependees:
            self.dependees.append(dependee)
            self.num_dependee += 1
    def check_model_exist(self, param_folder):
        """Find and checks the model for the element"""
        param_file = param_folder+'/model_parameters_'+self.name+'.txt'
        if os.path.exists(param_file):
            self.model_status = 1
            self.model_params = pd.read_csv(param_file, delimiter=' ')

class ArcModel():
    """Stores information for an arc model """
    def __init__(self, name, net_id):
        self.name = name
        self.type = 'a'
        self.net_id = net_id
        self.initial_state = 1.0
        self.state_hist = 0
        self.model_status = 0
        self.model_params = []
        self.y_n_t_1 = 0
        arc_id = name[2:].split('),(')
        self.dupl_name = 'y_'+'('+arc_id[1]+','+arc_id[0]+')'
        self.end_nodes = ['w_'+arc_id[0]+')', 'w_'+'('+arc_id[1]]
    def initialize_state_matrices(self, time_step, num_pred):
        """Initializes state and predictor matrices """
        self.state_hist = np.ones((time_step+1, num_pred))
        self.state_hist[0, :] = self.initial_state
        self.y_n_t_1 = np.zeros((time_step, num_pred))
    def check_model_exist(self, param_folder):
        """Find and checks the model for the element"""
        param_file = param_folder+'/model_parameters_'+self.name+'.txt'
        if os.path.exists(param_file):
            self.model_status = 1
            self.model_params = pd.read_csv(param_file, delimiter=' ')
        else:
            param_file_new = param_folder+'/model_parameters_'+self.dupl_name+'.txt'
            if os.path.exists(param_file_new):
                self.model_status = 2
                self.model_params = pd.read_csv(param_file_new, delimiter=' ')

def predict_resotration(obj, lyr, t_step, pred_dict):
    """ Predicts restoration plans for a given time step and layer"""
    ### Define a few vars and lists ###
    num_pred = pred_dict['num_pred']
    pred_results = {x:indputils.INDPResults(obj.layers) for x in range(num_pred)}
    run_times = [0, 0]
    ### Initialize element and network objects ###
    model_objs = initialize_model_objs(obj.net, lyr)
    for key, val in model_objs.items():
        val.initialize_state_matrices(1, num_pred)
        val.check_model_exist(pred_dict['param_folder'])

    costs = {}
    for h in obj.results_real.cost_types:
        costs[h.replace(' ', '_')] = np.zeros((t_step, num_pred))
        for t in range(t_step):
            costs[h.replace(' ', '_')][t, :] = obj.results_real.results[t]['costs'][h]
    ###Predict restoration plans for next time step###
    start_time = time.time()
    ### Feature extraction###
    layer_dict = extract_features(model_objs, obj.net, 0, obj.layers, num_pred)
    ###  Cost normalization ###
    costs_normed = {}
    for c in list(costs.keys()):
        costs_normed[c] = normalize_costs(costs[c], c)
    run_times[1] = predict_next_step(t_step, obj.time_steps, model_objs, pred_dict, lyr,
                                     costs_normed, pred_dict['V'], layer_dict, print_cmd=False)
    run_times[0] = time.time()-start_time
    ### Calculate the cost of scenario ###
    for pred_s in range(num_pred):
        start_time = time.time()
        #: Make the dict that used to fix the states of elements (only for the current layer)
        decision_vars = {0:{}} #0 becasue iINDP
        for key, val in model_objs.items():
            if val.net_id == lyr:
                decision_vars[0][val.name] = val.state_hist[1, pred_s]
                if val.type == 'a':
                    decision_vars[0][val.dupl_name] = val.state_hist[1, pred_s]
        #: Make functionality dic: each agent evaluates the performance seperatly
        neg_layer = [x for x in obj.layers if x != lyr] 
        functionality = obj.judgments.create_judgment_dict(obj, neg_layer)
        #: evaluates the performance
        flow_results = indpalt.flow_problem(obj.net, v_r=0, layers=obj.layers,
                                         controlled_layers=[lyr],
                                         decision_vars=decision_vars,
                                         functionality = functionality,
                                         print_cmd=True, time_limit=None)
        pred_results[pred_s].extend(flow_results[1], t_offset=1)
        #: Update the cost dict and run times ###
        if run_times[1] > 0:
            equiv_run_time = run_times[0]/run_times[1]+(time.time()-start_time)
        else:
            equiv_run_time = run_times[0]+(time.time()-start_time)
        pred_results[pred_s].results[1]['run_time'] = equiv_run_time
        ####Write models to file###
        # indp.save_INDP_model_to_file(flow_results[0], './models',t, l = 0)
    return pred_results

def initialize_model_objs(interdep_net, layer):
    """Imports initial data and initializes element objects"""
    model_objs = {}
    for v in interdep_net.G.nodes():
        model_objs['w_'+str(v)] = NodeModel('w_'+str(v), v[1])
        model_objs['w_'+str(v)].initial_state = interdep_net.G.nodes[v]['data']['inf_data'].functionality
    for u, v, a in interdep_net.G.edges(data=True):
        if not a['data']['inf_data'].is_interdep:
            if 'y_'+str(v)+','+str(u) not in list(model_objs.keys()):
                model_objs['y_'+str(u)+','+str(v)] = ArcModel('y_'+str(u)+','+str(v), v[1])
                model_objs['y_'+str(u)+','+str(v)].initial_state = interdep_net.G[u][v]['data']['inf_data'].functionality
                model_objs['w_'+str(u)].add_neighbor('w_'+str(v))
                model_objs['w_'+str(v)].add_neighbor('w_'+str(u))
        elif v[1] == layer:
            model_objs['w_'+str(v)].add_dependee('w_'+str(u))
    return model_objs

def predict_next_step(time_step, T, objs, pred_dict, lyr, costs_normed, res,
                      layer_dict, print_cmd=True):
    """define vars and lists"""
    cost_names = list(costs_normed.keys())
    num_pred = pred_dict['num_pred']
    no_pred_itr = 10
    res_unused = {x:pred_dict['V'] for x in range(num_pred)}
    node_cols = ['w_t', 'w_t_1', 'time', 'Rc', 'w_n_t_1', 'w_a_t_1', 'w_d_t_1',
                 'w_h_t_1', 'w_c_t_1', 'y_c_t_1']
    arc_cols = ['y_t', 'y_t_1', 'time', 'Rc', 'y_n_t_1', 'w_c_t_1', 'y_c_t_1']
    from_formula_calls = 0
    for key, val in objs.items():
        if val.net_id != lyr:
            continue
        ###initialize new predictions###
        non_pred_idx = []
        pred_decision = np.zeros(num_pred)
        for pred_s in range(num_pred):
            if val.state_hist[0, pred_s] == 1:
                pred_decision[pred_s] = 1
                non_pred_idx.append(pred_s)
        pred_idx = [x for x in range(num_pred) if x not in non_pred_idx]
        ###if any pred_s needs prediction###
        if len(pred_idx):
            if val.model_status:
                ###Making model formula###
                if print_cmd:
                    print('Predicting: '+key+'...')
                    if val.model_status == 2:
                        print('A model with the duplicate name exists: '+key)
                varibales = list(val.model_params['Unnamed: 0'])
                varibales.remove('Intercept')
                formula = make_formula(varibales, [], [key[0]+'_t'])
                ###Make input datframe###
                data = pd.DataFrame()
                for pred_s in pred_idx:
                    if val.type == 'n':
                        cols = node_cols+cost_names
                        special_feature = [val.w_n_t_1[0, pred_s],
                                           val.w_a_t_1[0, pred_s],
                                           val.w_d_t_1[0, pred_s],
                                           layer_dict[val.net_id]['w_h_t_1'][pred_s],
                                           layer_dict[val.net_id]['w_c_t_1'][pred_s],
                                           layer_dict[val.net_id]['y_c_t_1'][pred_s]]
                    elif val.type == 'a':
                        cols = arc_cols+cost_names
                        special_feature = [val.y_n_t_1[0, pred_s],
                                           layer_dict[val.net_id]['w_c_t_1'][pred_s],
                                           layer_dict[val.net_id]['y_c_t_1'][pred_s]]
                    else:
                        sys.exit('Wrong element type: '+ key)
                    norm_cost_values = [costs_normed[x][time_step-1, pred_s] for x in cost_names]
                    basic_features = [val.state_hist[1, pred_s], val.state_hist[0, pred_s],
                                      time_step/float(T), res/100.0]
                    row = np.array(basic_features+special_feature+norm_cost_values)
                    data = data.append(pd.Series(row, index=cols), ignore_index=True)
                ###Run models###
                with pm.Model() as logi_model:
                    pm.glm.GLM.from_formula(formula, data, family=pm.glm.families.Binomial())
                    from_formula_calls += 1
                    if val.model_status == 1:
                        trace = pm.load_trace(pred_dict['model_dir']+'/'+val.name)
                    elif val.model_status == 2:
                        trace = pm.load_trace(pred_dict['model_dir']+'/'+val.dupl_name)
                    ppc_test = pm.sample_posterior_predictive(trace, samples=no_pred_itr,
                                                              progressbar=False)
                    for idx in pred_idx:
                        sum_state = 0
                        for psr in range(no_pred_itr):
                            sum_state += ppc_test['y'][psr][pred_idx.index(idx)]
                        pred_decision[idx] = round(sum_state/float(no_pred_itr))
                        if pred_decision[idx] == 1:
                            if print_cmd:
                                print('Repaired')
                            res_unused[idx] -= 1
                pass
            elif val.model_status == 0:
                for pred_s in pred_idx:
                    if res_unused[pred_s] > 0:
                        pred_decision[pred_s] = np.random.randint(2)
                        if pred_decision[pred_s] == 1:
                            res_unused[pred_s] -= 1
                            if print_cmd:
                                print('Repaired',str(key),'P_sample:'+str(pred_s),'randomly')
                    else:
                        pred_decision[pred_s] = 0
            else:
                sys.exit('Wrong model status: '+key)
        val.state_hist[1, :] = pred_decision
    return from_formula_calls

def extract_features(objs, net_obj, t, layers, num_pred):
    """Extratcs predictors from data"""
    ### element feature extraction ###
    for pred_s in range(num_pred):
        for key, val in objs.items():
            if val.type == 'n':
                for n in val.neighbors:
                    val.w_n_t_1[t, pred_s] += objs[n].state_hist[t, pred_s]/val.degree
                for a in val.arcs:
                    try:
                        val.w_a_t_1[t, pred_s] += objs[a].state_hist[t, pred_s]/val.degree
                    except:
                        pass
                for n in val.dependees:
                    val.w_d_t_1[t, pred_s] += (1-objs[n].state_hist[t, pred_s])/val.num_dependee
            elif val.type == 'a':
                for n in val.end_nodes:
                    val.y_n_t_1[t, pred_s] += objs[n].state_hist[t, pred_s]/2.0
    ### layer feature extraction ###
    layer_dict = {x:{'w_c_t_1':np.zeros(num_pred),
                     'no_w_c':len([xx for xx in net_obj.G.nodes() if xx[1] == x]),
                     'y_c_t_1':np.zeros(num_pred),
                     'no_y_c':len([xx for xx in net_obj.G.edges() if xx[0][1] == x and xx[1][1] == x])//2,
                     'w_h_t_1':np.zeros(num_pred)} for x in layers}
    node_demand = {x:{} for x in layers}
    node_demand_highest = {}
    no_high_nodes = 5
    for n, d in net_obj.G.nodes(data=True):
        node_demand[n[1]]['w_'+str(n)] = abs(d['data']['inf_data'].demand)
    for l in list(node_demand.keys()):
        node_demand_highest[l] = dict(sorted(node_demand[l].items(), key=itemgetter(1),
                                             reverse=True)[:no_high_nodes])
    for pred_s in range(num_pred):
        for key, val in objs.items():
            if val.type == 'n':
                layer_dict[val.net_id]['w_c_t_1'][pred_s] += val.state_hist[t, pred_s]/layer_dict[val.net_id]['no_w_c']
            if val.type == 'a':
                layer_dict[val.net_id]['y_c_t_1'][pred_s] += val.state_hist[t, pred_s]/layer_dict[val.net_id]['no_y_c']
            if key in list(node_demand_highest[val.net_id].keys()):
                layer_dict[val.net_id]['w_h_t_1'][pred_s] += val.state_hist[t, pred_s]/no_high_nodes
    return layer_dict

def normalize_costs(costs, i):
    '''
    Data should be normailzied so that when using models such normalization is reproducible
    at each time step including the first one. Therefore, the normalization should be based
    on information that we have before starting the restoration

    Parameters
    ----------
    costs : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.

    Returns
    -------
    costs_normed : TYPE
        DESCRIPTION.

    '''
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

def make_formula(variables,exclusions,dependent):
    '''
    Parameters
    ----------
    variables : TYPE
        DESCRIPTION.
    exclusions : TYPE
        DESCRIPTION.
    dependent : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    predictors= [i for i in variables if i not in exclusions+dependent]
    formula=dependent[0]+'~'
    for i in predictors:
        formula+=i+'+'
    return formula[:-1]

def check_folder(output_dir):
    """Creates a new folder"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir
