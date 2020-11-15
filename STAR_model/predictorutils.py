"""Functions and classes to predict restoration scenarios"""
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
import infrastructure
import indpalt
import STAR_utils
import pickle

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
        param_file_dupl = param_folder+'/model_parameters_'+self.dupl_name+'.txt'
        if os.path.exists(param_file):
            self.model_status = 1
            self.model_params = pd.read_csv(param_file, delimiter=' ')
        elif os.path.exists(param_file_dupl):
                self.model_status = 2
                self.model_params = pd.read_csv(param_file_dupl, delimiter=' ')

def import_initial_data(params, fail_sce_param):
    """Imports initial data and initializes element objects"""
    print('Import Data...')
    # Set root directories and params
    base_dir = fail_sce_param['Base_dir']
    damage_dir = fail_sce_param['Damage_dir']
    sample = fail_sce_param['sample']
    mag = fail_sce_param['mag']
    topology = None
    shelby_data = None
    ext_interdependency = None
    if fail_sce_param['type']=='Andres':
        shelby_data = 'shelby_old'
        ext_interdependency = "../data/INDP_4-12-2016"
    elif fail_sce_param['type']=='WU':
        shelby_data = 'shelby_extended'
        if fail_sce_param['filtered_List']!=None:
            list_high_dam = pd.read_csv(fail_sce_param['filtered_List'])
    elif fail_sce_param['type']=='random':
        shelby_data = 'shelby_extended'
    elif fail_sce_param['type']=='synthetic':
        topology = fail_sce_param['topology']

    if not shelby_data:
        interdep_net, no_resource, layers = indp.initialize_network(BASE_DIR=base_dir,
                external_interdependency_dir=ext_interdependency, magnitude=mag,
                sample=sample, shelby_data=shelby_data, topology=topology)
        params["V"] = no_resource
        params["L"] = layers
    else:
        interdep_net, _, _ = indp.initialize_network(BASE_DIR=base_dir,
                external_interdependency_dir=ext_interdependency, sim_number=0,
                magnitude=6, sample=0, v=params["V"], shelby_data=shelby_data)
    params["N"] = interdep_net
    params["SIM_NUMBER"] = sample
    params["MAGNITUDE"] = mag
    # Damage the network
    if fail_sce_param['type'] == 'WU':
        infrastructure.add_Wu_failure_scenario(interdep_net, DAM_DIR=damage_dir, noSet=sample, noSce=mag)
    elif fail_sce_param['type'] == 'ANDRES':
        infrastructure.add_failure_scenario(interdep_net, DAM_DIR=damage_dir, magnitude=mag, v=params["V"],
                                  sim_number=sample)
    elif fail_sce_param['type'] == 'random':
        infrastructure.add_random_failure_scenario(interdep_net, DAM_DIR=damage_dir, sample=sample)
    elif fail_sce_param['type'] == 'synthetic':
        infrastructure.add_synthetic_failure_scenario(interdep_net, DAM_DIR=damage_dir, topology=topology,
                                            config=mag, sample=sample)
    # initialize objects
    objs = {}
    for v in interdep_net.G.nodes():
        objs['w_'+str(v)] = NodeModel('w_'+str(v), v[1])
        objs['w_'+str(v)].initial_state = interdep_net.G.nodes[v]['data']['inf_data'].functionality
    for u, v, a in interdep_net.G.edges(data=True):
        if not a['data']['inf_data'].is_interdep:
            if 'y_'+str(v)+','+str(u) not in list(objs.keys()):
                objs['y_'+str(u)+','+str(v)] = ArcModel('y_'+str(u)+','+str(v), v[1])
                objs['y_'+str(u)+','+str(v)].initial_state = interdep_net.G[u][v]['data']['inf_data'].functionality
                objs['w_'+str(u)].add_neighbor('w_'+str(v))
                objs['w_'+str(v)].add_neighbor('w_'+str(u))
        else:
            objs['w_'+str(v)].add_dependee('w_'+str(u))
    return objs

def predict_resotration(pred_dict, fail_sce_param, params, real_rep_sequence):
    """ Predicts restoration plans and writes to file"""
    print('\nMagnitude '+str(fail_sce_param['mag'])+' sample '+str(fail_sce_param['sample'])+\
          ' Rc '+str(params['V']))
    ### Define a few vars and lists ###
    num_pred = pred_dict['num_pred']
    T = params['NUM_ITERATIONS']
    pred_results = {x:indputils.INDPResults(params['L']) for x in range(num_pred)}
    run_times = {x:[0, 0] for x in range(T+1)}
    ### Initialize element and network objects ###
    objs = import_initial_data(params, fail_sce_param)
    net_obj = {x:copy.deepcopy(params['N']) for x in range(num_pred)}
    for key, val in objs.items():
        val.initialize_state_matrices(T, num_pred)
        val.check_model_exist(pred_dict['param_folder'])
    ### Initialize cost vectors ###
    indp_results = indp.indp(params['N'], v_r=0, T=1, layers=params['L'],
                             controlled_layers=params['L'], print_cmd=False,
                             time_limit=None)
    run_times[0][0] = indp_results[1][0]['run_time']
    for pred_s in range(num_pred):
        pred_results[pred_s].extend(indp_results[1], t_offset=0)
    costs = {x:{} for x in params['L']+[0]}
    for h in list(indp_results[1][0]['costs'].keys()):
        costs[0][h.replace(' ', '_')] = np.zeros((T+1, num_pred))
        costs[0][h.replace(' ', '_')][0, :] = indp_results[1].results[0]['costs'][h]
        for l in params['L']:
            costs[l][h.replace(' ', '_')] = np.zeros((T+1, num_pred))
            costs[l][h.replace(' ', '_')][0, :] = indp_results[1].results_layer[l][0]['costs'][h]
    ###Predict restoration plans###
    print('Predicting, time step:')
    for t in range(T): # t is the time index for previous time step
        print(str(t+1)+'.', end=" ")
        start_time = time.time()
        ### Feature extraction###
        layer_dict = extract_features(objs, net_obj, t, params['L'], num_pred)
        ###  Cost normalization ###
        costs_normed = {}
        for c in list(costs[0].keys()):
            costs_normed[c] = STAR_utils.normalize_costs(costs[0][c], c)
        run_times[t+1][1] = predict_next_step(t, T, objs, pred_dict, costs_normed,
                                              params['V'], layer_dict, print_cmd=True)
        run_times[t+1][0] = time.time()-start_time
        ### Calculate the cost of scenario ###
        for pred_s in range(num_pred):
            start_time = time.time()
            decision_vars = {0:{}} #0 becasue iINDP
            for key, val in objs.items():
                decision_vars[0][val.name] = val.state_hist[t+1, pred_s]
                if val.type == 'a':
                    decision_vars[0][val.dupl_name] = val.state_hist[t+1, pred_s]
            flow_results = indpalt.flow_problem(net_obj[pred_s], v_r=0, layers=params['L'],
                                                controlled_layers=params['L'],
                                                decision_vars=decision_vars,
                                                print_cmd=True, time_limit=None)
            pred_results[pred_s].extend(flow_results[1], t_offset=t+1)
            indp.apply_recovery(net_obj[pred_s], flow_results[1], 0)

            ### Update the cost dict for the next time step and run times ###
            equiv_run_time = 0
            if run_times[t+1][1] != 0:
                equiv_run_time = run_times[t+1][0]/run_times[t+1][1]+(time.time()-start_time)
            for h in list(costs[0].keys()):
                costs[0][h][t+1, pred_s] = flow_results[1][0]['costs'][h.replace('_', ' ')]
                pred_results[pred_s].results[t+1]['run_time'] = equiv_run_time
                for l in params['L']:
                    costs[l][h][t+1, pred_s] = flow_results[1].results_layer[l][0]['costs'][h.replace('_', ' ')]
                    pred_results[pred_s].results_layer[l][t+1]['run_time'] = equiv_run_time
            ###Write results to file###
            subfolder_results = 'stm_results_L'+str(len(params['L']))+'_m'+str(fail_sce_param['mag'])+'_v'+str(params['V'])
            output_dir = check_folder(pred_dict['output_dir']+'/'+subfolder_results)
            pred_results[pred_s].to_csv(output_dir, fail_sce_param["sample"], suffix='ps'+str(pred_s))
            output_dir_l = check_folder(pred_dict['output_dir']+'/'+subfolder_results+'/agents')
            pred_results[pred_s].to_csv_layer(output_dir_l, fail_sce_param["sample"], suffix='ps'+str(pred_s))
            ####Write models to file###
            # indp.save_INDP_model_to_file(flow_results[0], './models',t, l = 0)

    # Extract prediction results
    pred_error = pd.DataFrame(columns=['name','pred sample', 'pred rep time',
                                       'real rep time', 'prediction error'])
    rep_prec = pd.DataFrame(columns=['t','pred sample', 'pred rep prec','real rep prec'])
    sum_pred_rep_prec=np.zeros((T+1,num_pred))
    sum_real_rep_prec=np.zeros((T+1))
    for key, val in objs.items():
        sum_real_rep_prec += real_rep_sequence[key]
        for pred_s in range(num_pred):
            if real_rep_sequence[key][0] != val.state_hist[0,pred_s]:
                sys.exit('Error: Unmatched intial state')
            temp_dict = {'name':key, 'pred sample':pred_s,
                         'real rep time':T+1-sum(real_rep_sequence[key]),
                         'pred rep time':T+1-sum(val.state_hist[:,pred_s])}
            temp_dict['prediction error'] = temp_dict['real rep time'] - temp_dict['pred rep time']
            pred_error = pred_error.append(temp_dict, ignore_index=True)
            sum_pred_rep_prec[:,pred_s] += val.state_hist[:,pred_s]
    for pred_s in range(num_pred):
        for t in range(T+1):
            no_elements = len(objs.keys())
            temp_dict = {'t':t, 'pred sample':pred_s,
                         'pred rep prec':sum_pred_rep_prec[t,pred_s]/no_elements,
                         'real rep prec':sum_real_rep_prec[t]/no_elements}
            rep_prec = rep_prec.append(temp_dict, ignore_index=True)
    return pred_results, pred_error, rep_prec

def predict_next_step(t, T, objs, pred_dict, costs_normed, res, layer_dict, print_cmd=True):
    """define vars and lists"""
    cost_names = list(costs_normed.keys())
    num_pred = pred_dict['num_pred']
    no_pred_itr = 50 #Number of samples to compute each prediction sample
    node_cols = ['w_t', 'w_t_1', 'time', 'Rc', 'w_n_t_1', 'w_a_t_1', 'w_d_t_1',
                 'w_h_t_1', 'w_c_t_1', 'y_c_t_1']
    arc_cols = ['y_t', 'y_t_1', 'time', 'Rc', 'y_n_t_1', 'w_c_t_1', 'y_c_t_1',
                'w_c_t']
    from_formula_calls = 0
    extract_current_step = False
    for key, val in objs.items():
        if val.type == 'n':
            l = val.net_id
            ###initialize new predictions###
            non_pred_idx = []
            pred_decision = np.zeros(num_pred)
            for pred_s in range(num_pred):
                if val.state_hist[t, pred_s] == 1:
                    pred_decision[pred_s] = 1
                    non_pred_idx.append(pred_s)
            pred_idx = [x for x in range(num_pred) if x not in non_pred_idx]
            ###if any pred_s needs prediction###
            if len(pred_idx):
                if val.model_status:
                    ###Making model formula###
                    if print_cmd:
                        print('Predicting: '+key+'...')
                    varibales = list(val.model_params['Unnamed: 0'])
                    varibales.remove('Intercept')
                    formula = STAR_utils.make_formula(varibales, [], [key[0]+'_t'])
                    ###Make input datframe###
                    data = pd.DataFrame()
                    for pred_s in pred_idx:
                        cols = node_cols+cost_names
                        special_feature = [val.w_n_t_1[t, pred_s],
                                           val.w_a_t_1[t, pred_s],
                                           val.w_d_t_1[t, pred_s],
                                           layer_dict[l]['w_h_t_1'][pred_s],
                                           layer_dict[l]['w_c_t_1'][pred_s],
                                           layer_dict[l]['y_c_t_1'][pred_s]]
                        norm_cost_values = [costs_normed[x][t, pred_s] for x in cost_names]
                        basic_features = [val.state_hist[t+1, pred_s], val.state_hist[t, pred_s],
                                          (t+1)/float(T), res/100.0]
                        row = np.array(basic_features+special_feature+norm_cost_values)
                        data = data.append(pd.Series(row, index=cols), ignore_index=True)
                    ###Run models###
                    with pm.Model() as logi_model:
                        pm.glm.GLM.from_formula(formula, data, family=pm.glm.families.Binomial())
                        from_formula_calls += 1
                        trace = pm.load_trace(pred_dict['model_dir']+'/'+val.name)
                        ppc_test = pm.sample_posterior_predictive(trace, samples=no_pred_itr,
                                                                  progressbar=False)
                        for idx in pred_idx:
                            sum_state = 0
                            for psr in range(no_pred_itr):
                                sum_state += ppc_test['y'][psr][pred_idx.index(idx)]
                            pred_decision[idx] = round(sum_state/float(no_pred_itr))
                elif val.model_status == 0:
                    if print_cmd:
                        print('Predicting '+str(key)+', randomly')
                    for pred_s in pred_idx:
                        pred_decision[pred_s] = np.random.randint(2)
                else:
                    sys.exit('Wrong model status: '+key)
                val.state_hist[t+1, :] = pred_decision 
    for key, val in objs.items():
        if val.type == 'a':
            l = val.net_id
            ###initialize new predictions###
            non_pred_idx = []
            pred_decision = np.zeros(num_pred)
            for pred_s in range(num_pred):
                if val.state_hist[t, pred_s] == 1:
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
                    formula = STAR_utils.make_formula(varibales, [], [key[0]+'_t'])
                    ###Make input datframe###
                    data = pd.DataFrame()
                    if not extract_current_step:
                        layer_dict = extract_features_current(objs, t+1, num_pred, layer_dict)
                        extract_current_step = True
                    for pred_s in pred_idx:
                        cols = arc_cols+cost_names
                        special_feature = [val.y_n_t_1[t, pred_s],
                                           layer_dict[l]['w_c_t_1'][pred_s],
                                           layer_dict[l]['y_c_t_1'][pred_s],
                                           layer_dict[l]['w_c_t'][pred_s]]
                        norm_cost_values = [costs_normed[x][t, pred_s] for x in cost_names]
                        basic_features = [val.state_hist[t+1, pred_s], val.state_hist[t, pred_s],
                                          (t+1)/float(T), res/100.0]
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
                elif val.model_status == 0:
                    if print_cmd:
                        print('Predicting '+str(key)+', randomly')
                    for pred_s in pred_idx:
                        pred_decision[pred_s] = np.random.randint(2)
                else:
                    sys.exit('Wrong model status: '+key)
                val.state_hist[t+1, :] = pred_decision            
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
                    val.w_d_t_1[t, pred_s] += objs[n].state_hist[t, pred_s]/val.num_dependee
            elif val.type == 'a':
                for n in val.end_nodes:
                    val.y_n_t_1[t, pred_s] += objs[n].state_hist[t, pred_s]/2.0
    ### layer feature extraction ###
    layer_dict = {x:{'w_c_t_1':np.zeros(num_pred),
                     'no_w_c':len([xx for xx in net_obj[0].G.nodes() if xx[1] == x]),
                     'y_c_t_1':np.zeros(num_pred),
                     'no_y_c':len([xx for xx in net_obj[0].G.edges() if xx[0][1] == x and xx[1][1] == x])//2,
                     'w_h_t_1':np.zeros(num_pred)} for x in layers}
    node_demand = {x:{} for x in layers}
    node_demand_highest = {}
    no_high_nodes = 5
    for n, d in net_obj[pred_s].G.nodes(data=True):
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

def extract_features_current(objs, t, num_pred, layer_dict):
    """Extratcs predictors from data for the current step"""
    ### layer feature extraction ###
    for l in layer_dict.keys():
        layer_dict[l]['w_c_t'] = np.zeros(num_pred)
    for pred_s in range(num_pred):
        for key, val in objs.items():
            if val.type == 'n':
                layer_dict[val.net_id]['w_c_t'][pred_s] += val.state_hist[t, pred_s]/layer_dict[val.net_id]['no_w_c']
    return layer_dict

def check_folder(output_dir):
    """Creates a new folder"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir