import numpy as np
import pandas as pd
from indp import *
import os
import sys
import string
import pymc3 as pm
import sklearn.metrics
import time
import pickle
from infrastructure import *
from operator import itemgetter
import seaborn as sns
import copy
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
print('Running with PyMC3 version v.{}'.format(pm.__version__))
import networkx as nx
print('Running with Networkx version v.{}'.format(nx.__version__))
import seaborn as sns
print('Running with Networkx version v.{}'.format(sns.__version__))

def importData(params, failSce_param, suffix='', print_cmd=True):
    if print_cmd:
        print('\nNumber of resources: '+str(params["V"]))
    # Set root directories
    layers = params['L']
    base_dir = failSce_param['Base_dir']
    damage_dir = failSce_param['Damage_dir']
    topology = None
    shelby_data = None
    ext_interdependency = None
    if failSce_param['type']=='Andres':
        shelby_data = 'shelby_old'
        ext_interdependency = "../data/INDP_4-12-2016"
    elif failSce_param['type']=='WU':
        shelby_data = 'shelby_extended'
        if failSce_param['filtered_List']!=None:
            list_high_dam = pd.read_csv(failSce_param['filtered_List'])
    elif failSce_param['type']=='random':
        shelby_data = 'shelby_extended'
    elif failSce_param['type']=='synthetic':
        topology = failSce_param['topology']

    samples={}
    costs = {}
    initial_net = {}

    folder_name = 'data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for m in failSce_param['mags']:
        for i in failSce_param['sample_range']:
            # Check samples that do not exist
            results_dir=params['OUTPUT_DIR']+'_L'+str(len(layers))+'_m'+str(m)+'_v'+str(params['V'])
            action_file=results_dir+"/actions_"+str(i)+"_"+suffix+".csv"
            cost_file=results_dir+"/costs_"+str(i)+"_"+suffix+".csv"
            if not os.path.isfile(action_file) or not os.path.isfile(cost_file):
                with open(folder_name+'/missing_scenarios.txt', 'a') as filehandle:
                    filehandle.write(str(params["V"])+'\t'+str(m)+'\t'+str(i)+'\n')
                    filehandle.close()
                continue
            # Check samples that are excluded
            try:
                list_high_dam
                if len(list_high_dam.loc[(list_high_dam.set == i)&\
                                         (list_high_dam.sce == m)].index) == 0:
                    with open(folder_name+'/missing_scenarios.txt', 'a') as filehandle:
                        filehandle.write(str(params["V"])+'\t'+str(m)+'\t'+str(i)+'\n')
                        filehandle.close()
                    continue

            except NameError:
                pass

            if (i-failSce_param['sample_range'][0]+1)%2==0 and print_cmd:
                update_progress(i-failSce_param['sample_range'][0]+1,len(failSce_param['sample_range']))

            if shelby_data:
                params["N"], _, _ = initialize_network(BASE_DIR=base_dir,
                                                       external_interdependency_dir=ext_interdependency,
                                                       sim_number=0, magnitude=6,
                                                       sample=0, v=params["V"],
                                                       shelby_data=shelby_data)
            else:
                params["N"], params["V"], params["L"] = initialize_network(BASE_DIR=base_dir,
                                                                           external_interdependency_dir=ext_interdependency,
                                                                           magnitude=m,
                                                                           sample=i,
                                                                           shelby_data=shelby_data,
                                                                           topology=topology)
            params["SIM_NUMBER"]=i
            params["MAGNITUDE"]=m
            if not initial_net.keys():
                initial_net[0] = copy.deepcopy(params["N"])

            if failSce_param['type']=='WU':
                add_Wu_failure_scenario(params["N"], DAM_DIR=damage_dir, noSet=i, noSce=m,
                                        no_arc_damage=True)
            elif failSce_param['type']=='ANDRES':
                add_failure_scenario(params["N"], DAM_DIR=damage_dir, magnitude=m,
                                     v=params["V"], sim_number=i)
            elif failSce_param['type']=='random':
                add_random_failure_scenario(params["N"], DAM_DIR=damage_dir, sample=i,
                                            no_arc_damage=False)
            elif failSce_param['type']=='synthetic':
                add_synthetic_failure_scenario(params["N"], DAM_DIR=damage_dir,
                                               topology=topology, config=m, sample=i)

            samples = initialize_matrix(params["N"], samples, m, i, 10)
            samples,costs = read_restoration_plans(samples, costs, params["N"], m, i,
                                                   results_dir, suffix='')
    if print_cmd:
        update_progress(i-failSce_param['sample_range'][0]+1,len(failSce_param['sample_range']))
    return samples, costs, initial_net, params["V"], params["L"]

def initialize_matrix(N, sample, m, i, time_steps):
    for v in N.G.nodes():
        name = 'w_'+str(v)
        if name not in sample.keys():
            sample[name]= np.ones((time_steps+1,1))
        else:
            sample[name] = np.append(sample[name], np.ones((time_steps+1,1)), axis=1)
        if N.G.nodes[v]['data']['inf_data'].functionality==0.0:
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

def read_restoration_plans(sample, costs, InterdepNet, m, i, results_dir, suffix=''):
    # Read elemnts' states
    action_file=results_dir+"/actions_"+str(i)+"_"+suffix+".csv"
    if os.path.isfile(action_file):
        with open(action_file) as f:
            lines=f.readlines()[1:]
            for line in lines:
                data = str.strip(line).split(",")
                t = int(data[0])
                action = str.strip(data[1])
                k = int(action[-1])
                if '/' in action:
                    act_data = str.strip(action).split("/")
                    u = str.strip(act_data[0]).split(".")
                    v = str.strip(act_data[1]).split(".")
                    if u[1]!=v[1]:
                        sys.exit('Interdepndency '+act_data+' is counted as an arc')
                    arc_id = str((int(u[0]),k))+','+str((int(v[0]),k))
                    sample['y_'+arc_id][t:,-1] = 1.0
                else:
                    act_data = str.strip(action).split(".")
                    node_id = int(act_data[0])
                    sample['w_'+str((node_id,k))][t:,-1] = 1.0
    else:
        sys.exit('No results dir: '+action_file)

    # Read cost for the entire interdepndent network
    cost_file=results_dir+"/costs_"+str(i)+"_"+suffix+".csv"
    T=sample[list(sample.keys())[0]].shape[0]
    if os.path.isfile(cost_file):
        with open(cost_file) as f:
            lines=f.readlines()
            for line in lines:
                if line[0]=='t':
                    headers=str.strip(line).split(",")
                    headers = [c.replace(' ', '_') for c in headers]
                    for h in headers[1:]:
                        if h not in costs.keys():
                            costs[h]= np.zeros((T,1))
                        else:
                            costs[h] = np.append(costs[h], np.zeros((T,1)), axis=1)
                    continue
                data=str.strip(line).split(",")
                t=int(data[0])
                for ii in data[1:]:
                    cost_name = headers[data.index(ii)]
                    costs[cost_name][t,-1]=float(ii)
    else:
        sys.exit('No results dir: '+cost_file)

    return sample,costs

def prepare_data(samples, costs, initial_net, res, keys):
    print('Preparing data:')
    names = list(samples.keys())
    noSamples = int(samples[names[0]].shape[1])
    T = int(samples[names[0]].shape[0])

    w_n_t_1,w_d_t_1,w_a_t_1,w_h_t_1,w_c_t_1,y_n_t_1,y_c_t_1,w_c_t=extract_features(samples,
                                                                                   initial_net, 
                                                                                   keys)

    print('\nBuilding Dataframes:')
    cost_names=list(costs.keys())
    costs_normed={}
    for i in cost_names:
        costs_normed[i] = normalize_costs(costs[i], i)

    node_cols=['w_t','w_t_1','sample','time','Rc','w_n_t_1','w_a_t_1','w_d_t_1','w_h_t_1','w_c_t_1','y_c_t_1']
    arc_cols=['y_t','y_t_1','sample','time','Rc','y_n_t_1','w_c_t_1','y_c_t_1','w_c_t']
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
                        special_feature=[y_n_t_1[key][t,s],w_c_t_1[l][t,s],
                                         y_c_t_1[l][t,s],w_c_t[l][t,s]]
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
    keys = list(samples.keys())
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
    w_c_t={}  # all nodes
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
    for u,v,a in selected_g.edges(data=True):
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
    for n,d in selected_g.nodes(data=True):
        if n[1] not in node_demand.keys():
            node_demand[n[1]]={}
            w_c_t_1[n[1]] = np.zeros((T-1,noSamples))
            w_c_t[n[1]] = np.zeros((T-1,noSamples))
            no_w_c[n[1]]=0
        node_demand[n[1]]['w_'+str(n)]=abs(d['data']['inf_data'].demand)
        w_c_t_1[n[1]]+=w_t_1['w_'+str(n)]
        w_c_t[n[1]]+=w_t['w_'+str(n)]
        no_w_c[n[1]]+=1

    for u,v,a in selected_g.edges(data=True):
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

        w_c_t[nn]/=no_w_c[nn]
        w_c_t_1[nn]/=no_w_c[nn]
        y_c_t_1[nn]/=no_y_c[nn]
    if prog_bar:
        update_progress(6.0,7.0)

    for u,v,a in selected_g.edges(data=True):
        if not a['data']['inf_data'].is_interdep:
            key = 'y_'+str(u)+','+str(v)
            if key in arc_keys:
                y_n_t_1[key] = np.zeros((T-1,noSamples))
                y_n_t_1[key]+=w_t_1['w_'+str(v)]/2.0
                y_n_t_1[key]+=w_t_1['w_'+str(u)]/2.0
    if prog_bar:
        update_progress(7.0,7.0)

    return w_n_t_1,w_d_t_1,w_a_t_1,w_h_t_1,w_c_t_1,y_n_t_1,y_c_t_1,w_c_t

def train_model(train_data,exclusions):
    print('\nTraining models:')
    logistic_model={}
    trace={}
    folder_name = 'parameters'
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
            trace[key] = pm.sample(750, tune=500,chains=2, cores=2,init='adapt_diag')
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
    folder_name = 'parameters'
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
    folder_name = 'data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    pickle.dump([samples,costs,initial_net], open(folder_name+'/initial_data.pkl', "wb" ),
                protocol=pickle.HIGHEST_PROTOCOL)

def save_prepared_data(train_data,test_data):
    folder_name = 'data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    pickle.dump([train_data,test_data], open(folder_name+'/train_test_data.pkl', "wb" ),
                protocol=pickle.HIGHEST_PROTOCOL)

def save_traces(trace):
    fname={}
    folder_name = 'traces'
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
    folder_name = './results'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open('./results'+'/'+filename+'.txt', 'a') as filehandle:
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
        sys.exit('No missing scenario file: '+mis_sce_file)

def update_progress(progress,total):
    print('\r[%s] %1.1f%%' % ('#'*int(progress/float(total)*20), (progress/float(total)*100)),
          end='')
    sys.stdout.flush()