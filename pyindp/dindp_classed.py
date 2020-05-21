''' Decentralized restoration for interdepndent networks'''
import os.path
import operator
import copy
import itertools
import time
import sys
import pandas as pd
import numpy as np
import gurobipy
import indp
import indputils
from dindp_classes import *

def run_judgment_call(params, T=1, save_jc=True, print_cmd=True, save_jc_model=False):
    '''
    Solves an INDP problem with specified parameters using a decentralized hueristic.
    Judgment Call

    Parameters
    ----------
    params : dict
         Global parameters.
    T : int, optional
         Number of time steps per analyses (1 for D-iINDP and T>1 for D-tdINDP). The default is 1.
    save_jc : bool, optional
        If true, the results are saved to files. The default is True.
    print_cmd : bool, optional
        If true, the results are printed to console. The default is True.
    save_jc_model : bool, optional
        If true, optimization models and their solutions are printed to file. The default is False.

    Returns
    -------
    None :

    '''
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"] = 1
    num_iterations = params["NUM_ITERATIONS"]
    time_limit = 10*60 #!!! Might be adjusted later
    # Creating JC objects
    c = 0
    objs = {}
    params_copy = copy.deepcopy(params)  #!!! deepcopy
    for jc in params["JUDGMENT_TYPE"]:
        params_copy['JUDGMENT_TYPE'] = jc
        for rst in params["RES_ALLOC_TYPE"]:
            params_copy['RES_ALLOC_TYPE'] = rst
            if rst not in ["MDA", "MAA", "MCA"]:
                objs[c] = JcModel(c,params_copy)
                c += 1
            else:
                for vt in params["VALUATION_TYPE"]:
                    params_copy['VALUATION_TYPE'] = vt
                    objs[c] = JcModel(c,params_copy)
                    c += 1
    # t=0 costs and performance.
    indp_results_initial = indp.indp(objs[0].net, 0, 1, objs[0].layers,
                                     controlled_layers=objs[0].layers)
    for idx, obj in objs.items():
        print('--Running JC: '+obj.judge_type+', resource allocation: '+obj.res_alloc_type)
        if obj.resource.type == 'AUCTION':
            print('auction type: '+obj.resource.auction_model.auction_type+\
                  ', valuation: '+obj.resource.auction_model.valuation_type)
        if print_cmd:
            print("Num iters=", params["NUM_ITERATIONS"])
        # t=0 results.
        obj.results_judge = copy.deepcopy(indp_results_initial[1]) #!!! deepcopy
        obj.results_real = copy.deepcopy(indp_results_initial[1]) #!!! deepcopy
        temp_judge = obj.results_judge
        temp_real = obj.results_real
        for i in range(num_iterations):
            print("-Time Step (JC)",i+1,"/",num_iterations)
            #: Resource Allocation
            res_alloc_time_start = time.time()
            if obj.resource.type == 'AUCTION':
                obj.resource.auction_model.auction_resources(obj, i+1, print_cmd=print_cmd,
                                                             compute_poa=False)
            obj.resource.time[i+1] = time.time()-res_alloc_time_start
            # Judgment-based Decisions
            if print_cmd:
                print("Judge-based decisions: ")
            functionality = {l:{} for l in obj.layers}
            for l in obj.layers:
                if print_cmd:
                    print("Layer-%d"%(l))
                neg_layer = [x for x in obj.layers if x != l]
                functionality[l] = obj.judgments.create_judgment_dict(obj, neg_layer)
                obj.judgments.save_judgments(obj, functionality[l], l, i+1)
                # Make decision based on judgments before communication
                indp_results = indp.indp(obj.net, obj.v_r[i+1][l], 1, layers=obj.layers,
                                controlled_layers=[l], functionality=functionality[l],
                                print_cmd=print_cmd, time_limit=time_limit)
                obj.results_judge.extend(indp_results[1],t_offset=i+1)
                # Save models to file
                if save_jc_model:
                    indp.save_INDP_model_to_file(indp_results[0], output_dir+"/Model", i+1, l)
                # Modify network to account for recovery and calculate components.
                indp.apply_recovery(obj.net, obj.results_judge, i+1)
                obj.results_judge.add_components(i+1, indputils.INDPComponents.\
                                           calculate_components(indp_results[0],
                                                                obj.net, layers=[l]))
            # Re-evaluate judgments based on other agents' decisions
            if print_cmd:
                print("Re-evaluation: ")
            for l in obj.layers:
                if print_cmd:
                    print("Layer-%d"%(l))
                indp_results_real = realized_performance(obj, i+1, functionality=functionality[l],
                                                         judger_layer=l, print_cmd=print_cmd)
                obj.results_real.extend(indp_results_real[1],t_offset=i+1)
                obj.correct_results_real(l, i+1)
                if save_jc_model:
                    indp.save_INDP_model_to_file(indp_results_real[0], output_dir+"/Model",
                                                  i+1, l, suffix='real')
            #Calculate sum of costs
            obj.recal_result_sum(i+1)
        # Save results of D-iINDP run to file.
        if save_jc:
            obj.save_results_to_file(params["SIM_NUMBER"])
            obj.save_object_to_file(params["SIM_NUMBER"])

def realized_performance(obj, t_step, functionality, judger_layer, print_cmd=False):
    '''
    This function computes the realized values of flow cost, unbalanced cost, and
    demand deficit at the end of each step according to the what the other agent
    actually decides (as opposed to according to the judgment-based decisions).

    Parameters
    ----------
    N : TYPE
        DESCRIPTION.
    indp_results : TYPE
        DESCRIPTION.
    functionality : TYPE
        DESCRIPTION.
    layers : TYPE
        DESCRIPTION.
    T : TYPE, optional
        DESCRIPTION. The default is 1.
    judger_layer : TYPE, optional
        DESCRIPTION. The default is [1].
    print_cmd : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    indp_results_real : TYPE
        DESCRIPTION.
    realizations : TYPE
        DESCRIPTION.

    '''
    time_limit = 10*60 #!!! Might be adjusted later
    functionality_realized = copy.deepcopy(functionality) #!!! deepcopy
    dest_nodes = obj.judgments.dest_nodes[t_step][judger_layer]
    for v, val in dest_nodes.items():
        val.append(obj.net.G.nodes[v]['data']['inf_data'].functionality)
    judged_nodes = obj.judgments.judged_nodes[t_step][judger_layer]
    for u, val in judged_nodes.items():
        if functionality[0][u] == 1.0 and obj.net.G.nodes[u]['data']['inf_data'].functionality == 0.0:
            functionality_realized[0][u] = 0.0
            if print_cmd:
                print('Correct judgment: '+str(u)+'<-0')
        val.append(obj.net.G.nodes[u]['data']['inf_data'].functionality)
    indp_results_real = indp.indp(obj.net, v_r=0, T=1, layers=obj.layers,
                                  controlled_layers=[judger_layer],
                                  functionality=functionality_realized,
                                  print_cmd=print_cmd, time_limit=time_limit)
    for v, val in dest_nodes.items():
        nodeVar='w_'+str(v)+","+str(0)
        val.append(indp_results_real[0].getVarByName(nodeVar).x)
    return indp_results_real


def read_resourcec_allocation(result_df, combinations, optimal_combinations, ref_method='indp',
                              suffix="", root_result_dir='../results/'):
    '''

    Parameters
    ----------
    result_df : TYPE
        DESCRIPTION.
    combinations : TYPE
        DESCRIPTION.
    optimal_combinations : TYPE
        DESCRIPTION.
    ref_method : TYPE, optional
        DESCRIPTION. The default is 'indp'.
    suffix : TYPE, optional
        DESCRIPTION. The default is "".
    root_result_dir : TYPE, optional
        DESCRIPTION. The default is '../results/'.

    Returns
    -------
    df_res : TYPE
        DESCRIPTION.
    df_res_rel : TYPE
        DESCRIPTION.

    '''
    cols = ['t', 'resource', 'decision_type', 'auction_type', 'valuation_type', 'sample',
            'Magnitude', 'layer', 'no_resources', 'normalized_resources', 'poa']
    T = max(result_df.t.unique().tolist())
    df_res = pd.DataFrame(columns=cols, dtype=int)
    print('\nResource allocation\n', end='')
    for idx, x in enumerate(optimal_combinations):
        compare_to_dir = root_result_dir+x[4]+'_results_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])
        for t in range(T):
            for p in range(1, x[2]+1):
                df_res = df_res.append({'t':t+1, 'resource':0.0, 'normalized_resource':0.0,
                    'decision_type':x[4], 'auction_type':'', 'valuation_type':'', 'sample':x[1],
                    'Magnitude':x[0], 'layer':p, 'no_resources':x[3], 'poa':1}, ignore_index=True)
        # Read optimal resource allocation based on the actions
        action_file = compare_to_dir+"/actions_"+str(x[1])+"_"+suffix+".csv"
        if os.path.isfile(action_file):
            with open(action_file) as f:
                lines = f.readlines()[1:]
                for line in lines:
                    data = line.strip().split(',')
                    t = int(data[0])
                    action = str.strip(data[1])
                    p = int(action[-1])
                    if '/' in action:
                        addition = 0.5
                    else:
                        addition = 1.0
                    row = (df_res['t'] == t)&(df_res['decision_type'] == x[4])&\
                    (df_res['sample'] == x[1])&(df_res['Magnitude'] == x[0])&\
                    (df_res['layer'] == p)&(df_res['no_resources'] == x[3])
                    df_res.loc[row, 'resource'] += addition
                    df_res.loc[row, 'normalized_resource'] += addition/float(x[3])
        if idx%(len(combinations+optimal_combinations)/10+1) == 0:
            update_progress(idx+1, len(optimal_combinations)+len(combinations))
    # Read  resource allocation based on auction results
    for idx, x in enumerate(combinations):
        if x[5] in ['Uniform']:
            outdir = root_result_dir+x[4]+'_results_L'+str(x[2])+'_m'+str(x[0])+\
                '_v'+str(x[3])+'_uniform_alloc/auctions'
        else:
            outdir = root_result_dir+x[4]+'_results_L'+str(x[2])+'_m'+str(x[0])+\
                '_v'+str(x[3])+'_auction_'+x[5]+'_'+x[6]+'/auctions'
        auction_file = outdir+"/auctions_"+str(x[1])+"_"+suffix+".csv"
        if os.path.isfile(auction_file):
            with open(auction_file) as f:
                lines = f.readlines()[1:]
                for line in lines:
                    data = line.strip().split(',')
                    t = int(data[0])
                    for p in range(1, x[2]+1):
                        if x[5] in ['Uniform']:
                            poa = 0.0
                        else:
                            poa = float(data[x[2]+1])
                        df_res = df_res.append({'t':t, 'resource':float(data[p]),
                                                'normalized_resource':float(data[p])/float(x[3]),
                                                'decision_type':x[4], 'auction_type':x[5],
                                                'valuation_type':x[6], 'sample':x[1],
                                                'Magnitude':x[0], 'layer':p, 'no_resources':x[3],
                                                'poa':poa}, ignore_index=True)
        if idx%(len(combinations+optimal_combinations)/10+1) == 0:
            update_progress(len(optimal_combinations)+idx+1,
                            len(optimal_combinations)+len(combinations))
    update_progress(len(optimal_combinations)+idx+1, len(optimal_combinations)+len(combinations))
    cols = ['decision_type', 'auction_type', 'valuation_type', 'sample', 'Magnitude', 'layer',
            'no_resources', 'distance_to_optimal', 'norm_distance_to_optimal']
    T = max(result_df.t.unique().tolist())
    df_res_rel = pd.DataFrame(columns=cols, dtype=int)
    print('\nRelative allocation\n', end='')
    for idx, x in enumerate(combinations+optimal_combinations):
        # Construct vector of resource allocation of reference method
        if x[4] != ref_method:
            vector_res_ref = {p:np.zeros(T) for p in range(1, x[2]+1)}
            for p in range(1, x[2]+1):
                for t in range(T):
                    vector_res_ref[p][t] = df_res.loc[(df_res['t'] == t+1)&
                                                      (df_res['decision_type'] == ref_method)&
                                                      (df_res['sample'] == x[1])&
                                                      (df_res['Magnitude'] == x[0])&
                                                      (df_res['layer'] == p)&
                                                      (df_res['no_resources'] == x[3]), 'resource']
            # Compute distance of resource allocation vectors
            vector_res = {p:np.zeros(T) for p in range(1, x[2]+1)}
            for p in range(1, x[2]+1):
                row = (df_res['decision_type'] == x[4])&(df_res['sample'] == x[1])&\
                (df_res['Magnitude'] == x[0])&(df_res['layer'] == p)&\
                (df_res['no_resources'] == x[3])&(df_res['auction_type'] == x[5])&\
                (df_res['valuation_type'] == x[6])
                for t in range(T):
                    vector_res[p][t] = df_res.loc[(df_res['t'] == t+1)&row, 'resource']
                #L2 norm
                distance = np.linalg.norm(vector_res[p]-vector_res_ref[p])
                norm_distance = np.linalg.norm(vector_res[p]/float(x[3])-\
                                               vector_res_ref[p]/float(x[3]))
                # #L1 norm
                # distance = sum(abs(vector_res[p]-vector_res_ref[p]))
                # # correlation distance
                # distance = 1-scipy.stats.pearsonr(vector_res[p], vector_res_ref[p])[0]
                df_res_rel = df_res_rel.append({'decision_type':x[4], 'auction_type':x[5],
                                                'valuation_type':x[6], 'sample':x[1],
                                                'Magnitude':x[0], 'layer':p, 'no_resources':x[3],
                                                'distance_to_optimal':distance/float(vector_res[p].shape[0]),
                                                'norm_distance_to_optimal':norm_distance/float(vector_res[p].shape[0])},
                                               ignore_index=True)
            if idx%(len(combinations+optimal_combinations)/10+1) == 0:
                update_progress(idx+1, len(combinations+optimal_combinations))
    update_progress(idx+1, len(combinations+optimal_combinations))
    return df_res, df_res_rel

def read_results(combinations, optimal_combinations, cost_types, root_result_dir='../results/',
                 deaggregate=False, rslt_dir_lyr='/agents'):
    '''

    Parameters
    ----------
    combinations : TYPE
        DESCRIPTION.
    optimal_combinations : TYPE
        DESCRIPTION.
    suffixes : TYPE
        DESCRIPTION.
    root_result_dir : TYPE, optional
        DESCRIPTION. The default is '../results/'.
    deaggregate : TYPE, optional
        DESCRIPTION. The default is False.
    rslt_dir_lyr : TYPE, optional
        DESCRIPTION. The default is '/agents'.

    Returns
    -------
    cmplt_results : TYPE
        DESCRIPTION.

    '''
    columns = ['t', 'Magnitude', 'cost_type', 'decision_type', 'judgment_type',
               'auction_type', 'valuation_type', 'no_resources', 'sample',
               'cost', 'normalized_cost', 'layer']
    optimal_method = ['tdindp', 'indp', 'sample_indp_12Node']
    cost_types += ['Under Supply Perc']
    cmplt_results = pd.DataFrame(columns=columns, dtype=int)
    objs = {}
    print("\nAggregating Results")
    joinedlist = combinations + optimal_combinations
    for idx, x in enumerate(joinedlist):
        #: Make the directory
        full_suffix = '_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])
        if x[4] == 'jc':
            full_suffix += '_'+x[5]
            if x[6] in ["MDA", "MAA", "MCA"]:
                full_suffix += '_AUCTION_'+x[6]+'_'+x[7]
            else:
                full_suffix += '_'+x[6]
        result_dir = root_result_dir+x[4]+'_results'+full_suffix
        if os.path.exists(result_dir):
            # Save all results to Pandas dataframe
            sample_result = indputils.INDPResults()
            sam_rslt_lyr = {l+1:indputils.INDPResults() for l in range(x[2])}
            ### !!! Assume the layer name is l+1
            sample_result = sample_result.from_csv(result_dir, x[1], suffix=x[8])
            if deaggregate:
                for l in range(x[2]):
                    sam_rslt_lyr[l+1] = sam_rslt_lyr[l+1].from_csv(result_dir+rslt_dir_lyr, x[1],
                                                                   suffix='L'+str(l+1)+'_'+x[8])
            initial_cost = {}
            for c in cost_types:
                initial_cost[c] = sample_result[0]['costs'][c]
            norm_cost = 0
            for t in sample_result.results:
                for c in cost_types:
                    if initial_cost[c] != 0.0:
                        norm_cost = sample_result[t]['costs'][c]/initial_cost[c]
                    else:
                        norm_cost = -1.0
                    values = [t, x[0], c, x[4], x[5], x[6], x[7], x[3], x[1],
                              float(sample_result[t]['costs'][c]), norm_cost, 'nan']
                    cmplt_results = cmplt_results.append(dict(zip(columns, values)),
                                                         ignore_index=True)
            if deaggregate:
                for l in range(x[2]):
                    initial_cost = {}
                    for c in cost_types:
                        initial_cost[c] = sam_rslt_lyr[l+1][0]['costs'][c]
                    norm_cost = 0
                    for t in sam_rslt_lyr[l+1].results:
                        for c in cost_types:
                            if initial_cost[c] != 0.0:
                                norm_cost = sam_rslt_lyr[l+1][t]['costs'][c]/initial_cost[c]
                            else:
                                norm_cost = -1.0
                            values = [t, x[0], c, x[4], x[5], x[6], x[7], x[3], x[1],
                                      float(sam_rslt_lyr[l+1][t]['costs'][c]), norm_cost, l+1]
                            cmplt_results = cmplt_results.append(dict(zip(columns, values)),
                                                                 ignore_index=True)
            #: Getting back the JuCModel objects:
            if x[4] == 'jc':
                with open(result_dir+'/objs_'+str(x[1])+'.pkl', 'rb') as f:
                    objs[str(x)] = pickle.load(f)
            if idx%(len(joinedlist)/10+1) == 0:
                update_progress(idx+1, len(joinedlist))
        else:
            sys.exit('Error: The combination or folder does not exist'+str(x))
    update_progress(len(joinedlist), len(joinedlist))
    return cmplt_results, objs

def read_run_time(combinations, optimal_combinations, suffixes, root_result_dir='../results/'):
    '''

    Parameters
    ----------
    combinations : TYPE
        DESCRIPTION.
    optimal_combinations : TYPE
        DESCRIPTION.
    suffixes : TYPE
        DESCRIPTION.
    root_result_dir : TYPE, optional
        DESCRIPTION. The default is '../results/'.

    Returns
    -------
    run_time_results : TYPE
        DESCRIPTION.

    '''
    columns = ['t', 'Magnitude', 'decision_type', 'auction_type', 'valuation_type',
               'no_resources', 'sample', 'decision_time', 'auction_time', 'valuation_time']
    optimal_method = ['tdindp', 'indp', 'sample_indp_12Node']
    run_time_results = pd.DataFrame(columns=columns, dtype=int)
    print("\nReading tun times")
    joinedlist = combinations + optimal_combinations
    for idx, x in enumerate(joinedlist):
        if x[4] in optimal_method:
            full_suffix = '_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])
        elif x[5] == 'Uniform':
            full_suffix = '_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])+'_uniform_alloc'
        else:
            full_suffix = '_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])+'_auction_'+x[5]+'_'+x[6]
        result_dir = root_result_dir+x[4]+'_results'+full_suffix
        run_time_all = {}
        if os.path.exists(result_dir):
            for suf in suffixes:
                run_time_file = result_dir+"/run_time_"  +str(x[1])+"_"+suf+".csv"
                if os.path.exists(run_time_file):
                # Save all results to Pandas dataframe
                    with open(run_time_file) as f:
                        lines = f.readlines()[1:]
                        for line in lines:
                            data = line.strip().split(',')
                            t = int(data[0])
                            run_time_all[t] = [float(data[1]), 0, 0]
                if x[4] not in optimal_method and x[5] != 'Uniform':
                    auction_file = result_dir+"/auctions/auctions_"  +str(x[1])+"_"+suf+".csv"
                    if os.path.exists(auction_file):
                    # Save all results to Pandas dataframe
                        with open(auction_file) as f:
                            lines = f.readlines()[1:]
                            for line in lines:
                                data = line.strip().split(',')
                                t = int(data[0])
                                auction_time = float(data[2*x[2]+5])
                                decision_time = run_time_all[t][0]
                                valuation_time_max = 0.0
                                for vtm in range(x[2]):
                                    if float(data[2*x[2]+5+vtm+1]) > valuation_time_max:
                                        valuation_time_max = float(data[2*x[2]+5+vtm+1])
                                run_time_all[t] = [decision_time, auction_time, valuation_time_max]
            for t, value in run_time_all.items():
                values = [t, x[0], x[4], x[5], x[6], x[3], x[1], value[0], value[1], value[2]]
                run_time_results = run_time_results.append(dict(zip(columns, values)),
                                                           ignore_index=True)
            if idx%(len(joinedlist)/10+1) == 0:
                update_progress(idx+1, len(joinedlist))
        else:
            sys.exit('Error: The combination or folder does not exist')
    update_progress(len(joinedlist), len(joinedlist))
    return run_time_results

def relative_performance(r_df, combinations, optimal_combinations, ref_method='indp',
                         ref_jt='nan', ref_at='nan', ref_vt='nan', cost_type='Total'):
    '''

    Parameters
    ----------
    r_df : TYPE
        DESCRIPTION.
    combinations : TYPE
        DESCRIPTION.
    optimal_combinations : TYPE
        DESCRIPTION.
    ref_method : TYPE, optional
        DESCRIPTION. The default is 'indp'.
    ref_at : TYPE, optional
        DESCRIPTION. The default is ''.
    ref_vt : TYPE, optional
        DESCRIPTION. The default is ''.
    ref_vt : TYPE, optional
        DESCRIPTION. The default is ''.
    cost_type : TYPE, optional
        DESCRIPTION. The default is 'Total'.

    Returns
    -------
    lambda_df : TYPE
        DESCRIPTION.

    '''
    columns = ['Magnitude', 'cost_type', 'decision_type', 'judgment_type', 'auction_type',
               'valuation_type', 'no_resources', 'sample',
               'Area_TC', 'Area_P', 'lambda_tc', 'lambda_p', 'lambda_U']
    T = len(r_df['t'].unique())
    lambda_df = pd.DataFrame(columns=columns, dtype=int)
    # Computing reference area for lambda
    # Check if the method in optimal combination is the reference method #!!!
    print('\nRef area calculation\n', end='')
    for idx, x in enumerate(optimal_combinations):
        if x[4] == ref_method:
            rows = r_df[(r_df['Magnitude'] == x[0])&(r_df['decision_type'] == ref_method)&
                        (r_df['sample'] == x[1])&(r_df['auction_type'] == ref_at)&
                        (r_df['valuation_type'] == ref_vt)&(r_df['no_resources'] == x[3])&
                        (r_df['judgment_type'] == ref_jt)]
            if not rows.empty:
                area_tc = trapz_int(y=list(rows[rows['cost_type'] == cost_type].cost[:T]),
                                    x=list(rows[rows['cost_type'] == cost_type].t[:T]))
                area_p = -trapz_int(y=list(rows[rows['cost_type'] == 'Under Supply Perc'].cost[:T]),
                                    x=list(rows[rows['cost_type'] == 'Under Supply Perc'].t[:T]))
                values = [x[0], cost_type, x[4], ref_jt, ref_at, ref_vt, x[3], x[1],
                          area_tc, area_p, 'nan', 'nan', 'nan']
                lambda_df = lambda_df.append(dict(zip(columns, values)), ignore_index=True)
            if idx%(len(optimal_combinations)/10+1) == 0:
                update_progress(idx+1, len(optimal_combinations))
    update_progress(len(optimal_combinations), len(optimal_combinations))
    # Computing areas and lambdas
    print('\nLambda calculation\n', end='')
    for idx, x in enumerate(combinations+optimal_combinations):
        if x[4] != ref_method:
            # Check if reference area exists
            cond = ((lambda_df['Magnitude'] == x[0])&(lambda_df['decision_type'] == ref_method)&
                    (lambda_df['auction_type'] == ref_at)&(lambda_df['valuation_type'] == ref_vt)&
                    (lambda_df['cost_type'] == cost_type)&(lambda_df['sample'] == x[1])&
                    (lambda_df['no_resources'] == x[3])&(lambda_df['judgment_type'] == ref_jt))
            if not cond.any():
                sys.exit('Error:Reference type is not here! for %s,%s, m %d, resource %d'\
                         %(x[4], x[5], x[0], x[3]))
            ref_area_tc = float(lambda_df.loc[cond, 'Area_TC'])
            ref_area_P = float(lambda_df.loc[cond, 'Area_P'])
            rows = r_df[(r_df['Magnitude'] == x[0])&(r_df['decision_type'] == x[4])&
                        (r_df['judgment_type'] == x[5])&(r_df['auction_type'] == x[6])&
                        (r_df['valuation_type'] == x[7])&(r_df['sample'] == x[1])&
                        (r_df['no_resources'] == x[3])]
            if not rows.empty:
                area_tc = trapz_int(y=list(rows[rows['cost_type'] == cost_type].cost[:T]),
                                    x=list(rows[rows['cost_type'] == cost_type].t[:T]))
                area_p = -trapz_int(y=list(rows[rows['cost_type'] == 'Under Supply Perc'].cost[:T]),
                                    x=list(rows[rows['cost_type'] == 'Under Supply Perc'].t[:T]))
                lambda_tc = 'nan'
                lambda_p = 'nan'
                if ref_area_tc != 0.0 and area_tc != 'nan':
                    lambda_tc = (ref_area_tc-float(area_tc))/ref_area_tc
                elif area_tc == 0.0:
                    lambda_tc = 0.0
                if ref_area_P != 0.0 and area_p != 'nan':
                    lambda_p = (ref_area_P-float(area_p))/ref_area_P
                elif area_p == 0.0:
                    lambda_p = 0.0
                else:
                    pass
                values = [x[0], cost_type, x[4], x[5], x[6], x[7], x[3], x[1], area_tc,
                          area_p, lambda_tc, lambda_p, (lambda_tc+lambda_p)/2]
                lambda_df = lambda_df.append(dict(zip(columns, values)), ignore_index=True)
            else:
                sys.exit('Error: No entry for %s %s %s m %d|resource %d, ...'\
                         %(x[4], x[5], x[6], x[0], x[3]))
        if idx%(len(combinations+optimal_combinations)/10+1) == 0:
            update_progress(idx+1, len(combinations+optimal_combinations))
    update_progress(idx+1, len(combinations+optimal_combinations))
    return lambda_df

def generate_combinations(database, mags, sample, layers, no_resources, decision_type,
                          judgment_type, res_alloc_type, valuation_type, suffixes='',
                          list_high_dam_add=None, synthetic_dir=None):
    '''

    Parameters
    ----------
    database : TYPE
        DESCRIPTION.
    mags : TYPE
        DESCRIPTION.
    sample : TYPE
        DESCRIPTION.
    layers : TYPE
        DESCRIPTION.
    no_resources : TYPE
        DESCRIPTION.
    decision_type : TYPE
        DESCRIPTION.
    res_alloc_type : TYPE
        DESCRIPTION.
    valuation_type : TYPE
        DESCRIPTION.
    suffixes : TYPE, optional
        DESCRIPTION. The default is ''.
    list_high_dam_add : TYPE, optional
        DESCRIPTION. The default is None.
    synthetic_dir : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    combinations : TYPE
        DESCRIPTION.
    optimal_combinations : TYPE
        DESCRIPTION.

    '''
    combinations = []
    optimal_combinations = []
    optimal_method = ['tdindp', 'indp', 'sample_indp_12Node']
    print('\nCombination Generation\n', end='')
    idx = 0
    no_total = len(mags)*len(sample)
    if database == 'shelby':
        if list_high_dam_add:
            list_high_dam = pd.read_csv(list_high_dam_add)
        L = len(layers)
        for m, s in itertools.product(mags, sample):
            if list_high_dam_add is None or len(list_high_dam.loc[(list_high_dam.set == s)&\
                                                                  (list_high_dam.sce == m)].index):
                for rc in no_resources:
                    for dt, jt, at, vt in itertools.product(decision_type, judgment_type,
                                                            res_alloc_type, valuation_type):
                        if (dt in optimal_method) and [m, s, L, rc, dt, 'nan', 'nan', 'nan', '']\
                            not in optimal_combinations:
                            optimal_combinations.append([m, s, L, rc, dt, 'nan',
                                                         'nan', 'nan', ''])
                        elif (dt not in optimal_method) and (at not in ['UNIFORM']):
                            for sf in suffixes:
                                combinations.append([m, s, L, rc, dt, jt, at, vt, sf])
                        elif (dt not in optimal_method) and (at in ['UNIFORM']):
                            for sf in suffixes:
                                combinations.append([m, s, L, rc, dt, jt, at, 'nan', sf])
            idx += 1
            update_progress(idx, no_total)
    elif database == 'synthetic':
        # Read net configurations
        if synthetic_dir is None:
            sys.exit('Error: Provide the address of the synthetic databse')
        with open(synthetic_dir+'List_of_Configurations.txt') as f:
            config_data = pd.read_csv(f, delimiter='\t')
        for m, s in itertools.product(mags, sample):
            config_param = config_data.iloc[m]
            L = int(config_param.loc[' No. Layers'])
            no_resources = int(config_param.loc[' Resource Cap'])
            for rc in [no_resources]:
                for dt, jt, at, vt in itertools.product(decision_type, judgment_type,
                                                    res_alloc_type, valuation_type):
                    if (dt in optimal_method) and [m, s, L, rc, dt] not in optimal_combinations:
                        optimal_combinations.append([m, s, L, rc, dt, 'nan',
                                                         'nan', 'nan', ''])
                    elif (dt not in optimal_method) and (at not in ['UNIFORM']):
                        for sf in suffixes:
                            combinations.append([m, s, L, rc, dt, jt, at, vt, sf])
                    elif (dt not in optimal_method) and (at in ['UNIFORM']):
                        for sf in suffixes:
                            combinations.append([m, s, L, rc, dt, jt, 'nan', at, sf])
            idx += 1
            update_progress(idx, no_total)
    else:
        sys.exit('Error: Wrong database type')
    return combinations, optimal_combinations

def update_progress(progress, total):
    '''

    Parameters
    ----------
    progress : TYPE
        DESCRIPTION.
    total : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    print('\r[%s] %1.1f%%' % ('#'*int(progress/float(total)*20), (progress/float(total)*100)), end='')
    sys.stdout.flush()

def trapz_int(x, y):
    '''

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if not np.all([i < j for i, j in zip(x[:-1], x[1:])]):
        x, y = (list(t) for t in zip(*sorted(zip(x, y))))
    return np.trapz(y, x)
