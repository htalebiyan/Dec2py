'''
This module contains functions to run decentralized restoration for interdepndent networks
using Judgment Call method :cite:`Talebiyan2019c,Talebiyan2019`, read the results,
and compute comparison measures.
'''
import os.path
import operator
import copy
import itertools
import time
import sys
import pandas as pd
import numpy as np
import pickle
import indp
import indputils
from dindpclasses import JcModel

def run_judgment_call(params, save_jc=True, print_cmd=True, save_jc_model=False):
    '''
    Finds interdepndent restoration strategies using a decentralized hueristic,
    Judgment Call :cite:`Talebiyan2019c,Talebiyan2019`.

    Parameters
    ----------
    params : dict
         Global parameters, including number of iterations, judgment type, etc.
    save_jc : bool, optional
        If true, the results are saved to files. The default is True.
    print_cmd : bool, optional
        If true, the results are printed to console. The default is True.
    save_jc_model : bool, optional
        If true, optimization models and their solutions are printed to file. The default is False.

    Returns
    -------
    :
        None

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
                output_dir_full = params["OUTPUT_DIR"]+'_L'+str(len(params["L"]))+'_m'+\
                                str(params["MAGNITUDE"])+"_v"+str(params["V"])+'_'+jc+'_'+\
                                rst+'/actions_'+str(params["SIM_NUMBER"])+'_real.csv'
                if os.path.exists(output_dir_full):
                    print('Judgment Call:',jc,rst,'results are already there\n')
                else:
                    objs[c] = JcModel(c, params_copy)
                    c += 1
            else:
                for vt in params["VALUATION_TYPE"]:
                    params_copy['VALUATION_TYPE'] = vt
                    output_dir_full = params["OUTPUT_DIR"]+'_L'+str(len(params["L"]))+'_m'+\
                                    str(params["MAGNITUDE"])+"_v"+str(params["V"])+'_'+jc+'_AUCTION_'+\
                                    rst+'_'+vt+'/actions_'+str(params["SIM_NUMBER"])+'_real.csv'
                    if os.path.exists(output_dir_full):
                        print('Judgment Call:',jc,rst,vt,'results are already there\n')
                    else:
                        objs[c] = JcModel(c, params_copy)
                        c += 1
    if not objs:
        return 0
    # t=0 costs and performance.
    indp_results_initial = indp.indp(objs[0].net, 0, 1, objs[0].layers,
                                     controlled_layers=objs[0].layers)
    for _, obj in objs.items():
        print('--Running JC: '+obj.judge_type+', resource allocation: '+obj.res_alloc_type)
        if obj.resource.type == 'AUCTION':
            print('auction type: '+obj.resource.auction_model.auction_type+\
                  ', valuation: '+obj.resource.auction_model.valuation_type)
        if print_cmd:
            print("Num iters=", params["NUM_ITERATIONS"])
        # t=0 results.
        obj.results_judge = copy.deepcopy(indp_results_initial[1]) #!!! deepcopy
        obj.results_real = copy.deepcopy(indp_results_initial[1]) #!!! deepcopy
        for i in range(num_iterations):
            print("-Time Step (JC)", i+1, "/", num_iterations)
            #: Resource Allocation
            res_alloc_time_start = time.time()
            if obj.resource.type == 'AUCTION':
                obj.resource.auction_model.auction_resources(obj, i+1, print_cmd=print_cmd,
                                                             compute_poa=True)
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
                obj.results_judge.extend(indp_results[1], t_offset=i+1)
                obj.results_judge.results_layer[l][i+1]['costs']['Space Prep'] = indp_results[1].results[0]['costs']['Space Prep']
                # Save models to file
                if save_jc_model:
                    indp.save_INDP_model_to_file(indp_results[0], obj.output_dir+"/Model",
                                                 i+1, l)
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
                obj.results_real.extend(indp_results_real[1], t_offset=i+1)
                obj.correct_results_real(l, i+1)
                if save_jc_model:
                    indp.save_INDP_model_to_file(indp_results_real[0], obj.output_dir+"/Model",
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
    obj : :class:`~dindpclasses.JcModel`
        Objects that contain the Judgment Call model.
    t_step : int
        Current time step.
    functionality : dict
        Dictionary of actual functionality state of nodes in dependee networks.
    judger_layer : int
        Name of the layer whose agent has made the judgment.
    print_cmd : bool, optional
        If true, the results are printed to console. The default is False.

    Returns
    -------
    indp_results_real : :class:`~indputils.INDPResults`
        Results that represesnt the realized perforamnce of the decentralized strategy.

    '''
    time_limit = 10*60 #!!! Might be adjusted later
    functionality_realized = copy.deepcopy(functionality) #!!! deepcopy
    dest_nodes = obj.judgments.dest_nodes[t_step][judger_layer]
    for v, val in dest_nodes.items():
        val.append(obj.net.G.nodes[v]['data']['inf_data'].functionality)
    judged_nodes = obj.judgments.judged_nodes[t_step][judger_layer]
    for u, val in judged_nodes.items():
        if functionality[0][u] == 1.0 and\
            obj.net.G.nodes[u]['data']['inf_data'].functionality == 0.0:
            functionality_realized[0][u] = 0.0
            if print_cmd:
                print('Correct judgment: '+str(u)+'<-0')
        val.append(obj.net.G.nodes[u]['data']['inf_data'].functionality)
    indp_results_real = indp.indp(obj.net, v_r=0, T=1, layers=obj.layers,
                                  controlled_layers=[judger_layer],
                                  functionality=functionality_realized,
                                  print_cmd=print_cmd, time_limit=time_limit)
    for v, val in dest_nodes.items():
        nodeVar = 'w_'+str(v)+","+str(0)
        val.append(indp_results_real[0].getVarByName(nodeVar).x)
    return indp_results_real

def read_results(combinations, optimal_combinations, cost_types, root_result_dir='../results/',
                 deaggregate=False, rslt_dir_lyr='/agents'):
    '''
    This function reads the results of analyses (INDP, JC, etc.) and the corresponding
    objects from file and aggregates the results in a dictionary.

    Parameters
    ----------
    combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the JC (or any other decentralized results) collected by
        :func:`generate_combinations`.
    optimal_combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the INDP (or any other optimal results) collected by :func:`generate_combinations`.
    cost_types : str
        Cost types that should be read from results and will be shown in the plots.
    root_result_dir : 'str', optional
        Root directory where the results are stored. The default is '../results/'.
    deaggregate : bool, optional
        Should the deaggregated results (for seperate layers) be read. The default is False.
    rslt_dir_lyr : str, optional
        Directory insdie the :func:`root result directory <read_results>` where
        the deaggregated results (for seperate layers)  are. The default is '/agents'.

    Returns
    -------
    cmplt_results : dict
        Dictionary that contains the read results.
    objs : dict
        Dictionary that contains the objects corresponding to the read results.

    '''
    columns = ['t', 'Magnitude', 'cost_type', 'decision_type', 'judgment_type',
               'auction_type', 'valuation_type', 'no_resources', 'sample',
               'cost', 'normalized_cost', 'layer']
    cost_types += ['Under Supply Perc']
    cmplt_results = pd.DataFrame(columns=columns, dtype=int)
    objs = {}
    print("\nAggregating Results")
    joinedlist = combinations + optimal_combinations
    for idx, x in enumerate(joinedlist):
        #: Make the directory
        full_suffix = '_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])
        if x[4][:2] == 'jc' or x[4][:2] == 'ng':
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
            if x[4][:2] == 'jc' or x[4][:2] == 'ng':
                with open(result_dir+'/objs_'+str(x[1])+'.pkl', 'rb') as f:
                    objs[str(x)] = pickle.load(f)
            if idx%(len(joinedlist)//100+1) == 0:
                update_progress(idx+1, len(joinedlist))
        else:
            sys.exit('Error: The combination or folder does not exist'+str(x))
    update_progress(len(joinedlist), len(joinedlist))
    return cmplt_results, objs

def relative_performance(r_df, combinations, optimal_combinations, ref_method='indp',
                         ref_jt='nan', ref_at='nan', ref_vt='nan', cost_type='Total'):
    '''
    This functions computes the relative performance, relative cost, and univeral
    relative measure :cite:`Talebiyan2019c` based on results from JC and INDP.

    Parameters
    ----------
    r_df : dict
        Dictionary that contains complete results by JC and INDP collected by
        :func:`read_results`.
    combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the JC (or any other decentralized results) collected by
        :func:`generate_combinations`.
    optimal_combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the INDP (or any other optimal results) collected by :func:`generate_combinations`.
    ref_method : str, optional
        Referece method to computue relative measure in comparison to. The default is 'indp'.
    ref_jt : str, optional
        Referece judgment type to computue relative measure in comparison to. It is used only
        when the reference method is JC. The default is 'nan'.
    ref_at : str, optional
        Referece resource allocation type to computue relative measure in comparison to.
        It is used only when the reference method is JC. The default is 'nan'.
    ref_vt : str, optional
        Referece val;uation type to computue relative measure in comparison to.
        It is used only when the reference method is JC, and the reference resource
        allocation type is Auntion. The default is 'nan'.
    cost_type : str, optional
        Cost type for which the relative measure is computed. The default is 'Total'.

    Returns
    -------
    lambda_df : dict
        Dictionary that contains the relative measures.

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

def read_resourcec_allocation(result_df, combinations, optimal_combinations, objs,
                              ref_method='indp', root_result_dir='../results/'):
    '''
    This functions reads the resource allocation vectors by INDP and JC. Also,
    it computes the allocation gap between the respurce allocation by JC and
    and the optimal allocation by INDP :cite:`Talebiyan2020a`.

    Parameters
    ----------
    result_df : dict
        Dictionary that contains complete results by JC and INDP collected by
        :func:`read_results`.
    combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the JC (or any other decentralized results) collected by
        :func:`generate_combinations`.
    optimal_combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the INDP (or any other optimal results) collected by :func:`generate_combinations`.
    ref_method : str, optional
        Referece method to computue relative measure in comparison to. The default is 'indp'.
    root_result_dir : str, optional
        Directory that contains the results. The default is '../results/'.

    Returns
    -------
    df_res : dict
        Dictionary that contain the resoruce allcoation vectors.
    df_alloc_gap : dict
        Dictionary that contain the allcoation gap values.
    '''
    cols = ['t', 'resource', 'decision_type', 'judgment_type', 'auction_type',
            'valuation_type', 'sample', 'Magnitude', 'layer', 'no_resources',
            'normalized_resource', 'poa']
    T = max(result_df.t.unique().tolist())
    df_res = pd.DataFrame(columns=cols, dtype=int)
    print('\nResource allocation')
    for idx, x in enumerate(optimal_combinations):
        ref_dir = root_result_dir+x[4]+'_results_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])
        for t in range(T):
            for l in range(1, x[2]+1):
                df_res = df_res.append({'t':t+1, 'resource':0.0, 'normalized_resource':0.0,
                                        'decision_type':x[4], 'judgment_type':'nan',
                                        'auction_type':'nan', 'valuation_type':'nan',
                                        'sample':x[1], 'Magnitude':x[0], 'layer':l,
                                        'no_resources':x[3], 'poa':1}, ignore_index=True)
        # Read optimal resource allocation based on the actions
        action_file = ref_dir+"/actions_"+str(x[1])+"_"+x[8]+".csv"
        if os.path.isfile(action_file):
            with open(action_file) as f:
                lines = f.readlines()[1:]
                for line in lines:
                    data = line.strip().split(',')
                    t = int(data[0])
                    action = str.strip(data[1])
                    l = int(action[-1])
                    if '/' in action:
                        addition = 0.5
                    else:
                        addition = 1.0
                    row = (df_res['t'] == t)&(df_res['decision_type'] == x[4])&\
                    (df_res['sample'] == x[1])&(df_res['Magnitude'] == x[0])&\
                    (df_res['layer'] == l)&(df_res['no_resources'] == x[3])
                    df_res.loc[row, 'resource'] += addition
                    df_res.loc[row, 'normalized_resource'] += addition/float(x[3])
        if idx%(len(combinations+optimal_combinations)/10+1) == 0:
            update_progress(idx+1, len(optimal_combinations)+len(combinations))
    # Read resource allocation based on resource allocation results
    for idx, x in enumerate(combinations):
        obj = objs[str(x)]
        for t, tval in obj.v_r.items():
            if x[6] in ["MDA", "MAA", "MCA"]:
                poa = obj.resource.auction_model.poa[t]
            else:
                poa = 'nan'
            for l, lval in tval.items():
                df_res = df_res.append({'t':t, 'resource':lval, 'normalized_resource':lval/x[3],
                                        'decision_type':x[4], 'judgment_type':x[5],
                                        'auction_type':x[6], 'valuation_type':x[7],
                                        'sample':x[1], 'Magnitude':x[0], 'layer':l,
                                        'no_resources':x[3], 'poa':poa}, ignore_index=True)
        if idx%(len(combinations+optimal_combinations)/10+1) == 0:
            update_progress(len(optimal_combinations)+idx+1,
                            len(optimal_combinations)+len(combinations))
    update_progress(len(optimal_combinations)+idx+1, len(optimal_combinations)+len(combinations))
    #: populate allocation gap dictionary
    cols = ['decision_type', 'judgment_type', 'auction_type', 'valuation_type', 'sample',
            'Magnitude', 'layer', 'no_resources', 'gap', 'norm_gap']
    T = max(result_df.t.unique().tolist())
    df_alloc_gap = pd.DataFrame(columns=cols, dtype=int)
    print('\nAllocation Gap')
    for idx, x in enumerate(combinations+optimal_combinations):
        # Construct vector of resource allocation of reference method
        if x[4] != ref_method:
            vec_ref = {l:np.zeros(T) for l in range(1, x[2]+1)}
            for l in range(1, x[2]+1):
                for t in range(T):
                    vec_ref[l][t] = df_res.loc[(df_res['t'] == t+1)&
                                               (df_res['decision_type'] == ref_method)&
                                               (df_res['sample'] == x[1])&
                                               (df_res['Magnitude'] == x[0])&
                                               (df_res['layer'] == l)&
                                               (df_res['no_resources'] == x[3]), 'resource']
            # Compute distance of resource allocation vectors
            vector_res = {l:np.zeros(T) for l in range(1, x[2]+1)}
            for l in range(1, x[2]+1):
                row = (df_res['decision_type'] == x[4])&(df_res['sample'] == x[1])&\
                (df_res['Magnitude'] == x[0])&(df_res['layer'] == l)&\
                (df_res['no_resources'] == x[3])&(df_res['auction_type'] == x[6])&\
                (df_res['valuation_type'] == x[7])&(df_res['judgment_type'] == x[5])
                for t in range(T):
                    vector_res[l][t] = df_res.loc[(df_res['t'] == t+1)&row, 'resource']
                #L2 norm
                distance = np.linalg.norm(vector_res[l]-vec_ref[l])
                norm_distance = np.linalg.norm(vector_res[l]/float(x[3])-\
                                               vec_ref[l]/float(x[3]))
                # #L1 norm
                # distance = sum(abs(vector_res[l]-vec_ref[l]))
                # # correlation distance
                # distance = 1-scipy.stats.pearsonr(vector_res[l], vec_ref[l])[0]
                df_alloc_gap = df_alloc_gap.append({'decision_type':x[4], 'judgment_type':x[5],
                                                    'auction_type':x[6], 'valuation_type':x[7],
                                                    'sample':x[1], 'Magnitude':x[0], 'layer':l,
                                                    'no_resources':x[3],
                                                    'gap':distance/float(vector_res[l].shape[0]),
                                                    'norm_gap':norm_distance/float(vector_res[l].shape[0])},
                                                   ignore_index=True)
            if idx%(len(combinations+optimal_combinations)/10+1) == 0:
                update_progress(idx+1, len(combinations+optimal_combinations))
    update_progress(idx+1, len(combinations+optimal_combinations))
    return df_res, df_alloc_gap

def read_run_time(combinations, optimal_combinations, objs, root_result_dir='../results/'):
    '''
    This function reads the run time of computing restoration strategies by different methods.

    Parameters
    ----------
    combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the JC (or any other decentralized results) collected by
        :func:`generate_combinations`.
    optimal_combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the INDP (or any other optimal results) collected by :func:`generate_combinations`.
    objs : dict
        Dictionary that contains the objects corresponding to the results collected
        by :func:`read_results`.
    root_result_dir : str, optional
        Directory that contains the results. The default is '../results/'.

    Returns
    -------
    run_time_results : dict
        Dictionary that contain run time of for all computed strategies.

    '''
    columns = ['t', 'Magnitude', 'decision_type', 'judgment_type', 'auction_type', 'valuation_type',
               'no_resources', 'sample', 'decision_time', 'auction_time', 'valuation_time']
    run_time_results = pd.DataFrame(columns=columns, dtype=int)
    print("\nReading tun times")
    joinedlist = combinations + optimal_combinations
    for idx, x in enumerate(optimal_combinations):
        full_suffix = '_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])
        result_dir = root_result_dir+x[4]+'_results'+full_suffix
        if os.path.exists(result_dir):
            run_time_file = result_dir+"/run_time_"+str(x[1])+"_"+x[8]+".csv"
            if os.path.exists(run_time_file):
            # Save all results to Pandas dataframe
                with open(run_time_file) as f:
                    lines = f.readlines()[1:]
                    for line in lines:
                        data = line.strip().split(',')
                        t = int(data[0])
                        values = [t, x[0], x[4], x[5], x[6], x[7], x[3], x[1],
                                  float(data[1]), 0, 0]
                        run_time_results = run_time_results.append(dict(zip(columns, values)),
                                                                   ignore_index=True)
        if idx%(len(joinedlist)/10+1) == 0:
            update_progress(idx+1, len(joinedlist))
    for idx, x in enumerate(combinations):
        obj = objs[str(x)]
        for t in range(obj.time_steps+1):
            if x[4] == 'jc':
                decision_time = obj.results_judge.results[t]['run_time']+\
                    obj.results_real.results[t]['run_time']
            elif x[4] == 'ng':
                if t==0:
                    decision_time = obj.results.results[t]['run_time']
                else:
                    payoff_time = max(obj.objs[t].payoff_time.items(),
                                      key=operator.itemgetter(1))[1]
                    decision_time = obj.objs[t].solving_time + payoff_time
            else:
                sys.exit('Error: Wrong method name in computing decision time')
            auction_time = 0
            val_time_max = 0
            if x[6] in ["MDA", "MAA", "MCA"] and t > 0:
                auction_time = obj.resource.auction_model.auction_time[t]
                val_time_max = max(obj.resource.auction_model.valuation_time[t].items(),
                                   key=operator.itemgetter(1))[1]
            values = [t, x[0], x[4], x[5], x[6], x[7], x[3], x[1],
                      decision_time, auction_time, val_time_max]
            run_time_results = run_time_results.append(dict(zip(columns, values)),
                                                       ignore_index=True)
        if idx%(len(combinations)/10+1) == 0:
            update_progress(len(optimal_combinations)+idx+1, len(joinedlist))
    update_progress(len(joinedlist), len(joinedlist))
    return run_time_results

def generate_combinations(database, mags, sample, layers, no_resources, decision_type,
                          judgment_type, res_alloc_type, valuation_type,
                          list_high_dam_add=None, synthetic_dir=None):
    '''
    This fucntion returns all combinations of magnitude, sample, judgment type,
    resource allocation type, and valuation type (if applicable) involved in
    decentralized and centralized analyses. The returend dictionary are used by
    other functions to read results and calculate comparison measures.

    Parameters
    ----------
    database : str
        Name of the initial damage database. \n
        options:
            For shelby county network: 'shelby', 'random', 'ANDRES', 'WU' \n
            For synthetic networks: 'synthetic'
    mags : range
        Range of magnitude parameter of the current simulation.
    sample : range
        Range of sample parameter of the current simulation.
    layers : list
        List of layers.
    no_resources : list
        List of number of available resources, :math:`R_c`.
    decision_type : list
        List of methods.
    res_alloc_type : list
        List of resoure allocation methods.
    valuation_type : list
        List of valuation types.
    list_high_dam_add : str, optional
        Address of the file containing the list of damage scenarios that should be read
        from file. It is used to read a selected subset of results. The default is None.
    synthetic_dir : str, optional
        Address of the synthetic database files. The default is None.

    Returns
    -------
    combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the JC (or any other decentralized results).
    optimal_combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the INDP (or any other optimal results).

    '''
    combinations = []
    optimal_combinations = []
    optimal_method = ['tdindp', 'indp', 'sample_indp_12Node']
    print('\nCombination Generation\n', end='')
    idx = 0
    no_total = len(mags)*len(sample)
    if database in ['shelby', 'random', 'ANDRES', 'WU']:
        if list_high_dam_add:
            list_high_dam = pd.read_csv(list_high_dam_add)
        L = len(layers)
        for m, s in itertools.product(mags, sample):
            if list_high_dam_add is None or len(list_high_dam.loc[(list_high_dam.set == s)&\
                                                                  (list_high_dam.sce == m)].index):
                for rc in no_resources:
                    for dt, jt, at, vt in itertools.product(decision_type, judgment_type,
                                                            res_alloc_type, valuation_type):
                        if dt == 'JC':
                            sf = 'real'
                        else:
                            sf = ''

                        if (dt in optimal_method) and\
                            [m, s, L, rc, dt, 'nan', 'nan', 'nan', ''] not in optimal_combinations:
                            optimal_combinations.append([m, s, L, rc, dt, 'nan',
                                                         'nan', 'nan', sf])
                        elif (dt not in optimal_method) and (at not in ['UNIFORM']):
                            combinations.append([m, s, L, rc, dt, jt, at, vt, sf])
                        elif (dt not in optimal_method) and (at in ['UNIFORM']):
                            if [m, s, L, rc, dt, jt, at, 'nan', sf] not in combinations:
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
                    if dt == 'JC':
                        sf = 'real'
                    else:
                        sf = ''
                    if (dt in optimal_method) and\
                        [m, s, L, rc, dt, 'nan', 'nan', 'nan', ''] not in optimal_combinations:
                        optimal_combinations.append([m, s, L, rc, dt, 'nan',
                                                     'nan', 'nan', sf])
                    elif (dt not in optimal_method) and (at not in ['UNIFORM']):
                        combinations.append([m, s, L, rc, dt, jt, at, vt, sf])
                    elif (dt not in optimal_method) and (at in ['UNIFORM']):
                        if [m, s, L, rc, dt, jt, at, 'nan', sf] not in combinations:
                            combinations.append([m, s, L, rc, dt, jt, at, 'nan', sf])
            idx += 1
            update_progress(idx, no_total)
    else:
        sys.exit('Error: Wrong database type')
    return combinations, optimal_combinations

def update_progress(progress, total):
    '''
    This function updates and writes a progress bar to console.

    Parameters
    ----------
    progress : int
        The current progress.
    total : int
        Total number of cases.

    Returns
    -------
    :
        None.

    '''
    print('\r[%s] %1.1f%%' % ('#'*int(progress/float(total)*20),
                              (progress/float(total)*100)), end='')
    sys.stdout.flush()

def trapz_int(x, y):
    '''
    This function computes the area underneath a curve (y) over time vector (x) including
    checking if the time vector is sorted.

    Parameters
    ----------
    x : list
        Time vector.
    y : list
        Vector of the integrand.

    Returns
    -------
    float
        Area underneath x-y curve.

    '''
    if not np.all([i < j for i, j in zip(x[:-1], x[1:])]):
        x, y = (list(t) for t in zip(*sorted(zip(x, y))))
    return np.trapz(y, x)
