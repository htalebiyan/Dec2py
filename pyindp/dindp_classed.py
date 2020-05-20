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
    ### Creating JC objects
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
    # Initial calculations.
    indp_results_initial = indp.indp(objs[0].net, 0, 1, objs[0].layers,
                                     controlled_layers=objs[0].layers)

    # Initialize failure scenario.
    if T == 1:
        for idx, obj in objs.items():
            print('--Running JC: '+obj.judge_type+', resource allocation: '+obj.res_alloc_type)
            if obj.res_alloc_type == 'auction':
                print('auction type: '+obj.resource.auction_model.auction_type+\
                      ', valuation: '+obj.resource.auction_model.valuation_type)
            if print_cmd:
                print("Num iters=", params["NUM_ITERATIONS"])
            # t=0 results.
            obj.results = indp_results_initial[1]
            obj.results_real = indp_results_initial[1]
            for i in range(num_iterations):
                print("-Iteration "+str(i+1)+"/"+str(num_iterations))
                res_alloc_time_start = time.time()
                if obj.resource.type == 'auction':
                    obj.resource.auction_model.auction_resources(obj, i+1, print_cmd=print_cmd,
                                                                 compute_poa=False)
                functionality = {l:{} for l in obj.layers}
                uncorrected_results = {}
                if print_cmd:
                    print("Judgment: ")
                for l in obj.layers:
                    if print_cmd:
                        print("Layer-%d"%(l))
                    neg_layer = [x for x in obj.layers if x != l]
                    functionality[l] = obj.judgments.create_judgment_dict(obj, neg_layer)
                    obj.judgments.save_judgments(obj, functionality[l], l, i+1)
                # Make decision based on judgments before communication
                # indp_results = indp.indp(interdep_net, v_r_applied[l-1], 1, layers=layers,
                #                 controlled_layers=[l], functionality=functionality[l],
                #                 print_cmd=print_cmd, time_limit=10*60)
                # # Save models for re-evaluation after communication
                # uncorrected_results[l] = indp_results[1]
                # # Save results of decisions based on judgments
                # dindp_results[l].extend(indp_results[1], t_offset=i+1)
                # # Save models to file
                # if save_jc_model:
                #     indp.save_INDP_model_to_file(indp_results[0], output_dir+"/Model", i+1, l)
                # # Modify network to account for recovery and calculate components.
                # indp.apply_recovery(interdep_net, dindp_results[l], i+1)
                # dindp_results[l].add_components(i+1, indputils.INDPComponents.calculate_components(indp_results[0], interdep_net, layers=[l]))
    #         # Re-evaluate judgments based on other agents' decisions
    #         if print_cmd:
    #             print("Re-evaluation: ")
    #         for l in layers:
    #             if print_cmd:
    #                 print("Layer-%d"%(l))
    #             indp_results_real, realizations = realized_performance(interdep_net,
    #                             uncorrected_results[l], functionality=functionality[l],
    #                             T=1, layers=layers, controlled_layers=[l], print_cmd=print_cmd)
    #             dindp_results_real[l].extend(indp_results_real[1], t_offset=i+1)
    #             if save_jc_model:
    #                 indp.save_INDP_model_to_file(indp_results_real[0], output_dir+"/Model",
    #                                              i+1, l, suffix='Real')
    #                 output_dir_judgments = output_dir + '/judgments'
    #                 write_judgments_csv(output_dir_judgments, realizations,
    #                                     sample_num=params["SIM_NUMBER"],
    #                                     agent=l, time_step=i+1, suffix="")
    #     # Calculate sum of costs
    #     dindp_sum, dindp_real_sum = compute_cost_sum(dindp_results, dindp_results_real,
    #                                                  interdep_net, indp_results_initial,
    #                                                  num_iterations, layers)
    #     output_dir_auction = output_dir + '/auctions'
    #     if auction_type:
    #         write_auction_csv(output_dir_auction, res_allocate, res_alloc_time,
    #                           poa, valuations, sample_num=params["SIM_NUMBER"], suffix="")
    #     else:
    #         write_auction_csv(output_dir_auction, res_allocate, res_alloc_time,
    #                           sample_num=params["SIM_NUMBER"], suffix="")
    #     # Save results of D-iINDP run to file.
    #     if save_jc:
    #         output_dir_agents = output_dir + '/agents'
    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
    #         if not os.path.exists(output_dir_agents):
    #             os.makedirs(output_dir_agents)
    #         for l in layers:
    #             dindp_results[l].to_csv(output_dir_agents, params["SIM_NUMBER"],
    #                                     suffix=str(l))
    #             dindp_results_real[l].to_csv(output_dir_agents, params["SIM_NUMBER"],
    #                                          suffix='Real_'+str(l))
    #         dindp_sum.to_csv(output_dir, params["SIM_NUMBER"], suffix='sum')
    #         dindp_real_sum.to_csv(output_dir, params["SIM_NUMBER"], suffix='Real_sum')
    return objs

def compute_cost_sum(dindp_results, dindp_results_real, interdep_net,
                     indp_results_initial, num_iterations, layers):
    '''
    computes sum of all costs from the cost of individual layers

    Parameters
    ----------
    dindp_results : TYPE
        DESCRIPTION.
    dindp_results_real : TYPE
        DESCRIPTION.
    interdep_net : TYPE
        DESCRIPTION.
    indp_results_initial : TYPE
        DESCRIPTION.
    num_iterations : TYPE
        DESCRIPTION.
    layers : TYPE
        DESCRIPTION.

    Returns
    -------
    dindp_results_sum : TYPE
        DESCRIPTION.
    dindp_results_real_sum : TYPE
        DESCRIPTION.

    '''
    dindp_results_sum = indputils.INDPResults()
    dindp_results_real_sum = indputils.INDPResults()
    cost_types = dindp_results[1][0]['costs'].keys()
    # compute total demand of all layers and each layer
    total_demand = 0.0
    total_demand_layer = {l:0.0 for l in layers}
    for n, d in interdep_net.G.nodes(data=True):
        demand_value = d['data']['inf_data'].demand
        if demand_value < 0:
            total_demand += demand_value
            total_demand_layer[n[1]] += demand_value

    for i in range(num_iterations+1):
        sum_run_time = 0.0
        sum_run_time_real = 0.0
        for cost_type in cost_types:
            sum_temp = 0.0
            sum_temp_real = 0.0
            if i == 0:
                sum_temp = indp_results_initial[1][0]['costs'][cost_type]
                sum_temp_real = indp_results_initial[1][0]['costs'][cost_type]
            else:
                for p in layers:
                    if cost_type != 'Under Supply Perc':
                        sum_temp += dindp_results[p][i]['costs'][cost_type]
                        sum_temp_real += dindp_results_real[p][i]['costs'][cost_type]
                    else:
                        factor = total_demand_layer[p]/total_demand
                        sum_temp += dindp_results[p][i]['costs'][cost_type]*factor
                        sum_temp_real += dindp_results_real[p][i]['costs'][cost_type]*factor

            dindp_results_sum.add_cost(i, cost_type, sum_temp)
            dindp_results_real_sum.add_cost(i, cost_type, sum_temp_real)
        for p in layers:
            if dindp_results[p][i]['run_time'] > sum_run_time:
                sum_run_time = dindp_results[p][i]['run_time']
            if dindp_results_real[p][i]['run_time'] > sum_run_time_real:
                sum_run_time_real = dindp_results_real[p][i]['run_time']
            for a in dindp_results[p][i]['actions']:
                dindp_results_sum.add_action(i, a)
        dindp_results_sum.add_run_time(i, sum_run_time)
        dindp_results_real_sum.add_run_time(i, sum_run_time_real)
    return dindp_results_sum, dindp_results_real_sum

def realized_performance(N, indp_results, functionality, layers, T=1,
                         controlled_layers=[1], print_cmd=False):
    '''
    This function computes the realized values of flow cost, unbalanced cost, and
    demand deficit at the end of each step according to the what the other agent
    actually decides (as opposed to according to the guess).

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
    controlled_layers : TYPE, optional
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
    g_prime_nodes = [n[0] for n in N.G.nodes(data=True) if n[1]['data']['inf_data'].net_id in layers]
    g_prime = N.G.subgraph(g_prime_nodes)
    interdep_nodes = {}
    for u, v, a in g_prime.edges(data=True):
        if a['data']['inf_data'].is_interdep and g_prime.nodes[v]['data']['inf_data'].net_id in controlled_layers:
            if u not in interdep_nodes:
                interdep_nodes[u] = []
            interdep_nodes[u].append(v)
    list_interdep_nodes = interdep_nodes.keys()
    functionality_realized = copy.deepcopy(functionality)
    realizations = {t:{} for t in range(T)}
    for t in range(T):
        real_count = 0
        for u, _ in functionality[t].items():
            if u in list_interdep_nodes:
                real_count += 1
                v_values = [g_prime.nodes[x]['data']['inf_data'].functionality for x in interdep_nodes[u]]
                realizations[t][real_count] = {'vNames':interdep_nodes[u], 'uName':u,
                            'uJudge':functionality[t][u], 'uCorrected':False, 'v_values':v_values}
                if functionality[t][u] == 1.0 and g_prime.nodes[u]['data']['inf_data'].functionality == 0.0:
                    functionality_realized[t][u] = 0.0
                    realizations[t][real_count]['uCorrected'] = True
                    if print_cmd:
                        print('Correct '+str(u)+' to 0 (affect. '+str(v_values)+')')
    indp_results_real = indp.indp(N, v_r=0, T=1, layers=layers, controlled_layers=controlled_layers,
                                  functionality=functionality_realized,
                                  print_cmd=print_cmd, time_limit=10*60)
    for t in range(T):
        costs = indp_results.results[t]['costs']
        node_cost = costs["Node"]
        indp_results_real[1][t]['costs']["Node"] = node_cost
        arc_cost = costs["Arc"]
        indp_results_real[1][t]['costs']["Arc"] = arc_cost
        space_prep_cost = costs["Space Prep"]
        indp_results_real[1][t]['costs']["Space Prep"] = space_prep_cost
        flow_cost = indp_results_real[1][t]['costs']["Flow"]
        over_supp_cost = indp_results_real[1][t]['costs']["Over Supply"]
        under_supp_cost = indp_results_real[1][t]['costs']["Under Supply"]
        # Calculate total costs.
        indp_results_real[1][t]['costs']["Total"] = flow_cost+arc_cost+node_cost+\
            over_supp_cost+under_supp_cost+space_prep_cost
        indp_results_real[1][t]["Total no disconnection"] = space_prep_cost+arc_cost+\
            flow_cost+node_cost
    return indp_results_real, realizations



def write_auction_csv(outdir, res_allocate, res_alloc_time, poa=None, valuations=None,
                      sample_num=1, suffix=""):
    '''
    This function write full results of auctions to file
    Parameters
    ----------
    outdir : str
        Where to write the file.
    res_allocate : TYPE
        DESCRIPTION.
    res_alloc_time : TYPE
        DESCRIPTION.
    poa : TYPE, optional
        DESCRIPTION. The default is None.
    valuations : TYPE, optional
        DESCRIPTION. The default is None.
    sample_num : TYPE, optional
        DESCRIPTION. The default is 1.
    suffix : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    None.

    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    auction_file = outdir+"/auctions_"+str(sample_num)+"_"+suffix+".csv"
    # Making header
    header = "t, "
    for key, value in res_allocate[0].items():
        header += "p"+str(key)+", "
    if valuations:
        header += "poa, optimal_val, winner_val, "
        for p, value in valuations[0].items():
            header += "bidder_"+str(p)+"_valuation, "
    header += "Res Alloc Time, Auction Time, "
    if valuations:
        for key, value in res_allocate[0].items():
            header += "Val. Time p"+str(key)+", "
    # Write to file
    with open(auction_file, 'w') as f:
        f.write(header+"\n")
        for t, value in res_allocate.items():
            row = str(t+1)+", "
            for p, pvalue in value.items():
                row += str(len(pvalue))+', '
            if valuations:
                row += str(poa[t]['poa'])+', '+str(poa[t]['optimal'])+', '
                for pitem in poa[t]['winner']:
                    row += str(pitem)+"|"
                row += ', '
                for p, pvalue in valuations[t].items():
                    for pitem in pvalue:
                        row += str(pitem)+"|"
                    row += ', '
            row += str(res_alloc_time[t][0])+', '+str(res_alloc_time[t][1])+', '
            if valuations:
                for titem in res_alloc_time[t][2]:
                    row += str(titem)+', '
            f.write(row+"\n")

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

def write_judgments_csv(outdir, realizations, sample_num=1, agent=1, time_step=0, suffix=""):
    '''

    Parameters
    ----------
    outdir : TYPE
        DESCRIPTION.
    realizations : TYPE
        DESCRIPTION.
    sample_num : TYPE, optional
        DESCRIPTION. The default is 1.
    agent : TYPE, optional
        DESCRIPTION. The default is 1.
    time_step : TYPE, optional
        DESCRIPTION. The default is 0.
    suffix : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    None.

    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    judge_file = outdir+'/judge_'+str(sample_num)+"_agent"+str(agent)+'_time'+\
    str(time_step)+'_'+suffix+".csv"
    header = "no., src node, src layer, src judge, if src corr., dest Names, dest init. funcs"
    with open(judge_file, 'w') as f:
        f.write(header+"\n")
        for t, timeValue in realizations.items():
            if timeValue:
                for c, Value in timeValue.items():
                    row = str(c)+', '+str(Value['uName'])+', '+str(Value['uJudge'])+\
                        ', '+str(Value['uCorrected'])+', '\
                        +str(Value['vNames'])+', '+str(Value['v_values'])
                    f.write(row+'\n')
            else:
                print('<><><> No judgment by agent '+str(agent)+' t:'+str(time_step)+\
                      ' step:'+str(t))

def read_results(combinations, optimal_combinations, suffixes, root_result_dir='../results/',
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
    columns = ['t', 'Magnitude', 'cost_type', 'decision_type', 'auction_type', 'valuation_type',
               'no_resources', 'sample', 'cost', 'normalized_cost', 'layer']
    optimal_method = ['tdindp', 'indp', 'sample_indp_12Node']
    cmplt_results = pd.DataFrame(columns=columns, dtype=int)
    print("\nAggregating Results")
    joinedlist = combinations + optimal_combinations
    for idx, x in enumerate(joinedlist):
        if x[4] in optimal_method:
            full_suffix = '_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])
        elif x[5] == 'Uniform':
            full_suffix = '_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])+'_uniform_alloc'
        else:
            full_suffix = '_L'+str(x[2])+'_m'+str(x[0])+'_v'+str(x[3])+'_auction_'+x[5]+'_'+x[6]
        result_dir = root_result_dir+x[4]+'_results'+full_suffix
        if os.path.exists(result_dir):
            # Save all results to Pandas dataframe
            sample_result = indputils.INDPResults()
            sam_rslt_lyr = {l+1:indputils.INDPResults() for l in range(x[2])}
            ### !!! Assume the layer name is l+1
            for suf in suffixes:
                file_dir = result_dir+"/costs_"+str(x[1])+"_"+suf+".csv"
                if os.path.exists(file_dir):
                    sample_result = sample_result.from_csv(result_dir, x[1], suffix=suf)
                if deaggregate:
                    for l in range(x[2]):
                        if x[4]=='indp':#!!!remove this
                            suffix_adj = suf.split('_')[0]+'l'+str(l+1)+'_'
                        else:
                            suffix_adj = suf.split('_')[0]+'_'+str(l+1)
                        file_dir = result_dir+rslt_dir_lyr+"/costs_"+str(x[1])+"_"+suffix_adj+".csv"
                        if os.path.exists(file_dir):
                            sam_rslt_lyr[l+1] = sam_rslt_lyr[l+1].from_csv(result_dir+rslt_dir_lyr,
                                                                           x[1], suffix=suffix_adj)
            initial_cost = {}
            for c in sample_result.cost_types:
                initial_cost[c] = sample_result[0]['costs'][c]
            norm_cost = 0
            for t in sample_result.results:
                for c in sample_result.cost_types:
                    if initial_cost[c] != 0.0:
                        norm_cost = sample_result[t]['costs'][c]/initial_cost[c]
                    else:
                        norm_cost = -1.0
                    values = [t, x[0], c, x[4], x[5], x[6], x[3], x[1],
                              float(sample_result[t]['costs'][c]), norm_cost, 'nan']
                    cmplt_results = cmplt_results.append(dict(zip(columns, values)),
                                                         ignore_index=True)
            if deaggregate:
                for l in range(x[2]):
                    initial_cost = {}
                    for c in sam_rslt_lyr[l+1].cost_types:
                        initial_cost[c] = sam_rslt_lyr[l+1][0]['costs'][c]
                    norm_cost = 0
                    for t in sam_rslt_lyr[l+1].results:
                        for c in sam_rslt_lyr[l+1].cost_types:
                            if initial_cost[c] != 0.0:
                                norm_cost = sam_rslt_lyr[l+1][t]['costs'][c]/initial_cost[c]
                            else:
                                norm_cost = -1.0
                            values = [t, x[0], c, x[4], x[5], x[6], x[3], x[1],
                                      float(sam_rslt_lyr[l+1][t]['costs'][c]), norm_cost, l+1]
                            cmplt_results = cmplt_results.append(dict(zip(columns, values)),
                                                                 ignore_index=True)
            if idx%(len(joinedlist)/10+1) == 0:
                update_progress(idx+1, len(joinedlist))
        else:
            sys.exit('Error: The combination or folder does not exist'+str(x))
    update_progress(len(joinedlist), len(joinedlist))
    return cmplt_results

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

def correct_tdindp_results(r_df, optimal_combinations):
    '''

    Parameters
    ----------
    r_df : TYPE
        DESCRIPTION.
    optimal_combinations : TYPE
        DESCRIPTION.

    Returns
    -------
    r_df : TYPE
        DESCRIPTION.

    '''
    # correct total cost of td-indp
    print('\nCorrecting td-INDP Results\n', end='')
    tVector = r_df['t'].unique().tolist()
    for t in tVector:
        for _, x in enumerate(optimal_combinations):
            if x[4] == 'tdindp':
                rows = r_df[(r_df['t'] == t)&(r_df['Magnitude'] == x[0])&
                            (r_df['decision_type'] == 'tdindp')&(r_df['no_resources'] == x[3])&
                            (r_df['sample'] == x[1])]
                if t != int(tVector[-1]) and t != 0:
                    rowsNext = r_df[(r_df['t'] == t+1)&(r_df['Magnitude'] == x[0])&
                                    (r_df['decision_type'] == 'tdindp')&
                                    (r_df['no_resources'] == x[3])&
                                    (r_df['sample'] == x[1])]
                    node_cost = rows[rows['cost_type'] == 'Node']['cost'].values
                    arc_cost = rows[rows['cost_type'] == 'Arc']['cost'].values
                    flow_cost = rowsNext[rowsNext['cost_type'] == 'Flow']['cost'].values
                    over_supp_cost = rowsNext[rowsNext['cost_type'] == 'Over Supply']['cost'].values
                    under_supp_cost = rowsNext[rowsNext['cost_type'] == 'Under Supply']['cost'].values
                    space_prep_cost = rows[rows['cost_type'] == 'Space Prep']['cost'].values
                    totalCost = flow_cost+arc_cost+node_cost+over_supp_cost+under_supp_cost+space_prep_cost
                    r_df.loc[(r_df['t'] == t)&(r_df['Magnitude'] == x[0])&
                             (r_df['decision_type'] == 'tdindp')&
                             (r_df['no_resources'] == x[3])&
                             (r_df['sample'] == x[1])&
                             (r_df['cost_type'] == 'Total'), 'cost'] = totalCost
                    initial_cost = r_df[(r_df['t'] == 0)&
                                        (r_df['Magnitude'] == x[0])&
                                        (r_df['decision_type'] == 'tdindp')&
                                        (r_df['no_resources'] == x[3])&
                                        (r_df['sample'] == x[1])&
                                        (r_df['cost_type'] == 'Total')]['cost'].values
                    r_df.loc[(r_df['t'] == t)&
                             (r_df['Magnitude'] == x[0])&
                             (r_df['decision_type'] == 'tdindp')&
                             (r_df['no_resources'] == x[3])&
                             (r_df['sample'] == x[1])&
                             (r_df['cost_type'] == 'Total'),
                             'normalized_cost'] = totalCost/initial_cost
        update_progress(t+1, len(tVector))
    return r_df

def relative_performance(r_df, combinations, optimal_combinations, ref_method='indp',
                         ref_at='', ref_vt='', cost_type='Total'):
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
    cost_type : TYPE, optional
        DESCRIPTION. The default is 'Total'.

    Returns
    -------
    lambda_df : TYPE
        DESCRIPTION.

    '''
    columns = ['Magnitude', 'cost_type', 'decision_type', 'auction_type', 'valuation_type',
               'no_resources', 'sample', 'Area_TC', 'Area_P', 'lambda_tc', 'lambda_p', 'lambda_U']
    T = len(r_df['t'].unique())
    lambda_df = pd.DataFrame(columns=columns, dtype=int)
    # Computing reference area for lambda
    # Check if the method in optimal combination is the reference method #!!!
    print('\nRef area calculation\n', end='')
    for idx, x in enumerate(optimal_combinations):
        if x[4] == ref_method:
            rows = r_df[(r_df['Magnitude'] == x[0])&(r_df['decision_type'] == ref_method)&
                        (r_df['sample'] == x[1])&(r_df['auction_type'] == ref_at)&
                        (r_df['valuation_type'] == ref_vt)&(r_df['no_resources'] == x[3])]
            if not rows.empty:
                area_tc = trapz_int(y=list(rows[rows['cost_type'] == cost_type].cost[:T]),
                                    x=list(rows[rows['cost_type'] == cost_type].t[:T]))
                area_p = -trapz_int(y=list(rows[rows['cost_type'] == 'Under Supply Perc'].cost[:T]),
                                    x=list(rows[rows['cost_type'] == 'Under Supply Perc'].t[:T]))
                values = [x[0], cost_type, x[4], ref_at, ref_vt, x[3], x[1], area_tc, area_p,
                          'nan', 'nan', 'nan']
                lambda_df = lambda_df.append(dict(zip(columns, values)), ignore_index=True)
            if idx%(len(optimal_combinations)/10+1) == 0:
                update_progress(idx+1, len(optimal_combinations))
    update_progress(len(optimal_combinations), len(optimal_combinations))
    # Computing areaa and lambda
    print('\nLambda calculation\n', end='')
    for idx, x in enumerate(combinations+optimal_combinations):
        if x[4] != ref_method:
            # Check if reference area exists
            cond = ((lambda_df['Magnitude'] == x[0])&(lambda_df['decision_type'] == ref_method)&
                    (lambda_df['auction_type'] == ref_at)&(lambda_df['valuation_type'] == ref_vt)&
                    (lambda_df['cost_type'] == cost_type)&(lambda_df['sample'] == x[1])&
                    (lambda_df['no_resources'] == x[3]))
            if not cond.any():
                sys.exit('Error:Reference type is not here! for %s m %d|resource %d' %(x[4], x[0], x[3]))
            ref_area_tc = float(lambda_df.loc[cond, 'Area_TC'])
            ref_area_P = float(lambda_df.loc[cond, 'Area_P'])
            rows = r_df[(r_df['Magnitude'] == x[0])&(r_df['decision_type'] == x[4])&
                        (r_df['sample'] == x[1])&(r_df['auction_type'] == x[5])&
                        (r_df['valuation_type'] == x[6])&(r_df['no_resources'] == x[3])]
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
                values = [x[0], cost_type, x[4], x[5], x[6], x[3], x[1], area_tc,
                          area_p, lambda_tc, lambda_p, (lambda_tc+lambda_p)/2]
                lambda_df = lambda_df.append(dict(zip(columns, values)), ignore_index=True)
            else:
                sys.exit('Error: No entry for %s %s %s m %d|resource %d, ...' %(x[4], x[5], x[6], x[0], x[3]))
        if idx%(len(combinations+optimal_combinations)/10+1) == 0:
            update_progress(idx+1, len(combinations+optimal_combinations))
    update_progress(idx+1, len(combinations+optimal_combinations))
    return lambda_df

def generate_combinations(database, mags, sample, layers, no_resources, decision_type,
                          auction_type, valuation_type, list_high_dam_add=None, synthetic_dir=None):
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
    auction_type : TYPE
        DESCRIPTION.
    valuation_type : TYPE
        DESCRIPTION.
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
            if list_high_dam_add is None or len(list_high_dam.loc[(list_high_dam.set == s) & (list_high_dam.sce == m)].index):
                for rc in no_resources:
                    for dt, at, vt in itertools.product(decision_type, auction_type, valuation_type):
                        if (dt in optimal_method) and [m, s, L, rc, dt, '', ''] not in optimal_combinations:
                            optimal_combinations.append([m, s, L, rc, dt, '', ''])
                        elif (dt not in optimal_method) and (at not in ['Uniform']):
                            combinations.append([m, s, L, rc, dt, at, vt])
                        elif (dt not in optimal_method) and (at in ['Uniform']):
                            combinations.append([m, s, L, rc, dt, at, ''])
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
                for dt, at, vt in itertools.product(decision_type, auction_type, valuation_type):
                    if (dt in optimal_method) and [m, s, L, rc, dt, '', ''] not in optimal_combinations:
                        optimal_combinations.append([m, s, L, rc, dt, '', ''])
                    elif (dt not in optimal_method) and (at not in ['Uniform']):
                        combinations.append([m, s, L, rc, dt, at, vt])
                    elif (dt not in optimal_method) and (at in ['Uniform']):
                        combinations.append([m, s, L, rc, dt, at, ''])
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
