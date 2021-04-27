'''
Functions that are used to run different types of games for the restoration of 
interdependent networks
'''
import os
import sys
import copy
import gameclasses
import indp
import dindputils
import pandas as pd
import numpy as np

def run_game(params, save_results=True, print_cmd=True, save_model=False, plot2D=False):
    '''
    Finds interdepndent restoration strategies using a decentralized hueristic,
    Judgment Call :cite:`Talebiyan2019c,Talebiyan2019`.

    Parameters
    ----------
    params : dict
         Global parameters, including number of iterations, game type, etc.
    save_results : bool, optional
        Should the results be written to file. The default is True.
    print_cmd : bool, optional
        If true, the results are printed to console. The default is True.
    plot2D : bool, optional
        Should the payoff matrix be plotted (only for 2-players games). The default is False.
    save_model : bool, optional
        Should the games and indp models to compute payoffs be written to file. The default is False.

    Returns
    -------
    :
        None

    '''
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"] = 1
    num_iterations = params["NUM_ITERATIONS"]
    # Creating game objects
    c = 0
    objs = {}
    params_copy = copy.deepcopy(params)  #!!! deepcopy
    for jc in params["JUDGMENT_TYPE"]:
        params_copy['JUDGMENT_TYPE'] = jc
        for rst in params["RES_ALLOC_TYPE"]:
            params_copy['RES_ALLOC_TYPE'] = rst
            if rst not in ["MDA", "MAA", "MCA"]:
                output_dir_full = params["OUTPUT_DIR"]+'_L'+str(len(params["L"]))+'_m'+\
                                str(params["MAGNITUDE"])+"_v"+str(params["V"])+'_'+jc+\
                                '_'+rst+'/actions_'+str(params["SIM_NUMBER"])+'_.csv'
                if os.path.exists(output_dir_full):
                    print('Game:',rst,'results are already there\n')
                else:
                    objs[c] = gameclasses.InfrastructureGame(params_copy)
                    c += 1
            else:
                for vt in params["VALUATION_TYPE"]:
                    params_copy['VALUATION_TYPE'] = vt
                    output_dir_full = params["OUTPUT_DIR"]+'_L'+str(len(params["L"]))+'_m'+\
                                    str(params["MAGNITUDE"])+"_v"+str(params["V"])+'_'+jc+\
                                    '_AUCTION_'+rst+'_'+vt+'/actions_'+\
                                    str(params["SIM_NUMBER"])+'_.csv'
                    if os.path.exists(output_dir_full):
                        print('Game:',rst,vt,'results are already there\n')
                    else:
                        objs[c] = gameclasses.InfrastructureGame(params_copy)
                        c += 1
    # Run games
    if not objs:
        return 0
    # t=0 costs and performance.
    indp_results_initial = indp.indp(objs[0].net, 0, 1, objs[0].layers,
                                     controlled_layers=objs[0].layers)
    for _, obj in objs.items():
        print('--Running Game: '+obj.game_type+', resource allocation: '+obj.res_alloc_type)
        if obj.resource.type == 'AUCTION':
            print('auction type: '+obj.resource.auction_model.auction_type+\
                  ', valuation: '+obj.resource.auction_model.valuation_type)
        if print_cmd:
            print("Num iters=", params["NUM_ITERATIONS"])
        # t=0 results.
        obj.results = copy.deepcopy(indp_results_initial[1]) #!!! deepcopy
        # Run game
        obj.run_game(compute_optimal=True, plot=plot2D, save_results=save_results,
                     print_cmd=print_cmd, save_model=save_model)

def analyze_NE(objs, combinations, optimal_combinations):
    '''
    This function reads the results of analyses (INDP, JC, etc.) and the corresponding
    objects from file and aggregates the results in a dictionary.

    Parameters
    ----------
    objs : dict
        Dictionary that contains the objects corresponding to the read results.
    combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the JC (or any other decentralized results) collected by
        :func:`generate_combinations`.
    optimal_combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the INDP (or any other optimal results) collected by :func:`generate_combinations`.

    Returns
    -------
    
    '''
    columns = ['t', 'Magnitude', 'decision_type', 'judgment_type', 'auction_type',
               'valuation_type', 'no_resources', 'sample',
               'ne_total_cost', 'optimal_total_cost', 'action_similarity', 'payoff_ratio',
               'total_cost_ratio', 'no_ne', 'no_payoffs', 'cooperative',
               'partially_cooperative', 'OA', 'NA', 'NA_possible', 'opt_cooperative',
               'opt_partially_cooperative', 'opt_OA', 'opt_NA', 'opt_NA_possible']
    cmplt_analyze = pd.DataFrame(columns=columns, dtype=int)
    print("\nAnalyze NE")
    joinedlist = combinations + optimal_combinations
    for idx, x in enumerate(combinations):
        if x[4][:2] in ['ng', 'bg']:
            obj = objs[str(x)]
            for t in range(obj.time_steps):
                game = obj.objs[t+1]
                optimal_sol = game.optimal_solution
                ne_sol = game.chosen_equilibrium
                lyr_act, opt_tc, ne_payoff, ne_tc = compare_sol(optimal_sol, ne_sol, obj.layers)
                payoff_ratio = ne_payoff/opt_tc
                tc_ratio = ne_tc/opt_tc
                cooperation= {'C':0, 'P':0, 'OA':0, 'NA':0, 'NAP':0}
                cooperation_opt= {'C':0, 'P':0, 'OA':0, 'NA':0, 'NAP':0}
                no_ne = 0
                fa = [game.actions[l][0][0] for l in game.players]
                if set(fa) != {'NA'}:
                    no_ne = len(game.solution.sol.keys())
                    no_payoffs = {l:len(game.actions[l]) for l in game.players}
                    for _, val in game.solution.sol.items():
                        for idxl, l in enumerate(game.players):
                            action_key = 'P'+str(l)+' actions'
                            if x[4][:2] == 'bg':
                                action_key = bayesian_actions_relable(x[4], l, idxl)+' actions'
                            label = label_action(val[action_key][0], game.actions[l])
                            if not label:
                                sys.exit('Type of action cannot be found')
                            cooperation[label] += 1/len(game.players)/len(game.solution.sol.keys())
                    for l in game.players:
                        label = label_action(game.optimal_solution['P'+str(l)+' actions'],
                                             game.actions[l])
                        if not label:
                            sys.exit('Type of action cannot be found')
                        cooperation_opt[label] += 1/len(game.players)
                values = [t+1, x[0], x[4], x[5], x[6], x[7], x[3], x[1], ne_payoff,
                          opt_tc, lyr_act, payoff_ratio, tc_ratio, no_ne, no_payoffs,
                          cooperation['C'], cooperation['P'], cooperation['OA'],
                          cooperation['NA'], cooperation['NAP'], cooperation_opt['C'],
                          cooperation_opt['P'], cooperation_opt['OA'],
                          cooperation_opt['NA'], cooperation_opt['NAP']]
                cmplt_analyze = cmplt_analyze.append(dict(zip(columns, values)), ignore_index=True)
            if idx%(len(combinations)//100+1) == 0:
                dindputils.update_progress(idx+1, len(combinations))
        else:
            sys.exit('Error: The combination or folder does not exist'+str(x))
    dindputils.update_progress(len(combinations), len(combinations))
    return cmplt_analyze

def relative_actions(df, combinations):
    '''
    This functions computes the relative measure of % of player that take each type of
    action compared to the optimal solution.

    Parameters
    ----------
    df : dict
        Dictionary that contains complete results by JC and INDP collected by
        :func:`read_results`.
    combinations : dict
        All combinations of magnitude, sample, judgment type, resource allocation type
        involved in the JC (or any other decentralized results) collected by
        :func:`generate_combinations`.
    Returns
    -------
    df_rel : dict
        Dictionary that contains ...
    '''
    act_types = ['cooperative', 'partially_cooperative', 'OA', 'NA', 'NA_possible']
    cols = ['decision_type', 'judgment_type', 'auction_type', 'valuation_type', 'sample',
            'Magnitude', 'no_resources']+['rel_'+ac for ac in act_types]
    T = max(df.t.unique().tolist())
    df_rel = pd.DataFrame(columns=cols)
    print('\nRelative Actions')
    for idx, x in enumerate(combinations):
        rel_dict = {'decision_type':x[4], 'judgment_type':x[5], 'auction_type':x[6],
                    'valuation_type':x[7], 'sample':x[1], 'Magnitude':x[0], 'no_resources':x[3]}
        row = (df['decision_type'] == x[4])&(df['sample'] == x[1])&(df['Magnitude'] == x[0])&\
        (df['no_resources'] == x[3])&(df['auction_type'] == x[6])&\
        (df['valuation_type'] == x[7])&(df['judgment_type'] == x[5])
        vec_act = {ac:np.zeros(T) for ac in act_types}
        vec_act_optimal = {ac:np.zeros(T) for ac in act_types}
        for ac in act_types:
            for t in range(T):
                vec_act[ac][t] = df.loc[(df['t'] == t+1)&row, ac]
                vec_act_optimal[ac][t] = df.loc[(df['t'] == t+1)&row, 'opt_'+ac]
            # # Area between
            distance = sum(vec_act[ac]-vec_act_optimal[ac])
            rel_dict['rel_'+ac] = distance
        df_rel = df_rel.append(rel_dict, ignore_index=True)
        if idx%(len(combinations)/10+1) == 0:
            dindputils.update_progress(idx+1, len(combinations))
    dindputils.update_progress(idx+1, len(combinations))
    return df_rel

def compare_sol(opt, ne, layers):
    sum_lyr_act = 0
    if opt:
        for idxl, l in enumerate(layers):
            if opt['P'+str(l)+' actions'] == ne['solution combination'][0][idxl]:
                sum_lyr_act += 1/len(layers)
        opt_total_cost = opt['full result'].results[0]['costs']['Total']
        ne_total_payoff = ne['total cost']
        if 'full results' in ne.keys():
            ne_total_cost = ne['full results'][0].results[0]['costs']['Total']
        else:
            ne_total_cost = ne['full result'][0].results[0]['costs']['Total']
    elif not opt and not ne:
        sum_lyr_act = 1
        opt_total_cost = 1
        ne_total_payoff = 1
        ne_total_cost = 1
    else:
        sys.exit('No optimal results')
    return sum_lyr_act, opt_total_cost, ne_total_payoff, ne_total_cost

def label_action(action, all_actions):
    label = None
    if len(action) == 1 and action[0][0] == 'OA':
        label = 'OA'
    elif action[0] == 'NA':
        if len(all_actions)!=1:
            label = 'NA'
        else:
            label = 'NAP' # No more actions possinble either becasue the sustems is repaired completely or Rc=0
    else:
        label = 'C'
        for a in action:
            if a[0] == 'OA':
                label = 'P'
                break
    return label

def bayesian_actions_relable(method, layer, layer_idx):
    signal = method[2+layer_idx]
    return 'P(\''+signal+'\', '+str(layer)+')'