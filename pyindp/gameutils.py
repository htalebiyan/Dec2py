'''
This module contains functions to run decentralized restoration for interdepndent networks
using Judgment Call method :cite:`Talebiyan2019c,Talebiyan2019`, read the results,
and compute comparison measures.
'''
import os
import copy
import gameclasses
import indp

def run_game(params, save_game=True, print_cmd=True, save_payoff=False, plot2D=False):
    '''
    Finds interdepndent restoration strategies using a decentralized hueristic,
    Judgment Call :cite:`Talebiyan2019c,Talebiyan2019`.

    Parameters
    ----------
    params : dict
         Global parameters, including number of iterations, game type, etc.
    save_game : bool, optional
        Should the results and game be written to file. The default is True.
    print_cmd : bool, optional
        If true, the results are printed to console. The default is True.
    plot2D : bool, optional
        Should the payoff matrix be plotted (only for 2-players games). The default is False.
    save_payoff : bool, optional
        Should the indp modles to compute payoffs be written to file. The default is False.

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
        obj.run_game(compute_optimal=plot2D, plot=plot2D, save_results=save_game,
                     print_cmd=print_cmd, save_payoff=save_payoff)