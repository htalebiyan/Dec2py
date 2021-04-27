# %%
""" Functions that are used to run different types of analysis on the restoration of 
interdependent networks
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import indp
import dindputils
import plots
import gametree
import itertools
# import Metaheuristics.metaheuristics as mh
import gameutils 

# %%
def batch_run(params, fail_sce_param):
    '''
    Batch run different methods for a given list of damage scenarios,
    given global parameters.

    Parameters
    ----------
    params : dict
        DESCRIPTION.
    fail_sce_param : dict
        DESCRIPTION.

    Returns
    -------
    None. Writes to file

    '''
    # Set root directories
    base_dir = fail_sce_param['BASE_DIR']
    damage_dir = fail_sce_param['DAMAGE_DIR']
    topology = None
    infrastructure_data = None
    ext_interdependency = None
    if fail_sce_param['TYPE'] == 'Andres':
        infrastructure_data = 'shelby_old'
        ext_interdependency = "../data/INDP_4-12-2016"
    elif fail_sce_param['TYPE'] == 'WU':
        infrastructure_data = 'shelby_extended'
        if fail_sce_param['FILTER_SCE'] is not None:
            list_high_dam = pd.read_csv(fail_sce_param['FILTER_SCE'])
    elif fail_sce_param['TYPE'] == 'random':
        infrastructure_data = 'shelby_extended'
    elif fail_sce_param['TYPE'] == 'synthetic':
        topology = fail_sce_param['TOPO']

    print('----Running for resources: '+str(params['V']))
    for m in fail_sce_param['MAGS']:
        for i in fail_sce_param['SAMPLE_RANGE']:
            try:
                list_high_dam
                if len(list_high_dam.loc[(list_high_dam.set == i)&\
                                         (list_high_dam.sce == m)].index) == 0:
                    continue
            except NameError:
                pass

            print('---Running Magnitude '+str(m)+' sample '+str(i)+'...')
            print("Initializing network...")
            if infrastructure_data:
                params["N"], _, _ = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            sim_number=0, magnitude=6, sample=0, v=params["V"],
                            infrastructure_data=infrastructure_data)
            else:
                params["N"], params["V"], params['L'] = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            magnitude=m, sample=i, infrastructure_data=infrastructure_data,
                            topology=topology)
            params["SIM_NUMBER"] = i
            params["MAGNITUDE"] = m
            # Check if the results exist
            output_dir_full = ''
            if params["ALGORITHM"] in ["INDP"]:
                output_dir_full = params["OUTPUT_DIR"]+'_L'+str(len(params["L"]))+'_m'+\
                    str(params["MAGNITUDE"])+"_v"+str(params["V"])+'/actions_'+str(i)+'_.csv'
            if os.path.exists(output_dir_full):
                print('results are already there\n')
                continue

            if fail_sce_param['TYPE'] == 'WU':
                indp.add_Wu_failure_scenario(params["N"], DAM_DIR=damage_dir,
                                             noSet=i, noSce=m)
            elif fail_sce_param['TYPE'] == 'ANDRES':
                indp.add_failure_scenario(params["N"], DAM_DIR=damage_dir,
                                          magnitude=m, v=params["V"], sim_number=i)
            elif fail_sce_param['TYPE'] == 'random':
                indp.add_random_failure_scenario(params["N"], DAM_DIR=damage_dir,
                                                 sample=i)
            elif fail_sce_param['TYPE'] == 'synthetic':
                indp.add_synthetic_failure_scenario(params["N"], DAM_DIR=base_dir,
                                                    topology=topology, config=m, sample=i)

            if params["ALGORITHM"] == "INDP":
                indp.run_indp(params, validate=False, T=params["T"], layers=params['L'],
                              controlled_layers=params['L'], saveModel=False, print_cmd_line=False,
                              co_location=False)
            if params["ALGORITHM"] == "MH":
                mh.run_mh(params, validate=False, T=params["T"], layers=params['L'],
                          controlled_layers=params['L'], saveModel=True, print_cmd_line=False,
                          co_location=True)
            elif params["ALGORITHM"] == "INFO_SHARE":
                indp.run_info_share(params, layers=params['L'], T=params["T"])
            elif params["ALGORITHM"] == "INRG":
                indp.run_inrg(params, layers=params['L'], player_ordering=player_ordering)
            elif params["ALGORITHM"] == "BACKWARDS_INDUCTION":
                gametree.run_backwards_induction(params["N"], i, players=params['L'],
                                                 player_ordering=player_ordering,
                                                 T=params["T"], outdir=params["OUTPUT_DIR"])
            elif params["ALGORITHM"] == "JC":
                dindputils.run_judgment_call(params, save_jc_model=False, print_cmd=False)
            elif params["ALGORITHM"] in ["NORMALGAME", "BAYESGAME"]:
                gameutils.run_game(params, save_results=True, print_cmd=False,
                                    save_model=False, plot2D=False) #!!!

def run_method(fail_sce_param, v_r, layers, method, judgment_type=None,
               res_alloc_type=None, valuation_type=None, output_dir='..', misc =None):
    '''
    This function runs a given method for different numbers of resources,
    and a given judge, auction, and valuation type in the case of JC.

    Parameters
    ----------
    fail_sce_param : dict
        informaton of damage scenrios.
    v_r : float, list of float, or list of lists of floats
        number of resources,
        if this is a list of floats, each float is interpreted as a different total
        number of resources, and indp is run given the total number of resources.
        It only works when auction_type != None.
        If this is a list of lists of floats, each list is interpreted as fixed upper
        bounds on the number of resources each layer can use (same for all time step)..
    layers : TYPE
        DESCRIPTION.
    method : TYPE
        DESCRIPTION.
    judgment_type : str, optional
        Type of Judgments in Judfment Call Method. The default is None.
    res_alloc_type : str, optional
        Type of resource allocation method for resource allocation. The default is None.
    valuation_type : str, optional
        Type of valuation in auction. The default is None.
    output_dir : str, optional
        DESCRIPTION. The default is '..'.
    Returns
    -------
    None.  Writes to file

    '''
    for v in v_r:
        if method == 'INDP':
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":output_dir+'indp_results',
                      "V":v, "T":1, 'L':layers, "ALGORITHM":"INDP"}
        elif method == 'TDINDP':
            params = {"OUTPUT_DIR":output_dir+'/tdindp_results', "V":v, "T":10,
                      'L':layers, "ALGORITHM":"INDP"} # "WINDOW_LENGTH":3, 
        elif method == 'JC':
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":output_dir+'jc_results',
                      "V":v, "T":1, 'L':layers, "ALGORITHM":"JC",
                      "JUDGMENT_TYPE":judgment_type, "RES_ALLOC_TYPE":res_alloc_type,
                      "VALUATION_TYPE":valuation_type}
            if 'STM' in valuation_type:
                params['STM_MODEL_DICT'] = misc['STM_MODEL']
        elif method in ['NORMALGAME', 'BAYESGAME']:
            if method == "NORMALGAME":
                out_dir = output_dir+'ng_results'
            elif method == "BAYESGAME":
                out_dir = output_dir+'bg'+''.join(misc['SIGNALS'].values())+\
                    ''.join(misc['BELIEFS'].values())+'_results'
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":out_dir,
                      "V":v, "T":1, "L":layers, "ALGORITHM":method,
                      'EQUIBALG':'enumerate_pure', "JUDGMENT_TYPE":judgment_type,
                      "RES_ALLOC_TYPE":res_alloc_type, "VALUATION_TYPE":valuation_type}
            if misc:
                params['REDUCED_ACTIONS'] = misc['REDUCED_ACTIONS']
                params['PAYOFF_DIR'] = misc['PAYOFF_DIR']
                if params['PAYOFF_DIR']:
                    params['PAYOFF_DIR'] += 'ng_results'
            if method == 'BAYESGAME':
                params["SIGNALS"] = misc['SIGNALS']
                params["BELIEFS"] = misc['BELIEFS']
            else:
                params["SIGNALS"] = None
                params["BELIEFS"] = None
        else:
            sys.exit('Wrong method name: '+method)

        params['DYNAMIC_PARAMS'] = misc['DYNAMIC_PARAMS']
        if misc['DYNAMIC_PARAMS']:
            prefix = params['OUTPUT_DIR'].split('/')[-1]
            params['OUTPUT_DIR'] = params['OUTPUT_DIR'].replace(prefix,'dp_'+prefix)

        batch_run(params, fail_sce_param)

def run_indp_sample(layers):
    interdep_net= indp.initialize_sample_network(layers=layers)
    params={"NUM_ITERATIONS":7, "OUTPUT_DIR":'../results/indp_sample_12Node_results',
            "V":len(layers), "T":1, "L":layers, "WINDOW_LENGTH":1, "ALGORITHM":"INDP",
            "N":interdep_net, "MAGNITUDE":0, "SIM_NUMBER":0, 'DYNAMIC_PARAMS':None}
    indp.run_indp(params, layers=layers, T=params["T"], suffix="", saveModel=True,
              print_cmd_line=True)
    print('\n\nPlot restoration plan by INDP')
    indp.plot_indp_sample(params)
    plt.show()

def run_tdindp_sample(layers):
    interdep_net= indp.initialize_sample_network(layers=layers)
    params={"OUTPUT_DIR":'../results/tdindp_sample_12Node_results', "V":len(layers),
            "T":7, "L":layers, "ALGORITHM":"INDP", "WINDOW_LENGTH":3, 
            "N":interdep_net, "MAGNITUDE":0, "SIM_NUMBER":0} #"WINDOW_LENGTH":6, 
    indp.run_indp(params, layers=layers, T=params["T"], suffix="", saveModel=True,
              print_cmd_line=True)
    print('\n\nPlot restoration plan by INDP')
    indp.plot_indp_sample(params)
    plt.show()
    
def run_jc_sample(layers, judge_types, auction_type, valuation_type):
    interdep_net=indp.initialize_sample_network(layers=layers)
    params={"NUM_ITERATIONS":7, "OUTPUT_DIR":'../results/jc_sample_12Node_results',
            "V":len(layers), "T":1, "L":layers, "WINDOW_LENGTH":1, "ALGORITHM":"JC",
            "N":interdep_net, "MAGNITUDE":0, "SIM_NUMBER":0,
            "JUDGMENT_TYPE":judge_types, "RES_ALLOC_TYPE":auction_type,
            "VALUATION_TYPE":valuation_type}
    dindputils.run_judgment_call(params, save_jc_model=True, print_cmd=False)
    for jt, rst, vt in itertools.product(judge_types, auction_type, valuation_type):
        print('\n\nPlot restoration plan by JC',jt,rst,vt)
        if rst == 'UNIFORM':
            indp.plot_indp_sample(params, folderSuffix='_'+jt+'_'+rst, suffix="real")
        else:
            indp.plot_indp_sample(params, folderSuffix='_'+jt+'_AUCTION_'+rst+'_'+vt, suffix="real")
        plt.show()

def run_game_sample(layers, judge_types, auction_type, valuation_type,
                    game_type="NORMALGAME", signals=None, beliefs=None, reduced_act=None):
    interdep_net= indp.initialize_sample_network(layers=layers)
    if game_type == "NORMALGAME":
        out_dir = '../results/ng_sample_12Node_results'
    elif game_type == "BAYESGAME":
        out_dir = '../results/bg'+''.join(signals.values())+''.join(beliefs.values())+\
            '_sample_12Node_results'
    params={"NUM_ITERATIONS":7, "OUTPUT_DIR":out_dir, "V":len(layers)*2, "T":1, "L":layers,
            "WINDOW_LENGTH":1, "ALGORITHM":game_type, 'EQUIBALG':'enumerate_pure',
            "N":interdep_net, "MAGNITUDE":0, "SIM_NUMBER":0, "JUDGMENT_TYPE":judge_types,
            "RES_ALLOC_TYPE":auction_type, "VALUATION_TYPE":valuation_type, 'DYNAMIC_PARAMS':None,
            'PAYOFF_DIR':None, "SIGNALS":signals, "BELIEFS":beliefs, 'REDUCED_ACTIONS':reduced_act}
    gameutils.run_game(params, save_results=True, print_cmd=True, save_model=True, plot2D=True)
    for jt, rst, vt in itertools.product(judge_types, auction_type, valuation_type):
        print('\n\nPlot restoration plan by Game',jt,rst,vt)
        if rst == 'UNIFORM' or 'FIXED_LAYER':
            indp.plot_indp_sample(params, folderSuffix='_'+jt+'_'+rst, suffix="")
        else:
            indp.plot_indp_sample(params, folderSuffix='_'+jt+'_AUCTION_'+rst+'_'+vt, suffix="")
        plt.show()

def run_mh_sample(layers):
    interdep_net= indp.initialize_sample_network(layers=layers)
    params={"NUM_ITERATIONS":1, "OUTPUT_DIR":'../results/mh_sample_12Node_results',
            "V":len(layers), "T":1, "L":layers, "WINDOW_LENGTH":1, "ALGORITHM":"MH",
            "N":interdep_net, "MAGNITUDE":0, "SIM_NUMBER":0}
    result_mh = mh.run_mh(params, layers=layers, T=params["T"], suffix="", saveModel=True,
              print_cmd_line=True)
    return result_mh #!!!
    # print('\n\nPlot restoration plan by INDP')
    # indp.plot_indp_sample(params)
    # plt.show()
    pass

def run_sample_problems(): 
    layers=[1,2]#,3]
    auction_type = ['UNIFORM']#"MCA", "MAA", "MDA", "LAYER_FIXED"
    valuation_type = ["DTC"]
    judge_types = ["OPTIMISTIC"]#"PESSIMISTIC",
    # run_indp_sample(layers)
    # run_tdindp_sample(layers)
    # run_jc_sample(layers, judge_types, auction_type, valuation_type)
    run_game_sample(layers, judge_types, auction_type, valuation_type,
                    game_type="NORMALGAME", reduced_act='EDM')
    # run_game_sample(layers, judge_types, auction_type, valuation_type, game_type="BAYESGAME",
    #                 beliefs={1:'I', 2:'I'}, signals={1:'N', 2:'C'})
    # # result_mh = run_mh_sample(layers) #!!!

    # with open('../results/ng_sample_12Node_results_L2_m0_v4_OPTIMISTIC_UNIFORM/objs_0.pkl', 'rb') as f:
        # obj = pickle.load(f)
    # COMBS = []
    # OPTIMAL_COMBS = [[0, 0, len(layers), len(layers), 'indp_sample_12Node', 'nan',
    #                   'nan', 'nan', ''],
    #                   [0, 0, len(layers), len(layers), 'tdindp_sample_12Node', 'nan',
    #                   'nan', 'nan', '']]
    # for jt, rst, vt in itertools.product(judge_types, auction_type, valuation_type):
    #     if rst == 'UNIFORM':
    #         COMBS.append([0, 0, len(layers), len(layers), 'jc_sample_12Node', jt, rst, 'nan', 'real'])
    #         COMBS.append([0, 0, len(layers), len(layers), 'ng_sample_12Node', jt, rst, 'nan', ''])
    #     else:
    #         COMBS.append([0, 0, len(layers), len(layers), 'jc_sample_12Node', jt, rst, vt, 'real'])
    #         COMBS.append([0, 0, len(layers), len(layers), 'ng_sample_12Node', jt, rst, vt, ''])
    # BASE_DF, objs = dindputils.read_results(COMBS, OPTIMAL_COMBS, ['Total'],
    #                                     root_result_dir='../results/', deaggregate=True)
    # LAMBDA_DF = dindputils.relative_performance(BASE_DF, COMBS, OPTIMAL_COMBS,
    #                                         ref_method='indp_sample_12Node', cost_type='Total')
    # RES_ALLOC_DF, ALLOC_GAP_DF = dindputils.read_resourcec_allocation(BASE_DF, COMBS, OPTIMAL_COMBS,
    #                                                               objs, root_result_dir='../results/',
    #                                                               ref_method='indp_sample_12Node')
    # plots.plot_performance_curves(BASE_DF, cost_type='Total', ci=None,
    #                               deaggregate=True, plot_resilience=True)
    # plots.plot_relative_performance(LAMBDA_DF, lambda_type='U')
    # plots.plot_auction_allocation(RES_ALLOC_DF, ci=None)
    # plots.plot_relative_allocation(ALLOC_GAP_DF, distance_type='gap')
