""" Runs INDP, td-INDP, Judgment Call, and infrastructure games"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import indp
import dindputils
import plots
import gametree
import itertools
import Metaheuristics.metaheuristics as mh
# import gameutils

try:
    # Change the current working Directory
    DIR_MAIN = 'C:/Users/ht20/Documents/GitHub/td-DINDP/pyindp'
    os.chdir(DIR_MAIN)
    print("Directory changed to "+DIR_MAIN)
except OSError:
    print("Can't change the Current Working Directory")

def batch_run(params, fail_sce_param, player_ordering=[3, 1]):
    '''
    Batch run different methods for a given list of damage scenarios,
        given global parameters.

    Parameters
    ----------
    params : dict
        DESCRIPTION.
    fail_sce_param : dict
        DESCRIPTION.
    player_ordering : list, optional
        DESCRIPTION. The default is [3, 1].

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
            params["SIM_NUMBER"] = i
            params["MAGNITUDE"] = m
            try:
                list_high_dam
                if len(list_high_dam.loc[(list_high_dam.set == i)&\
                                         (list_high_dam.sce == m)].index) == 0:
                    continue
            except NameError:
                pass

            # Check if the results exist 
            #!!! move it after initializing network for synthetic nets since L is identified there
            output_dir_full = ''
            if params["ALGORITHM"] in ["INDP"]:
                outDirSuffixRes = indp.get_resource_suffix(params)
                output_dir_full = params["OUTPUT_DIR"]+'_L'+str(len(params["L"]))+'_m'+\
                    str(params["MAGNITUDE"])+"_v"+outDirSuffixRes+'/actions_'+str(i)+'_.csv'
            if os.path.exists(output_dir_full):
                print('results are already there\n')
                continue

            print('---Running Magnitude '+str(m)+' sample '+str(i)+'...')
            if params['TIME_RESOURCE']:
                indp.time_resource_usage_curves(base_dir, damage_dir, i)
            print("Initializing network...")
            if infrastructure_data:
                params["N"], _, _ = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            sim_number=0, magnitude=m, sample=i, v=params["V"],
                            infrastructure_data=infrastructure_data,
                            extra_commodity=params["EXTRA_COMMODITY"])
            else:
                params["N"], params["V"], params['L'] = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            magnitude=m, sample=i, infrastructure_data=infrastructure_data,
                            topology=topology)

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
                              controlled_layers=params['L'], saveModel=True,
                              print_cmd_line=True, co_location=True)
            if params["ALGORITHM"] == "MH":
                mh.run_mh(params, validate=False, T=params["T"], layers=params['L'],
                          controlled_layers=params['L'], saveModel=True,
                          print_cmd_line=False, co_location=True)
            elif params["ALGORITHM"] == "INFO_SHARE":
                indp.run_info_share(params, layers=params['L'], T=params["T"])
            elif params["ALGORITHM"] == "INRG":
                indp.run_inrg(params, layers=params['L'], player_ordering=player_ordering)
            elif params["ALGORITHM"] == "BACKWARDS_INDUCTION":
                gametree.run_backwards_induction(params["N"], i, players=params['L'],
                                                 player_ordering=player_ordering,
                                                 T=params["T"], outdir=params["OUTPUT_DIR"])
            elif params["ALGORITHM"] == "JC":
                dindputils.run_judgment_call(params, save_jc_model=True, print_cmd=False)
            elif params["ALGORITHM"] in ["NORMALGAME", "BAYESGAME"]:
                gameutils.run_game(params, save_results=True, print_cmd=False,
                                    save_model=False, plot2D=False) #!!!

def run_indp_sample(layers):
    interdep_net= indp.initialize_sample_network(layers=layers)
    params={"NUM_ITERATIONS":7, "OUTPUT_DIR":'../results/indp_sample_12Node_results',
            "V":len(layers), "T":1, "L":layers, "WINDOW_LENGTH":1, "ALGORITHM":"INDP",
            "N":interdep_net, "MAGNITUDE":0, "SIM_NUMBER":0}
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
                    game_type="NORMALGAME", signals=None, beliefs=None):
    interdep_net= indp.initialize_sample_network(layers=layers)
    if game_type == "NORMALGAME":
        out_dir = '../results/ng_sample_12Node_results'
    elif game_type == "BAYESGAME":
        out_dir = '../results/bg'+''.join(signals.values())+''.join(beliefs.values())+\
            '_sample_12Node_results'
    params={"NUM_ITERATIONS":7, "OUTPUT_DIR":out_dir, "V":1+len(layers), "T":1, "L":layers,
            "WINDOW_LENGTH":1, "ALGORITHM":game_type, 'EQUIBALG':'enumerate_pure',
            "N":interdep_net, "MAGNITUDE":0, "SIM_NUMBER":0, "JUDGMENT_TYPE":judge_types,
            "RES_ALLOC_TYPE":auction_type, "VALUATION_TYPE":valuation_type, 'PAYOFF_DIR':None,
            "SIGNALS":signals, "BELIEFS":beliefs}
    gameutils.run_game(params, save_results=True, print_cmd=True, save_model=True, plot2D=True)
    for jt, rst, vt in itertools.product(judge_types, auction_type, valuation_type):
        print('\n\nPlot restoration plan by Game',jt,rst,vt)
        if rst == 'UNIFORM':
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
            params = {"NUM_ITERATIONS":20, "OUTPUT_DIR":output_dir+'indp_results',
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
                output_dir += 'ng_results'
            elif method == "BAYESGAME":
                output_dir += 'bg'+''.join(misc['SIGNALS'].values())+\
                    ''.join(misc['BELIEFS'].values())+'_results'
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":output_dir,
                      "V":v, "T":1, "L":layers, "ALGORITHM":method,
                      'EQUIBALG':'enumerate_pure', "JUDGMENT_TYPE":judgment_type,
                      "RES_ALLOC_TYPE":res_alloc_type, "VALUATION_TYPE":valuation_type}
            if misc:
                params['PAYOFF_DIR'] = misc['PAYOFF_DIR']
                if params['PAYOFF_DIR']:
                    params['PAYOFF_DIR'] += 'ng_results'
            if method == 'BAYESGAME':
                params["SIGNALS"] = misc['SIGNALS']
                params["BELIEFS"] = misc['BELIEFS']
        else:
            sys.exit('Wrong method name: '+method)

        params['EXTRA_COMMODITY'] = misc['EXTRA_COMMODITY']
        params['TIME_RESOURCE'] = misc['TIME_RESOURCE']
        params['DYNAMIC_PARAMS'] = misc['DYNAMIC_PARAMS']
        if misc['DYNAMIC_PARAMS']:
            prefix = params['OUTPUT_DIR'].split('/')[-1]
            params['OUTPUT_DIR'] = params['OUTPUT_DIR'].replace(prefix,'dp_'+prefix)

        batch_run(params, fail_sce_param)

def run_toy_examples():
    ''' Run a toy example for different methods '''
    plt.close('all')
    layers=[1,2]#,3]
    auction_type = ["MCA", "UNIFORM"]#, "MAA", "MDA"
    valuation_type = ["DTC"]
    judge_types = ["OPTIMISTIC"]#"PESSIMISTIC",
    run_indp_sample(layers)
    run_tdindp_sample(layers)
    # run_jc_sample(layers, judge_types, auction_type, valuation_type)
    run_game_sample(layers, judge_types, auction_type, valuation_type, game_type="NORMALGAME")
    run_game_sample(layers, judge_types, auction_type, valuation_type, game_type="BAYESGAME",
                    beliefs={1:'U', 2:'U'}, signals={1:'N', 2:'C'})
    result_mh = run_mh_sample(layers) #!!!

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
    pass
        
if __name__ == "__main__":
    ''' Run a toy example for different methods '''
    # run_toy_examples()

    ''' Set analysis directories '''
    #: The address to the list of scenarios that should be included in the analyses.
    FILTER_SCE = None
    # 'C:/Users/ht20/Box Sync/Shelby County Database/Damage_scenarios/damagedElements_sliceQuantile_0.95.csv'
    # 'C:/Users/ht20/Box Sync/Shelby County Database/damagedElements_sliceQuantile_0.90.csv'
    # '../data/damagedElements_sliceQuantile_0.90.csv'
    
    #: The address to the basic (topology, parameters, etc.) information of the network.
    BASE_DIR = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Node_arc_info/"
    # "../data/Extended_Shelby_County/"
    # 'C:/Users/ht20/Box Sync/Shelby County Database/Node_arc_info/'
    # "C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\"
    # "/home/hesam/Desktop/Files/Generated_Network_Dataset_v3.1"
    # "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Joplin/Node_arc_info/"
    # "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Node_arc_info/"
    #: The address to damge scenario data.
    DAMAGE_DIR = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Damage_scenarios/eq_100yr_dmg/"
    # ../data/random_disruption_shelby/"
    #"../data/Wu_Damage_scenarios/" 
    # "C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\"
    # "/home/hesam/Desktop/Files/Generated_Network_Dataset_v3.1"
    # 'C:/Users/ht20/Box Sync/Shelby County Database/Damage_scenarios/'
    # "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Joplin/Damage_scenarios/random_disruption/"
    # "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Damage_scenarios/random_disruption/"
    #: The address to where output are stored.
    OUTPUT_DIR = '../results/'
    # '/home/hesam/Desktop/Files/Game_Shelby_County/results_0.9_perc'
    # 'C:/Users/ht20/Documents/Files/Auction_Extended_Shelby_County_Data/results/'
    #'../results/
    # 'C:/Users/ht20/Documents/Files/Auction_synthetic_networks_v3.1/'
    # 'C:/Users/ht20/Documents/Files/Shelby_data_paper/results/'
    # FAIL_SCE_PARAM['TOPO']+'/results/'
    
    ''' Set analysis dictionaries '''
    #: Informatiom on the ype of the failure scenario (Andres or Wu)
    #: and network dataset (shelby or synthetic)
    #: Help:
    #: For Andres scenario: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    #:     magnitudes: FAIL_SCE_PARAM['MAGS']
    #: For Wu scenario: set range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    #:     sce range: FAIL_SCE_PARAM['MAGS']
    #: For Synthetic nets: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    #:     configurations: FAIL_SCE_PARAM['MAGS']
    # FAIL_SCE_PARAM = {'TYPE':"WU", 'SAMPLE_RANGE':range(14), 'MAGS':range(2),
    #                   'FILTER_SCE':FILTER_SCE, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"ANDRES", 'SAMPLE_RANGE':range(1, 1001), 'MAGS':[6, 7, 8, 9],
    #                  'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    FAIL_SCE_PARAM = {'TYPE':"random", 'SAMPLE_RANGE':range(0, 10), 'MAGS':range(0, 1),
                      'FILTER_SCE':None, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"synthetic", 'SAMPLE_RANGE':range(0, 1), 'MAGS':range(68, 69),
    #                   'FILTER_SCE':None, 'TOPO':'Grid',
    #                   'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}

    ### Dynamic parameters dict
    DYNAMIC_PARAMS = None

    # DYNAMIC_PARAMS = {'TYPE': 'shelby_adopted', 'RETURN': 'step_function',
    #                   'DIR': 'C:/Users/ht20/Documents/Files/dynamic_demand/'}
    
    # ROOT_DISLOC = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Joplin/"
    # DYNAMIC_PARAMS = {'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED':'joplin', 'OUT_DIR': BASE_DIR,
    #                   'POP_DISLOC_DATA': ROOT_DISLOC+'Joplin_testbed/pop-dislocation-results.csv',
    #                   'MAPPING': {'POWER': ROOT_DISLOC+'/Power/Joplin interdependency table - buildings,\
    #                               substations, and poles/Joplin_interdependency_table.csv'}}
    
    ROOT_DISLOC = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/"
    POP_DISLOC_DATA = ROOT_DISLOC+'Seaside_testbed/housingunit_eq_100yr_popdis_result.csv'
    DYNAMIC_PARAMS = {'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED':'seaside',
                      'OUT_DIR': BASE_DIR, 'POP_DISLOC_DATA': POP_DISLOC_DATA,
                      'MAPPING': {'POWER': ROOT_DISLOC+'Power/bldgs2elec_Seaside.csv',
                                  'WATER': ROOT_DISLOC+'Water/bldgs2wter_Seaside.csv'}}    

    ###  Multicommodity parameters dict
    # EXTRA_COMMODITY = None
    EXTRA_COMMODITY = {1:['PW'], 3:[]}
    
    ### Dict contains information about the statistical models approximating INDP
    MODEL_DIR = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc'
    STM_MODEL_DICT = None
    # {'num_pred':1, 'model_dir':MODEL_DIR+'/traces', 'param_folder':MODEL_DIR+'/parameters'}

    ### Directory with objects containing payoff values for games
    PAYOFF_DIR = '/home/hesam/Desktop/Files/Game_Shelby_County/results_NE_only_objs/'

    # Output and base dir for sythetic database
    SYNTH_DIR = None
    if FAIL_SCE_PARAM['TYPE'] == 'synthetic':
        SYNTH_DIR = BASE_DIR+FAIL_SCE_PARAM['TOPO']+'Networks/'
        OUTPUT_DIR += FAIL_SCE_PARAM['TOPO']+'/results/'

    ''' Set analysis parameters '''
    # No restriction on number of resources for each layer
    RC = [{'budget':120000, 'time':35}, {'budget':240000, 'time':35},
          {'budget':120000, 'time':70}, {'budget':120000, 'time':105},
          {'budget':240000, 'time':105}]

    # Not necessary for synthetic nets
    # Prescribed for each layer -> RC = [{'budget':{1:60000, 3:700}, 'time':{1:2, 3:10}}] 
    LAYERS = [1,3]#[1, 2, 3, 4]
    # Not necessary for synthetic nets
    JUDGE_TYPE = ["PESSIMISTIC", "OPTIMISTIC", "DET-DEMAND"]
    #["PESSIMISTIC", "OPTIMISTIC", "DEMAND", "DET-DEMAND", "RANDOM"]
    RES_ALLOC_TYPE = ["MCA", 'UNIFORM', 'OPTIMAL']
    #["MDA", "MAA", "MCA", 'UNIFORM', 'OPTIMAL']
    VAL_TYPE = ['DTC']
    #['DTC', 'DTC_uniform', 'MDDN', 'STM', 'DTC-LP']
    ''' Run different methods '''
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP', output_dir=OUTPUT_DIR,
    #             misc = {'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
    #                     'EXTRA_COMMODITY':EXTRA_COMMODITY,
    #                     'TIME_RESOURCE':True})
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='TDINDP', output_dir=OUTPUT_DIR,
    #             misc = {'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
    #                     'EXTRA_COMMODITY':EXTRA_COMMODITY})
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='JC', judgment_type=JUDGE_TYPE,
    #             res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE,
    #             output_dir=OUTPUT_DIR, dynamic_params=DYNAMIC_PARAMS,
    #             misc = {'STM_MODEL':STM_MODEL_DICT, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
    #                     'EXTRA_COMMODITY':EXTRA_COMMODITY})
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='NORMALGAME', judgment_type=JUDGE_TYPE,
    #             res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR,
    #             misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
    #                     'EXTRA_COMMODITY':EXTRA_COMMODITY})
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='BAYESGAME', judgment_type=JUDGE_TYPE,
    #             res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR,
    #             misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
    #                     'EXTRA_COMMODITY':EXTRA_COMMODITY,
    #                     "SIGNALS":{x:'C' for x in LAYERS}, "BELIEFS":{x:'U' for x in LAYERS}})

    ''' Post-processing '''
    # COST_TYPES = ['Total'] # 'Under Supply', 'Over Supply'
    # REF_METHOD = 'indp'
    # METHOD_NAMES = ['indp', 'dp_indp'] #'ng', 'jc', 'dp_indp', 'tdindp' ''bgCCCCUUUU'

    # COMBS, OPTIMAL_COMBS = dindputils.generate_combinations(FAIL_SCE_PARAM['TYPE'],
    #             FAIL_SCE_PARAM['MAGS'], FAIL_SCE_PARAM['SAMPLE_RANGE'], LAYERS,
    #             RC, METHOD_NAMES, JUDGE_TYPE, RES_ALLOC_TYPE, VAL_TYPE,
    #             list_high_dam_add=FAIL_SCE_PARAM['FILTER_SCE'],
    #             synthetic_dir=SYNTH_DIR)

    # BASE_DF, objs = dindputils.read_results(COMBS, OPTIMAL_COMBS, COST_TYPES,
    #                                     root_result_dir=OUTPUT_DIR, deaggregate=True)

    # # LAMBDA_DF = dindputils.relative_performance(BASE_DF, COMBS, OPTIMAL_COMBS,
    # #                                         ref_method=REF_METHOD, cost_type=COST_TYPES[0])
    # # # RES_ALLOC_DF, ALLOC_GAP_DF = dindputils.read_resourcec_allocation(BASE_DF, COMBS, OPTIMAL_COMBS,
    # # #                                                               objs, root_result_dir=OUTPUT_DIR,
    # # #                                                               ref_method=REF_METHOD)
    # # RUN_TIME_DF = dindputils.read_run_time(COMBS, OPTIMAL_COMBS, objs, root_result_dir=OUTPUT_DIR)
    # # # ANALYZE_NE_DF = gameutils.analyze_NE(objs, COMBS, OPTIMAL_COMBS)

    # ''' Save Variables to file '''
    # # OBJ_LIST = [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF,
    # #             RES_ALLOC_DF, ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPES, ANALYZE_NE_DF]
    # OBJ_LIST = [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, COST_TYPES]

    # ### Saving the objects ###
    # with open(OUTPUT_DIR+'postprocess_dicts.pkl', 'wb') as f:
    #     pickle.dump(OBJ_LIST, f)

    # ''' Plot results '''
    plt.close('all')
        
    ### Getting back the objects ###
    # with open(OUTPUT_DIR+'postprocess_dicts.pkl', 'rb') as f:
    #     # [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
    #     #   ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPE, ANALYZE_NE_DF] = pickle.load(f)
    #     [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, COST_TYPES] = pickle.load(f)

    plots.plot_performance_curves(BASE_DF[(BASE_DF['decision_type']=='dp_indp')],
                                  cost_type='Total', ci=None,
                                  deaggregate=False, plot_resilience=True)

    # plots.plot_seperated_perform_curves(BASE_DF, x='t', y='cost', cost_type='Total',
    #                                     ci=95, normalize=False)

    # plots.plot_relative_performance(LAMBDA_DF, lambda_type='U')
    # plots.plot_auction_allocation(RES_ALLOC_DF, ci=95)
    # plots.plot_relative_allocation(ALLOC_GAP_DF, distance_type='gap')
    # plots.plot_run_time(RUN_TIME_DF, ci=95)
    # plots.plot_ne_analysis(ANALYZE_NE_DF, ci=None)
    # plots.plot_ne_cooperation(ANALYZE_NE_DF, ci=None)
   
    
    # [(RUN_TIME_DF['auction_type']!='MDA')&(RUN_TIME_DF['auction_type']!='MAA')]
    # [(BASE_DF['judgment_type']!='PESSIMISTIC')&\
    #  (BASE_DF['judgment_type']!='DET-DEMAND')&\
    #      (BASE_DF['decision_type']!='indp')&\
    #          (BASE_DF['auction_type']!='OPTIMAL')]