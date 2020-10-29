""" Runs INDP, td-INDP, Judgment Call, and infrastructure games"""
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
    shelby_data = None
    ext_interdependency = None
    if fail_sce_param['TYPE'] == 'Andres':
        shelby_data = 'shelby_old'
        ext_interdependency = "../data/INDP_4-12-2016"
    elif fail_sce_param['TYPE'] == 'WU':
        shelby_data = 'shelby_extended'
        if fail_sce_param['FILTER_SCE'] is not None:
            list_high_dam = pd.read_csv(fail_sce_param['FILTER_SCE'])
    elif fail_sce_param['TYPE'] == 'random':
        shelby_data = 'shelby_extended'
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
            if shelby_data:
                params["N"], _, _ = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            sim_number=0, magnitude=6, sample=0, v=params["V"],
                            shelby_data=shelby_data)
            else:
                params["N"], params["V"], params['L'] = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            magnitude=m, sample=i, shelby_data=shelby_data,
                            topology=topology)
            params["SIM_NUMBER"] = i
            params["MAGNITUDE"] = m
            # Check if the results exist
            output_dir_full = ''
            if params["ALGORITHM"] != "JC":
                output_dir_full = params["OUTPUT_DIR"]+'_L'+str(len(params["L"]))+'_m'+\
                    str(params["MAGNITUDE"])+"_v"+str(params["V"])+'/agents/actions_'+str(i)+'_L1_.csv'
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
                              controlled_layers=params['L'], saveModel=True, print_cmd_line=False)
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
            elif params["ALGORITHM"] == "NORMALGAME":
                gameutils.run_game(params, save_results=True, print_cmd=True,
                                   save_model=True, plot2D=True)

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

def run_game_sample(layers, judge_types, auction_type, valuation_type):
    interdep_net= indp.initialize_sample_network(layers=layers)
    params={"NUM_ITERATIONS":7, "OUTPUT_DIR":'../results/ng_sample_12Node_results',
            "V":len(layers), "T":1, "L":layers, "WINDOW_LENGTH":1, "ALGORITHM":"NORMALGAME",
            'EQUIBALG':'gnm_solve', "N":interdep_net, "MAGNITUDE":0, "SIM_NUMBER":0,
            "JUDGMENT_TYPE":judge_types, "RES_ALLOC_TYPE":auction_type,
            "VALUATION_TYPE":valuation_type}   
    gameutils.run_game(params, save_results=True, print_cmd=False, save_model=True, plot2D=True)
    for jt, rst, vt in itertools.product(judge_types, auction_type, valuation_type):
        print('\n\nPlot restoration plan by Game',jt,rst,vt)
        if rst == 'UNIFORM':
            indp.plot_indp_sample(params, folderSuffix='_'+jt+'_'+rst, suffix="")
        else:
            indp.plot_indp_sample(params, folderSuffix='_'+jt+'_AUCTION_'+rst+'_'+vt, suffix="")
        plt.show()

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
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":output_dir+'/indp_results',
                      "V":v, "T":1, 'L':layers, "ALGORITHM":"INDP"}
        elif method == 'JC':
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":output_dir+'/jc_results',
                      "V":v, "T":1, 'L':layers, "ALGORITHM":"JC",
                      "JUDGMENT_TYPE":judgment_type, "RES_ALLOC_TYPE":res_alloc_type,
                      "VALUATION_TYPE":valuation_type}
            if 'STM' in valuation_type:
                params['STM_MODEL_DICT'] = misc['STM_MODEL_DICT']
        elif method == 'NORMALGAME':
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":output_dir+'/ng_results',
                      "V":v, "T":1, "L":layers, "ALGORITHM":"NORMALGAME",
                      'EQUIBALG':'enumpure_solve', "JUDGMENT_TYPE":judgment_type,
                      "RES_ALLOC_TYPE":res_alloc_type, "VALUATION_TYPE":valuation_type}
        elif method == 'TD_INDP':
            params = {"NUM_ITERATIONS":1, "OUTPUT_DIR":output_dir+'/tdindp_results',
                      "V":v, "T":10, "WINDOW_LENGTH":3, 'L':layers, "ALGORITHM":"INDP"}
        else:
            sys.exit('Wrong method name: '+method)
        batch_run(params, fail_sce_param)
 
if __name__ == "__main__":
    ''' Run a toy example for different methods '''
    plt.close('all')
    # layers=[1,2,3]
    # auction_type = ["MCA", "UNIFORM"]#, "MAA", "MDA"
    # valuation_type = ["DTC"]
    # judge_types = ["OPTIMISTIC"]#"PESSIMISTIC",
    # run_indp_sample(layers)
    # run_jc_sample(layers, judge_types, auction_type, valuation_type)
    # run_game_sample(layers, judge_types, auction_type, valuation_type)
    
    # COMBS = []
    # OPTIMAL_COMBS = [[0, 0, len(layers), len(layers), 'indp_sample_12Node', 'nan',
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

    # obj_dir = "../results/ng_sample_12Node_results_L2_m0_v2_OPTIMISTIC_AUCTION_MCA_DTC/"
    # with open(obj_dir+'objs_0.pkl', 'rb') as f:
    #     obj = pickle.load(f)
    ''' ^^^ '''
    
    #: The address to the list of scenarios that should be included in the analyses.
    FILTER_SCE = '../data/damagedElements_sliceQuantile_0.90.csv'

    #: The address to the basic (topology, parameters, etc.) information of the network.
    BASE_DIR = 'C:/Users/ht20/Box Sync/Shelby County Database/Node_arc_info/'
    # "../data/Extended_Shelby_County/"
    # 'C:/Users/ht20/Box Sync/Shelby County Database/Node_arc_info'
    # "C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\"
    # "/home/hesam/Desktop/Files/Generated_Network_Dataset_v3.1"

    #: The address to damge scenario data.
    DAMAGE_DIR = 'C:/Users/ht20/Box Sync/Shelby County Database/Damage_scenarios/'
    # ../data/random_disruption_shelby/"
    #"../data/Wu_Damage_scenarios/" 
    # "C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\"
    # "/home/hesam/Desktop/Files/Generated_Network_Dataset_v3.1"
    # 'C:/Users/ht20/Box Sync/Shelby County Database/Damage_scenarios'

    #: The address to where output are stored.
    OUTPUT_DIR = 'C:/Users/ht20/Documents/Files/Shelby_data_paper/Restoration_results/'
    #'C:/Users/ht20/Documents/Files/Auction_Extended_Shelby_County_Data/results/'
    #'../results/
    # 'C:/Users/ht20/Documents/Files/Auction_synthetic_networks_v3.1/'
    # 'C:/Users/ht20/Documents/Files/Shelby_data_paper/Restoration_results/'
    # FAIL_SCE_PARAM['TOPO']+'/results/'

    #: Informatiom on the ype of the failure scenario (Andres or Wu)
    #: and network dataset (shelby or synthetic)
    #: Help:
    #: For Andres scenario: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    #:     magnitudes: FAIL_SCE_PARAM['MAGS']
    #: For Wu scenario: set range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    #:     sce range: FAIL_SCE_PARAM['MAGS']
    #: For Synthetic nets: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    #:     configurations: FAIL_SCE_PARAM['MAGS']
    FAIL_SCE_PARAM = {'TYPE':"WU", 'SAMPLE_RANGE':range(0, 50), 'MAGS':range(0, 96),
                      'FILTER_SCE':FILTER_SCE, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"ANDRES", 'SAMPLE_RANGE':range(1, 1001), 'MAGS':[6, 7, 8, 9],
    #                  'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"random", 'SAMPLE_RANGE':range(10, 11), 'MAGS':range(0, 1),
    #                   'FILTER_SCE':None, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"synthetic", 'SAMPLE_RANGE':range(0, 1), 'MAGS':range(68, 69),
    #                   'FILTER_SCE':None, 'TOPO':'Grid',
    #                   'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # Oytput and base dir for sythetic database
    SYNTH_DIR = None
    if FAIL_SCE_PARAM['TYPE'] == 'synthetic':
        SYNTH_DIR = BASE_DIR+FAIL_SCE_PARAM['TOPO']+'Networks/'
        OUTPUT_DIR += FAIL_SCE_PARAM['TOPO']+'/results/'
    # No restriction on number of resources for each layer
    RC = [3, 6, 8, 12]
    # Not necessary for synthetic nets
    # [3, 6, 8, 12]
    # [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]# Prescribed for each layer
    LAYERS = [1, 2, 3, 4]
    # Not necessary for synthetic nets
    JUDGE_TYPE = ["OPTIMISTIC"]
    #["PESSIMISTIC", "OPTIMISTIC", "DEMAND", "DET-DEMAND", "RANDOM"]
    RES_ALLOC_TYPE = ["MCA", 'UNIFORM']
    #["MDA", "MAA", "MCA", 'UNIFORM']
    VAL_TYPE = ['DTC']
    #['DTC', 'DTC_uniform', 'MDDN', 'STM', 'DTC-LP']
    # MODEL_DIR = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc'
    # STM_MODEL_DICT = {'num_pred':1, 'model_dir':MODEL_DIR+'/traces',
    #                   'param_folder':MODEL_DIR+'/parameters'}

    ''' Run different methods '''
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP', output_dir=OUTPUT_DIR)
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='JC', judgment_type=JUDGE_TYPE,
    #             res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE,
    #             output_dir=OUTPUT_DIR)#, misc = {'STM_MODEL_DICT':STM_MODEL_DICT})
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='NORMALGAME', judgment_type=JUDGE_TYPE,
    #            res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR)

    ''' Post-processing '''
    COST_TYPES = ['Total']
    REF_METHOD = 'indp'
    METHOD_NAMES = ['indp', 'jc'] #'ng'

    # COMBS, OPTIMAL_COMBS = dindputils.generate_combinations(FAIL_SCE_PARAM['TYPE'],
    #             FAIL_SCE_PARAM['MAGS'], FAIL_SCE_PARAM['SAMPLE_RANGE'], LAYERS,
    #             RC, METHOD_NAMES, JUDGE_TYPE, RES_ALLOC_TYPE, VAL_TYPE,
    #             list_high_dam_add=FAIL_SCE_PARAM['FILTER_SCE'],
    #             synthetic_dir=SYNTH_DIR)

    # BASE_DF, objs = dindputils.read_results(COMBS, OPTIMAL_COMBS, COST_TYPES,
    #                                     root_result_dir=OUTPUT_DIR, deaggregate=True)

    # LAMBDA_DF = dindputils.relative_performance(BASE_DF, COMBS, OPTIMAL_COMBS,
    #                                         ref_method=REF_METHOD, cost_type=COST_TYPES[0])
    # RES_ALLOC_DF, ALLOC_GAP_DF = dindputils.read_resourcec_allocation(BASE_DF, COMBS, OPTIMAL_COMBS,
    #                                                               objs, root_result_dir=OUTPUT_DIR,
    #                                                               ref_method=REF_METHOD)
    # RUN_TIME_DF = dindputils.read_run_time(COMBS, OPTIMAL_COMBS, objs, root_result_dir=OUTPUT_DIR)

    # ''' Save Variables to file '''
    # OBJ_LIST = [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF,
    #             RES_ALLOC_DF, ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPES]

    ### Saving the objects ###
    with open(OUTPUT_DIR+'postprocess_dicts.pkl', 'wb') as f:
        pickle.dump(OBJ_LIST, f)

    # # Getting back the objects ###
    # with open(OUTPUT_DIR+'postprocess_dicts.pkl', 'rb') as f:
    #     [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
    #       ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPE] = pickle.load(f)

    ### Plot results ###
    plt.close('all')

    plots.plot_performance_curves(BASE_DF[(BASE_DF['no_resources']==3)|(BASE_DF['no_resources']==8)],
                                  cost_type='Total', ci=None,
                                  deaggregate=False, plot_resilience=True)

    # plots.plot_seperated_perform_curves(BASE_DF, x='t', y='cost', cost_type='Total',
    #                                     ci=None, normalize=False)

    # plots.plot_relative_performance(LAMBDA_DF, lambda_type='U')
    # plots.plot_auction_allocation(RES_ALLOC_DF, ci=None)
    # plots.plot_relative_allocation(ALLOC_GAP_DF, distance_type='gap')
    # plots.plot_run_time(RUN_TIME_DF[(RUN_TIME_DF['auction_type']!='MDA')&(RUN_TIME_DF['auction_type']!='MAA')], ci=95)
    ## [(RUN_TIME_DF['auction_type']!='MDA')&(RUN_TIME_DF['auction_type']!='MAA')]