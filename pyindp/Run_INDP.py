""" Runs INDP, td-INDP, and Jidgment Call """
import sys
import _pickle as pickle
import pandas as pd
import matplotlib.pyplot as plt
import indp
import Dindp
import plots
import gametree

def batch_run(params, fail_sce_param, layers, player_ordering=[3, 1]):
    """ Batch run INDP optimization problem for all samples (currently 1-1000),
        given global parameters.
    Format for params:
    "NUM_ITERATIONS": For single timestep INDP, specifies how many iterations to run.
                      For InfoShare, specifies how many rounds of information sharing to perform.
    "OUTPUT_DIR"    : Directory to output results.
    "MAGNITUDE"     : Magnitude of earthquake. Used to look up failure scenarios.
    "V"             : How many resources used in a scenario. Used to look up failure scenarios,
                    and specify v_r for indp.
    "T"             : Number of timesteps to optimize over. Used for td-INDP and InfoShare.
    "WINDOW_LENGTH" : Slding time window length for td-INDP (for efficiency).
    "N"             : InfrastructureNetwork to use in indp.
    "SIM_NUMBER"    : What failure scenario simulation to use. Used to look up failure scenarios."""

    # Set root directories
    base_dir = fail_sce_param['BASE_DIR']
    damage_dir = fail_sce_param['DAMAGE_DIR']
    topology = None
    shelby_data = True
    ext_interdependency = None
    if fail_sce_param['TYPE'] == 'Andres':
        ext_interdependency = "../data/INDP_4-12-2016"
    elif fail_sce_param['TYPE'] == 'WU':
        if fail_sce_param['FILTER_SCE'] is not None:
            list_high_dam = pd.read_csv(fail_sce_param['FILTER_SCE'])
    elif fail_sce_param['TYPE'] == 'random':
        pass
    elif fail_sce_param['TYPE'] == 'synthetic':
        shelby_data = False
        topology = fail_sce_param['TOPO']

    for m in fail_sce_param['MAGS']:
        for i in fail_sce_param['SAMPLE_RANGE']:
            try:
                list_high_dam
                if len(list_high_dam.loc[(list_high_dam.set == i) & (list_high_dam.sce == m)].index) == 0:
                    continue
            except NameError:
                pass

            print('\n---Running Magnitude '+str(m)+' sample '+str(i)+'...')
            print("Initializing network...")
            if shelby_data:
                interdep_net, _, _ = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            sim_number=0, magnitude=6, sample=0, v=params["V"],
                            shelby_data=shelby_data)
            else:
                interdep_net, no_resource, layers = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            magnitude=m, sample=i, shelby_data=shelby_data,
                            topology=topology)
                params["V"] = no_resource

            params["N"] = interdep_net
            params["SIM_NUMBER"] = i
            params["MAGNITUDE"] = m

            if fail_sce_param['TYPE'] == 'WU':
                indp.add_Wu_failure_scenario(interdep_net, DAM_DIR=damage_dir,
                                              noSet=i, noSce=m)
            elif fail_sce_param['TYPE'] == 'ANDRES':
                indp.add_failure_scenario(interdep_net, DAM_DIR=damage_dir,
                                           magnitude=m, v=params["V"], sim_number=i)
            elif fail_sce_param['TYPE'] == 'random':
                indp.add_random_failure_scenario(interdep_net, DAM_DIR=damage_dir,
                                                  sample=i)
            elif fail_sce_param['TYPE'] == 'synthetic':
                indp.add_synthetic_failure_scenario(interdep_net, DAM_DIR=base_dir,
                                                     topology=topology, config=m, sample=i)

            if params["ALGORITHM"] == "INDP":
                indp.run_indp(params, validate=False, T=params["T"], layers=layers,
                               controlled_layers=layers, saveModel=True, print_cmd_line=True)
            elif params["ALGORITHM"] == "INFO_SHARE":
                indp.run_info_share(params, layers=layers, T=params["T"])
            elif params["ALGORITHM"] == "INRG":
                indp.run_inrg(params, layers=layers, player_ordering=player_ordering)
            elif params["ALGORITHM"] == "BACKWARDS_INDUCTION":
                gametree.run_backwards_induction(interdep_net, i, players=layers,
                                                 player_ordering=player_ordering,
                                                 T=params["T"], outdir=params["OUTPUT_DIR"])
            elif params["ALGORITHM"] == "JC":
                Dindp.run_judgment_call(params, layers=layers, T=params["T"],
                                        saveJCModel=True, print_cmd=True)

# def run_indp_sample():
#     import warnings
#     warnings.filterwarnings("ignore")
#     interdep_net=initialize_sample_network()
#     params={"NUM_ITERATIONS":7, "OUTPUT_DIR":'../results/sample_indp_12Node_results_L2',
#             "V":2, "T":1, "WINDOW_LENGTH":1, "ALGORITHM":"INDP"}
#     params["N"]=interdep_net
#     params["MAGNITUDE"]=0
#     params["SIM_NUMBER"]=0
#     run_indp(params, layers=[1, 2], T=params["T"], suffix="", saveModel=True,
#               print_cmd_line=True)
#     plot_indp_sample(params)
#     auction_type = ["MCA", "MAA", "MDA"]
#     valuation_type = ["DTC", "MDDN"]
#     for at, vt in itertools.product(auction_type, valuation_type):
#         for jc in ["PESSIMISTIC", "OPTIMISTIC"]:
#             interdep_net=initialize_sample_network()
#             params["N"]=interdep_net
#             params["NUM_ITERATIONS"]=7
#             params["ALGORITHM"]="JC"
#             params["JUDGMENT_TYPE"]=jc
#             params["OUTPUT_DIR"]='../results/sample_judgeCall_12Node_'+jc+'_results_L2'
#             params["V"]=[1, 1]
#             params["T"]=1
#             params["AUCTION_TYPE"]= at
#             params["VALUATION_TYPE"]= vt
            # run_judgment_call(params, layers=[1, 2], T=params["T"], saveJCModel=True,
            #                   print_cmd=True)
            # plot_indp_sample(params,
            # folderSuffix='_auction_'+params["AUCTION_TYPE"]+'_'+params["VALUATION_TYPE"],
            # suffix="sum")

#     METHOD_NAMES=['sample_indp_12Node', 'sample_judgeCall_12Node_PESSIMISTIC']
#     SUFFIXES=['', 'Real_sum']

#     df = read_results(mags=[0], METHOD_NAMES=METHOD_NAMES,
#                     auction_type=auction_type, valuation_type=valuation_type,
#                     SUFFIXES=SUFFIXES, L=2, sample_range=[0], no_resources=[2])
#     LAMBDA_DF = relative_performance(df, sample_range=[0], REF_METHOD='sample_indp_12Node')
#     resource_allocation=read_resourcec_allocation(df, sample_range=[0], L=2, T=5,
#                                     layers=[1, 2], REF_METHOD='sample_indp_12Node')

#     plot_performance_curves(df, COST_TYPE='Total', decision_names=METHOD_NAMES,
#                             auction_type=auction_type, valuation_type=valuation_type, ci=None)
#     plot_relative_performance(LAMBDA_DF)
#     plot_auction_allocation(resource_allocation, ci=None)
#     plot_relative_allocation(resource_allocation)

def run_method(fail_sce_param, v_r, layers, method, judgment_type=None,
               auction_type=None, valuation_type=None):
    """
    This function runs a given method for different numbers of resources,
    and a given judge, auction, and valuation type
    Args:
        fail_sce_param (dict): informaton of damage scenrios
        v_r (list, float or list of float): number of resources,
        if this is a list of floats, each float is interpreted as a different total
        number of resources, and indp is run given the total number of resources.
        It only works when auction_type != None.
        If this is a list of lists of floats, each list is interpreted as fixed upper
        bounds on the number of resources each layer can use (same for all time step).
        judgment_type (str): Type of Judgments in Judfment Call Method.
        auction_type (str): Type of auction for resource allocation. If None,
        fixed number of resources is allocated based on v_r, which MUST be a list
        of float when auction_type == None.
        valuation_type (str): Type of valuation in auction.
    Returns:
        None - writes to file
    """
    for v in v_r:
        if method == 'INDP':
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":'../results/indp_results',
                      "V":v, "T":1, "ALGORITHM":"INDP"}
        elif method == 'JC':
            params = {"NUM_ITERATIONS":10,
                      "OUTPUT_DIR":'../results/judgeCall_'+judgment_type+'_results',
                      "V":v, "T":1, "ALGORITHM":"JC",
                      "JUDGMENT_TYPE":judgment_type, "AUCTION_TYPE":auction_type,
                      "VALUATION_TYPE":valuation_type}
        elif method == 'TD_INDP':
            params = {"NUM_ITERATIONS":1, "OUTPUT_DIR":'../results/tdindp_results',
                      "V":v, "T":10, "WINDOW_LENGTH":3, "ALGORITHM":"INDP"}
        else:
            sys.exit('Wrong method name: '+method)
        batch_run(params, fail_sce_param, layers=layers)

if __name__ == "__main__":
    ### Run a toy example for different methods ###
#    run_indp_sample()

    ### Decide the failure scenario (Andres or Wu) and network dataset (shelby or synthetic)
    ### Help:
    ### For Andres scenario: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    ###     magnitudes: FAIL_SCE_PARAM['MAGS']
    ### For Wu scenario: set range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    ###     sce range: FAIL_SCE_PARAM['MAGS']
    ### For Synthetic nets: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    ###     configurations: FAIL_SCE_PARAM['MAGS']
    FILTER_SCE = '../data/damagedElements_sliceQuantile_0.95.csv'
    BASE_DIR = "../data/Extended_Shelby_County/"
    DAMAGE_DIR = "../data/Wu_Damage_scenarios/" #random_disruption_shelby/"
    OUTPUT_DIR = 'C:/Users/ht20/Documents/Files/Auction_Extended_Shelby_County_Data/results/'#'../results/'

    # failSce = read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/", magnitude=8)
    # FAIL_SCE_PARAM = {'TYPE':"ANDRES", 'SAMPLE_RANGE':range(1, 1001), 'MAGS':[6, 7, 8, 9],
    #                  'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    FAIL_SCE_PARAM = {'TYPE':"WU", 'SAMPLE_RANGE':range(0, 50), 'MAGS':range(0, 95),
                    'FILTER_SCE':FILTER_SCE, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"random", 'SAMPLE_RANGE':range(10, 11), 'MAGS':range(0, 1),
    #                   'FILTER_SCE':None, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"synthetic", 'SAMPLE_RANGE':range(0, 5), 'MAGS':range(0, 100),
    #                   'FILTER_SCE':None, 'TOPO':'Grid'
    #                   'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}

    # No restriction on number of resources for each layer
    RC = [3,6,8,12]
    # Not necessary for synthetic nets
    # [3, 6, 8, 12]
    # [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]# Prescribed for each layer
    LAYERS = [1, 2, 3, 4]
    # Not necessary for synthetic nets
    JUDGE_TYPE = ['OPTIMISTIC']
    #["PESSIMISTIC", "OPTIMISTIC", "DEMAND", "DET-DEMAND", "RANDOM"]
    AUC_TYPE = ["MDA", "MAA", "MCA"]
    #["MDA", "MAA", "MCA"]
    VAL_TYPE = ['DTC']
    #['DTC', 'DTC_uniform', 'MDDN']

    # ### Run different methods###
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP')
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='TD_INDP')
    # for jc in JUDGE_TYPE:
    #     run_method(FAIL_SCE_PARAM, RC, LAYERS, method='JC', judgment_type=jc,
    #                 auction_type=None, valuation_type=None)
    #     for at in AUC_TYPE:
    #         for vt in VAL_TYPE:
    #             run_method(FAIL_SCE_PARAM, RC, LAYERS, method='JC',
    #                         judgment_type=jc, auction_type=at, valuation_type=vt)

    ### Compute metrics ###
    COST_TYPE = 'Total'
    REF_METHOD = 'indp'
    METHOD_NAMES = ['indp']
    for jc in JUDGE_TYPE:
        METHOD_NAMES.append('judgeCall_'+jc)
    AUC_TYPE.append('Uniform')
    SUFFIXES = ['Real_sum', '']

    SYNTH_DIR = None #BASE_DIR+FAIL_SCE_PARAM['TOPO']+'Networks/'
    COMBS, OPTIMAL_COMBS = Dindp.generate_combinations('shelby',
                FAIL_SCE_PARAM['MAGS'], FAIL_SCE_PARAM['SAMPLE_RANGE'], LAYERS,
                RC, METHOD_NAMES, AUC_TYPE, VAL_TYPE, 
                list_high_dam_add=FAIL_SCE_PARAM['FILTER_SCE'],
                synthetic_dir=SYNTH_DIR)

    ROOT = OUTPUT_DIR #+FAIL_SCE_PARAM['TOPO']+'/results/'
    BASE_DF = Dindp.read_results(COMBS, OPTIMAL_COMBS, SUFFIXES, root_result_dir=ROOT,
                                  deaggregate=True)
##    BASE_DF = correct_tdindp_results(BASE_DF, OPTIMAL_COMBS)

    LAMBDA_DF = Dindp.relative_performance(BASE_DF, COMBS, OPTIMAL_COMBS,
                                            ref_method=REF_METHOD, cost_type=COST_TYPE)
    RES_ALLOC_DF, RES_ALLOC_REL_DF = Dindp.read_resourcec_allocation(BASE_DF, COMBS,
                OPTIMAL_COMBS, root_result_dir=ROOT, ref_method=REF_METHOD)
    RUN_TIME_DF = Dindp.read_run_time(COMBS, OPTIMAL_COMBS, SUFFIXES,
                                      root_result_dir=ROOT)

    ## Save Variables to file ###
    object_list = [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF,
                    RES_ALLOC_DF, RES_ALLOC_REL_DF, COST_TYPE, RUN_TIME_DF]
    # Saving the objects:
    with open(OUTPUT_DIR+'objs.pkl', 'wb') as f:
        pickle.dump(object_list, f)

    # # Getting back the objects:
    # with open('./NOTS/objs.pkl') as f:  # Python 3: open(..., 'rb')
    #     [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
    #       RES_ALLOC_REL_DF, COST_TYPE, RUN_TIME_DF] = pickle.load(f)

    ### Plot results ###
    plt.close('all')

    plots.plot_performance_curves_shelby(BASE_DF, cost_type='Total',
                                          decision_names=METHOD_NAMES,
                                          ci=None, normalize=True, deaggregate=True)
    plots.plot_relative_performance_shelby(LAMBDA_DF, lambda_type='U')
    plots.plot_auction_allocation_shelby(RES_ALLOC_DF, ci=None)
    plots.plot_relative_allocation_shelby(RES_ALLOC_REL_DF)
    plots.plot_run_time_synthetic(RUN_TIME_DF, ci=None)

    # plots.plot_performance_curves_synthetic(BASE_DF, ci=None, x='t', y='cost',
    #                                         cost_type=COST_TYPE)
    # plots.plot_performance_curves_synthetic(BASE_DF, ci=None, x='t', y='cost',
    #                                         cost_type='Under Supply Perc')
    # plots.plot_relative_performance_synthetic(LAMBDA_DF, cost_type=COST_TYPE,
    #                                           lambda_type='U')
    # plots.plot_auction_allocation_synthetic(RES_ALLOC_DF, ci=None,
    #                                         resource_type='normalized_resource')
    # plots.plot_relative_allocation_synthetic(RES_ALLOC_REL_DF)
    # plots.plot_run_time_synthetic(RUN_TIME_DF, ci=None)
