""" Runs INDP, td-INDP, and Jidgment Call """
import sys
import Dindp
import plots
import gametree
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def batch_run(params, failSce_param, layers, player_ordering=[3, 1]):
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
    base_dir = failSce_param['BASE_DIR']
    damage_dir = failSce_param['DAMAGE_DIR']
    topology = None
    shelby_data = True
    ext_interdependency = None
    if failSce_param['type'] == 'Andres':
        ext_interdependency = "../data/INDP_4-12-2016"
    elif failSce_param['type'] == 'WU':
        if failSce_param['FILTER_SCE'] != None:
            listHD = pd.read_csv(failSce_param['FILTER_SCE'])
    elif failSce_param['type'] == 'random':
        pass
    elif failSce_param['type'] == 'synthetic':
        shelby_data = False
        topology = failSce_param['TOPO']

    for m in failSce_param['mags']:
        for i in failSce_param['sample_range']:
            if failSce_param['FILTER_SCE'] == None or len(listHD.loc[(listHD.set == i) & (listHD.sce == m)].index):
                print('\n---Running Magnitude '+str(m)+' sample '+str(i)+'...')
                print("Initializing network...")
                if shelby_data:
                    InterdepNet, _, _ = Dindp.initialize_network(BASE_DIR=base_dir,
                                external_interdependency_dir=ext_interdependency,
                                sim_number=0, magnitude=6, sample=0, v=params["V"],
                                shelby_data=shelby_data)
                else:
                    InterdepNet, noResource, layers = Dindp.initialize_network(BASE_DIR=base_dir,
                                external_interdependency_dir=ext_interdependency,
                                magnitude=m, sample=i, shelby_data=shelby_data,
                                topology=topology)
                    params["V"] = noResource

                params["N"] = InterdepNet
                params["SIM_NUMBER"] = i
                params["MAGNITUDE"] = m

                if failSce_param['type'] == 'WU':
                    Dindp.add_Wu_failure_scenario(InterdepNet, DAM_DIR=damage_dir,
                                                  noSet=i, noSce=m)
                elif failSce_param['type'] == 'ANDRES':
                    Dindp.add_failure_scenario(InterdepNet, DAM_DIR=damage_dir,
                                               magnitude=m, v=params["V"], sim_number=i)
                elif failSce_param['type'] == 'random':
                    Dindp.add_random_failure_scenario(InterdepNet, DAM_DIR=damage_dir,
                                                      sample=i)
                elif failSce_param['type'] == 'synthetic':
                    Dindp.add_synthetic_failure_scenario(InterdepNet, DAM_DIR=base_dir,
                                                         topology=topology, config=m, sample=i)

                if params["ALGORITHM"] == "INDP":
                    Dindp.run_indp(params, validate=False, T=params["T"], layers=layers,
                                   controlled_layers=layers, saveModel=True, print_cmd_line=True)
                elif params["ALGORITHM"] == "INFO_SHARE":
                    Dindp.run_info_share(params, layers=layers, T=params["T"])
                elif params["ALGORITHM"] == "INRG":
                    Dindp.run_inrg(params, layers=layers, player_ordering=player_ordering)
                elif params["ALGORITHM"] == "BACKWARDS_INDUCTION":
                    gametree.run_backwards_induction(InterdepNet, i, players=layers,
                                                     player_ordering=player_ordering,
                                                     T=params["T"], outdir=params["OUTPUT_DIR"])
                elif params["ALGORITHM"] == "JC":
                    Dindp.run_judgment_call(params, layers=layers, T=params["T"],
                                            saveJCModel=True, print_cmd=True)

# def run_indp_sample():
#     import warnings
#     warnings.filterwarnings("ignore")
#     InterdepNet=initialize_sample_network()
#     params={"NUM_ITERATIONS":7, "OUTPUT_DIR":'../results/sample_indp_12Node_results_L2',
#             "V":2, "T":1, "WINDOW_LENGTH":1, "ALGORITHM":"INDP"}
#     params["N"]=InterdepNet
#     params["MAGNITUDE"]=0
#     params["SIM_NUMBER"]=0
#     run_indp(params, layers=[1, 2], T=params["T"], suffix="", saveModel=True,
#               print_cmd_line=True)
#     plot_indp_sample(params)
#     auction_type = ["MCA", "MAA", "MDA"]
#     valuation_type = ["DTC", "MDDN"]
#     for at, vt in itertools.product(auction_type, valuation_type):
#         for jc in ["PESSIMISTIC", "OPTIMISTIC"]:
#             InterdepNet=initialize_sample_network()
#             params["N"]=InterdepNet
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

#     method_name=['sample_indp_12Node', 'sample_judgeCall_12Node_PESSIMISTIC']
#     suffixes=['', 'Real_sum']

#     df = read_and_aggregate_results(mags=[0], method_name=method_name,
#                     auction_type=auction_type, valuation_type=valuation_type,
#                     suffixes=suffixes, L=2, sample_range=[0], no_resources=[2])
#     lambda_df = relative_performance(df, sample_range=[0], ref_method='sample_indp_12Node')
#     resource_allocation=read_resourcec_allocation(df, sample_range=[0], L=2, T=5,
#                                     layers=[1, 2], ref_method='sample_indp_12Node')

#     plot_performance_curves(df, cost_type='Total', decision_names=method_name,
#                             auction_type=auction_type, valuation_type=valuation_type, ci=None)
#     plot_relative_performance(lambda_df)
#     plot_auction_allocation(resource_allocation, ci=None)
#     plot_relative_allocation(resource_allocation)

def run_method(failSce_param, v_r, layers, method, judgment_type=None,
               auction_type=None, valuation_type=None):
    """
    This function runs a given method for different numbers of resources,
    and a given judge, auction, and valuation type
    Args:
        failSce_param (dict): informaton of damage scenrios
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
        batch_run(params, failSce_param, layers=layers)

if __name__ == "__main__":
    ### Run a toy example for different methods ###
#    run_indp_sample()

    ### Decide the failure scenario (Andres or Wu) and network dataset (shelby or synthetic)
    ### Help:
    ### For Andres scenario: sample range: Fail_SCE_PARAM["sample_range"],
    ###     magnitudes: Fail_SCE_PARAM['mags']
    ### For Wu scenario: set range: Fail_SCE_PARAM["sample_range"],
    ###     sce range: Fail_SCE_PARAM['mags']
    ### For Synthetic nets: sample range: Fail_SCE_PARAM["sample_range"],
    ###     configurations: Fail_SCE_PARAM['mags']
    FILTER_SCE = '../data/damagedElements_sliceQuantile_0.95.csv'
    BASE_DIR = "../data/Extended_Shelby_County/"
    DAMAGE_DIR = "../data/random_disruption_shelby/"
    OUTPUT_DIR = '../results/'

    # failSce = read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/", magnitude=8)
    # Fail_SCE_PARAM = {"type":"ANDRES", "sample_range":range(1, 1001), "mags":[6, 7, 8, 9],
    #                  'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # Fail_SCE_PARAM = {"type":"WU", "sample_range":range(23, 24), "mags":range(5, 6),
    #                 'FILTER_SCE':FILTER_SCE, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    Fail_SCE_PARAM = {"type":"random", "sample_range":range(10, 11), "mags":range(0, 1),
                      'FILTER_SCE':None, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # Fail_SCE_PARAM = {"type":"synthetic", "sample_range":range(0, 5), "mags":range(0, 100),
    #                   'FILTER_SCE':None, 'TOPO':'Grid'
    #                   'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}

    # No restriction on number of resources for each layer
    RC = [8]
    # Not necessary for synthetic nets
    # [3, 6, 8, 12]
    # [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]# Prescribed for each layer
    LAYERS = [1, 2, 3, 4]
    # Not necessary for synthetic nets
    judge_types = ['OPTIMISTIC']
    #["PESSIMISTIC", "OPTIMISTIC", "DEMAND", "DET-DEMAND", "RANDOM"]
    auction_types = ['MCA']
    #["MDA", "MAA", "MCA"]
    valuation_types = ['MDDN']
    #['DTC', 'DTC_uniform', 'MDDN']

    ### Run different methods###
    run_method(Fail_SCE_PARAM, RC, LAYERS, method='INDP')
    run_method(Fail_SCE_PARAM, RC, LAYERS, method='TD_INDP')
    for jc in judge_types:
        run_method(Fail_SCE_PARAM, RC, LAYERS, method='JC', judgment_type=jc,
                   auction_type=None, valuation_type=None)
        for at in auction_types:
            for vt in valuation_types:
                run_method(Fail_SCE_PARAM, RC, LAYERS, method='JC',
                           judgment_type=jc, auction_type=at, valuation_type=vt)

    ### Compute metrics ###
    cost_type = 'Total'
    ref_method = 'indp'
    method_name = ['indp']
    for jc in judge_types:
        method_name.append('judgeCall_'+jc)
    # auction_types.append('Uniform')
    suffixes = ['Real_sum', '']
    sample_range = Fail_SCE_PARAM["sample_range"]
    mags = Fail_SCE_PARAM['mags']

    synthetic_dir = None #BASE_DIR+Fail_SCE_PARAM['TOPO']+'Networks/'
    combinations, optimal_combinations = Dindp.generate_combinations('shelby',
                mags, sample_range, LAYERS, RC, method_name, auction_types,
                valuation_types, listHDadd=None, synthetic_dir=synthetic_dir)

    root = OUTPUT_DIR #+Fail_SCE_PARAM['TOPO']+'/results/'
    df = Dindp.read_and_aggregate_results(combinations, optimal_combinations,
                                          suffixes, root_result_dir=root)
##    df = correct_tdindp_results(df, optimal_combinations)

    lambda_df = Dindp.relative_performance(df, combinations, optimal_combinations,
                                           ref_method=ref_method, cost_type=cost_type)
    resource_allocation, res_alloc_rel = Dindp.read_resourcec_allocation(df, combinations,
                optimal_combinations, root_result_dir=root, ref_method=ref_method)
    run_time_df = Dindp.read_run_time(combinations, optimal_combinations, suffixes,
                                      root_result_dir=root)

    ### Save Variables to file ###
    # object_list = [combinations, optimal_combinations, df, method_name, lambda_df,
    #                resource_allocation, res_alloc_rel, cost_type, run_time_df]
    # # Saving the objects:
    # with open(OUTPUT_DIR+'objs.pkl', 'w') as f:
    #     pickle.dump(object_list, f)

    # # Getting back the objects:
    # with open('./NOTS/objs.pkl') as f:  # Python 3: open(..., 'rb')
    #     [combinations, optimal_combinations, df, method_name, lambda_df, resource_allocation,
    #      res_alloc_rel, cost_type, run_time_df] = pickle.load(f)

    ### Plot results ###
    plt.close('all')

    plots.plot_performance_curves_shelby(df, cost_type='Total', decision_names=method_name,
                                         ci=None, normalize=True)
    plots.plot_relative_performance_shelby(lambda_df, lambda_type='TC')
    plots.plot_auction_allocation_shelby(resource_allocation, ci=None)
    plots.plot_relative_allocation_shelby(res_alloc_rel)
    plots.plot_run_time_synthetic(run_time_df, ci=None)

    plots.plot_performance_curves_synthetic(df, ci=None, x='t', y='cost', cost_type=cost_type)
    plots.plot_performance_curves_synthetic(df, ci=None, x='t', y='cost',
                                            cost_type='Under Supply Perc')
    plots.plot_relative_performance_synthetic(lambda_df, cost_type=cost_type,
                                              lambda_type='U')
    plots.plot_auction_allocation_synthetic(resource_allocation, ci=None,
                                            resource_type='normalized_resource')
    plots.plot_relative_allocation_synthetic(res_alloc_rel)
    plots.plot_run_time_synthetic(run_time_df, ci=None)
