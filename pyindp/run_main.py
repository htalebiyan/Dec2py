""" Runs INDP, td-INDP, and Judgment Call """
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import indp
import dindp
import plots
import gametree

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
                              controlled_layers=params['L'], saveModel=False, print_cmd_line=False)
            elif params["ALGORITHM"] == "INFO_SHARE":
                indp.run_info_share(params, layers=params['L'], T=params["T"])
            elif params["ALGORITHM"] == "INRG":
                indp.run_inrg(params, layers=params['L'], player_ordering=player_ordering)
            elif params["ALGORITHM"] == "BACKWARDS_INDUCTION":
                gametree.run_backwards_induction(params["N"], i, players=params['L'],
                                                 player_ordering=player_ordering,
                                                 T=params["T"], outdir=params["OUTPUT_DIR"])
            elif params["ALGORITHM"] == "JC":
                dindp.run_judgment_call(params, save_jc_model=False, print_cmd=False)

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
#             run_judgment_call(params, layers=[1, 2], T=params["T"], saveJCModel=True,
#                               print_cmd=True)
#             plot_indp_sample(params,
#             folderSuffix='_auction_'+params["AUCTION_TYPE"]+'_'+params["VALUATION_TYPE"],
#             suffix="sum")

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
               res_alloc_type=None, valuation_type=None, output_dir='..'):
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
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":output_dir+'/results/indp_results',
                      "V":v, "T":1, 'L':layers, "ALGORITHM":"INDP"}
        elif method == 'JC':
            params = {"NUM_ITERATIONS":10,
                      "OUTPUT_DIR":output_dir+'/results/jc_results',
                      "V":v, "T":1, 'L':layers, "ALGORITHM":"JC",
                      "JUDGMENT_TYPE":judgment_type, "RES_ALLOC_TYPE":res_alloc_type,
                      "VALUATION_TYPE":valuation_type}
        elif method == 'TD_INDP':
            params = {"NUM_ITERATIONS":1, "OUTPUT_DIR":output_dir+'/results/tdindp_results',
                      "V":v, "T":10, "WINDOW_LENGTH":3, 'L':layers, "ALGORITHM":"INDP"}
        else:
            sys.exit('Wrong method name: '+method)
        batch_run(params, fail_sce_param)

if __name__ == "__main__":
    ### Run a toy example for different methods ###
    #    run_indp_sample()

    #: The address to the list of scenarios that should be included in the analyses.
    FILTER_SCE = '../data/damagedElements_sliceQuantile_0.95.csv'
    #: The address to the basic (topology, parameters, etc.) information of the network.
    BASE_DIR = "../data/Extended_Shelby_County/"
    #: The address to damge scenario data.
    DAMAGE_DIR = "../data/Wu_Damage_scenarios/" #random_disruption_shelby/"
    #: The address to where output are stored.
    OUTPUT_DIR = '../results/'
    #'C:/Users/ht20/Documents/Files/Auction_Extended_Shelby_County_Data/results/'
    ###############'../results/'#+FAIL_SCE_PARAM['TOPO']+'/results/'

    #: Informatiom on the ype of the failure scenario (Andres or Wu)
    #: and network dataset (shelby or synthetic)
    #: Help:
    #: For Andres scenario: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    #:     magnitudes: FAIL_SCE_PARAM['MAGS']
    #: For Wu scenario: set range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    #:     sce range: FAIL_SCE_PARAM['MAGS']
    #: For Synthetic nets: sample range: FAIL_SCE_PARAM['SAMPLE_RANGE'],
    #:     configurations: FAIL_SCE_PARAM['MAGS']
    FAIL_SCE_PARAM = {'TYPE':"WU", 'SAMPLE_RANGE':range(0, 2), 'MAGS':range(0, 95),
                      'FILTER_SCE':FILTER_SCE, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"ANDRES", 'SAMPLE_RANGE':range(1, 1001), 'MAGS':[6, 7, 8, 9],
    #                  'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"random", 'SAMPLE_RANGE':range(10, 11), 'MAGS':range(0, 1),
    #                   'FILTER_SCE':None, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
    # FAIL_SCE_PARAM = {'TYPE':"synthetic", 'SAMPLE_RANGE':range(0, 5), 'MAGS':range(0, 100),
    #                   'FILTER_SCE':None, 'TOPO':'Grid'
    #                   'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}

    # No restriction on number of resources for each layer
    RC = [3, 6]
    # Not necessary for synthetic nets
    # [3, 6, 8, 12]
    # [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]# Prescribed for each layer
    LAYERS = [1, 2, 3, 4]
    # Not necessary for synthetic nets
    JUDGE_TYPE = ["PESSIMISTIC", "OPTIMISTIC"]
    #["PESSIMISTIC", "OPTIMISTIC", "DEMAND", "DET-DEMAND", "RANDOM"]
    RES_ALLOC_TYPE = ["MDA", "MAA", "MCA", 'UNIFORM']
    #["MDA", "MAA", "MCA", 'UNIFORM']
    VAL_TYPE = ['DTC', 'MDDN']
    #['DTC', 'DTC_uniform', 'MDDN']

    ### Run different methods###
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP')
    # run_method(FAIL_SCE_PARAM, RC, LAYERS, method='TD_INDP')
    run_method(FAIL_SCE_PARAM, RC, LAYERS, method='JC', judgment_type=JUDGE_TYPE,
                res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE)

    # ### Post-processing ###
    COST_TYPES = ['Total']
    REF_METHOD = 'indp'
    METHOD_NAMES = ['indp', 'jc']
    SUFFIXES = ['real']

    # SYNTH_DIR = None #BASE_DIR+FAIL_SCE_PARAM['TOPO']+'Networks/'
    # COMBS, OPTIMAL_COMBS = dindp.generate_combinations('shelby',
    #             FAIL_SCE_PARAM['MAGS'], FAIL_SCE_PARAM['SAMPLE_RANGE'], LAYERS,
    #             RC, METHOD_NAMES, JUDGE_TYPE, RES_ALLOC_TYPE, VAL_TYPE, SUFFIXES,
    #             list_high_dam_add=FAIL_SCE_PARAM['FILTER_SCE'],
    #             synthetic_dir=SYNTH_DIR)

    # BASE_DF, objs = dindp.read_results(COMBS, OPTIMAL_COMBS, COST_TYPES,
    #                                    root_result_dir=OUTPUT_DIR, deaggregate=True)

    # LAMBDA_DF = dindp.relative_performance(BASE_DF, COMBS, OPTIMAL_COMBS,
    #                                        ref_method=REF_METHOD, cost_type=COST_TYPES[0])
    # RES_ALLOC_DF, ALLOC_GAP_DF = dindp.read_resourcec_allocation(BASE_DF, COMBS, OPTIMAL_COMBS,
    #                                                              objs, root_result_dir=OUTPUT_DIR,
    #                                                              ref_method=REF_METHOD)
    # RUN_TIME_DF = dindp.read_run_time(COMBS, OPTIMAL_COMBS, objs, root_result_dir=OUTPUT_DIR)

    # ### Save Variables to file ###
    # OBJ_LIST = [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF,
    #             RES_ALLOC_DF, ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPES]

    # Saving the objects:
    # with open(OUTPUT_DIR+'postprocess_dicts.pkl', 'wb') as f:
    #     pickle.dump(OBJ_LIST, f)

    # Getting back the objects:
    with open(OUTPUT_DIR+'/postprocess_dicts.pkl', 'rb') as f:
        [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
         ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPE] = pickle.load(f)

    ### Plot results ###
    plt.close('all')

    plots.plot_performance_curves_shelby(BASE_DF, cost_type='Total', ci=None,
                                         normalize=True, deaggregate=True,
                                         plot_resilience=True)

    plots.plot_seperated_perform_curves(BASE_DF[BASE_DF['valuation_type'] != 'MDDN'],
                                        x='t', y='cost', cost_type='Total',
                                        ci=None, normalize=False)

    plots.plot_relative_performance_shelby(LAMBDA_DF, lambda_type='p')
    plots.plot_auction_allocation_shelby(RES_ALLOC_DF, ci=None)
    plots.plot_relative_allocation_shelby(ALLOC_GAP_DF)
    plots.plot_run_time(RUN_TIME_DF, ci=95)

    plots.plot_performance_curves_synthetic(BASE_DF, ci=None, x='t', y='cost',
                                            cost_type=COST_TYPE)
    plots.plot_performance_curves_synthetic(BASE_DF, ci=None, x='t', y='cost',
                                            cost_type='Under Supply Perc')
    plots.plot_relative_performance_synthetic(LAMBDA_DF, cost_type=COST_TYPE,
                                              lambda_type='U')
    plots.plot_auction_allocation_synthetic(RES_ALLOC_DF, ci=None,
                                            resource_type='normalized_resource')
    plots.plot_relative_allocation_synthetic(ALLOC_GAP_DF)
    plots.plot_run_time(RUN_TIME_DF, ci=None)
