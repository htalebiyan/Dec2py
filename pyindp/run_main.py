# %%
'''
# Runs INDP, td-INDP, Judgment Call, and infrastructure games
'''

# %%
import runutils
import matplotlib.pyplot as plt

try:
    # Change the current working Directory
    DIR_MAIN = 'C:/Users/ht20/Documents/GitHub/td-DINDP/pyindp'
    os.chdir(DIR_MAIN)
    print("Directory changed to "+DIR_MAIN)
except OSError:
    print("Can't change the Current Working Directory")

# %%
''' 
## Run a toy example for different methods 
'''

# %%
# runutils.run_sample_problems()

# %%
''' 
## Run different methods for different networks, postprocess th outputs, and plot the result
### Input/Output file addresses
1. `BASE_DIR`: the address of the folder where the basic network information
    (topology, parameters, etc.) is stored
2. `DAMAGE_DIR`: the address of the folder where the damage information is stored
3. `OUTPUT_DIR`: the address of the folder where the output files should be written
4. `FILTER_SCE`(optional): The address of the list of scenarios that should be 
    included in the analyses. It is used to remove less damaging scenarios from
    the list of damage scenarios. Set it to *None* if you don't want to use this option.
5. `PAYOFF_DIR`(only for Games): The address of the folder that contaions the objects
    that store the payoff values for the game so that they are read from file and
    not calcualted again. Set it to *None* if you don't want to use this option.
'''

# %%
BASE_DIR = '/home/hesam/Desktop/Files/Generated_Network_Dataset_v4/'
# '/home/hesam/Desktop/Files/Generated_Network_Dataset_v4/'
# "../data/Extended_Shelby_County/"
# 'C:/Users/ht20/Box Sync/Shelby County Database/Node_arc_info'
# "C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\"
# "/home/hesam/Desktop/Files/Generated_Network_Dataset_v3.1"

DAMAGE_DIR = '/home/hesam/Desktop/Files/Generated_Network_Dataset_v4/'
# '/home/hesam/Desktop/Files/Generated_Network_Dataset_v4/'
# ../data/random_disruption_shelby/"
#"../data/Wu_Damage_scenarios/" 
# "C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\"
# "/home/hesam/Desktop/Files/Generated_Network_Dataset_v3.1"
# 'C:/Users/ht20/Box Sync/Shelby County Database/Damage_scenarios'

OUTPUT_DIR = '../results/'
# '/home/hesam/Desktop/Files/Game_Shelby_County/results_NE/'
# 'C:/Users/ht20/Documents/Files/Game_Shelby_County/results_0.9_perc/'
# 'C:/Users/ht20/Documents/Files/Auction_Extended_Shelby_County_Data/results/'
# '../results/'
# 'C:/Users/ht20/Documents/Files/Auction_synthetic_networks_v3.1/'
# 'C:/Users/ht20/Documents/Files/Shelby_data_paper/Restoration_results/'
# FAIL_SCE_PARAM['TOPO']+'/results/'

FILTER_SCE = '../data/damagedElements_sliceQuantile_0.90.csv'


### Directory with objects containing payoff values for games
PAYOFF_DIR = OUTPUT_DIR+'General/results/reudced_action_matrix_100/'
#'/home/hesam/Desktop/Files/Game_Shelby_County/results_NE_only_objs/'

# %%
''' 
### Set analysis dictionaries 
1. `FAIL_SCE_PARAM`: stores informatiom on the type of the damage scenario and 
    network dataset. This dictionary should have the following items:
    1. `TYPE`: type of the network. Options are `shelby` for the infrastructure network 
    of Shelby County, TN, and `synthetic` for the synthetic interdepndent dataset.
    2. `MAGS`: the damage scenarios for all datasets comes in a two-level format.
    The implication of each level is different as explained below. `MAGS` sets 
    the range of the scenarios in the first level that should be included in the 
    analysis.
    3. `SAMPLE_RANGE`: sets  the range of the scenarios in the second level that
    should be included in the analysis.
    4. `BASE_DIR`: sets the folder where the basic network information is stored.
    5. `DAMAGE_DIR`: sets the folder where the damage information is stored
    6. `FILTER_SCE` (optional): sets a given list of scenarios that should be 
    included in the analyses, and exculde the rest (mostly used with **WU format** below). 
    7. `TOPO` (only when `TYPE`=*synthetic*): sets the topology of the sunthetic networks that
    should be analyzed
    <br><br>
    When the network dataset is the infrastructure network of Shelby County, TN,
    there are three formats for network data and damage scenarios files:
    * **ANDRES**: this the old format that Andres Gonzalez employed during the 
    development of INDP, and is based on the input data for Xpress software.
    The network data that are available in this format are the first version Shleby 
    County dataset [cite] and the damage data are 1000 realizations of 
    hazard maps correponding to hypothetical earthquakes with Magnitudes 6 to 9
    occuring at a specific epicenter [cite] . To use this format, set the dictionary to:<br>
    `{'TYPE':"ANDRES", 'SAMPLE_RANGE':range(1, 1001), 'MAGS':[6, 7, 8, 9], 'FILTER_SCE':None,
      'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}`<br>
    Here, the range of magnitudes in the analysis is set by `MAGS`, and for each magnitude,
    the range of analyzed samples is set by `SAMPLE_RANGE`.
    * **WU**: this is the new format that is designed by Hesam Talebiyan and
    used in the Shelby County data paper [cite]. The damage data for this dataset comes
    in a format similar to the hazard maps from Jason Wu [cite], which consist of N
    sets (`SAMPLE_RANGE`) of M damage scenarios (`MAGS`). For shelby county, for example,
    N=50 and M=96. To use this format, set the dictionary to:<br>
    `{'TYPE':"WU", 'SAMPLE_RANGE':range(50), 'MAGS':range(96),'FILTER_SCE':FILTER_SCE,
      'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}`
    * **from_csv**: this type uses the same network data format as the **WU format**.
    However, the damage data come in the form of two csv files that contain all damage data
    for nodes and arcs. This is a more compressed representation of damage data. In this format,
    there is only one `MAGS`=0, and `SAMPLE_RANGE` defines all scenarios that should be analyzed.
    To use this format, set the dictionary to:<br>
    `{'TYPE':"from_csv", 'SAMPLE_RANGE':range(100), 'MAGS':range(0, 1), 'FILTER_SCE':None,
      'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}`
    <br><br>
    When the network dataset is synthetic, there are one format for network data and
    damage scenarios files:<br><br>
    * **synthetic**: in this format network data and damage data are in the same folder, and
    hence, `BASE_DIR`= `DAMAGE_DIR`. Also, `MAGS` represents the range of network configuration, and 
    `SAMPLE_RANGE` sets the range of sampele network for each configuraytion in the analysis.
    To use this format, set the dictionary to:<br>
    `{'TYPE':"synthetic", 'SAMPLE_RANGE':range(0, 1), 'MAGS':range(0, 100), 'FILTER_SCE':None,
      'TOPO':'General', 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}`
    <br><br>
2. `DYNAMIC_PARAMS`: sets the features of the models that incorporate dynamic parameters
    into the analysis. Set it to *None* if you want to use static paramters that are
    constant for different time steps. So far, we only have one type of dynamic paramters,
    which is the dynamic demand that is calculated based on population dislocation models, 
    for which, the dictionary should have the following items:
    1. `TYPE`: type of the dislocation data (see below).
    2. `RETURN`: type of the model for the return of dislocated population. Options
    are *step_function* and *linear*.
    3. `DIR`: sets the folder where the dislocation data are stored.
    4. `TESTBED` (only when `TYPE`=*incore*) : sets the name of the testbed in analysis.
    <br><br>
    The are two types of dislocation data:
    * **shelby_adopted**: this is a precalculated dictionary that stores stylized 
    dislocation  data for Shelby County dataset and the code reads those files.
    To use this type, set the dictionary to:<br> 
    `{'TYPE': 'shelby_adopted', 'RETURN': 'step_function', 'DIR': 'C:/Users/ht20/Documents/Files/dynamic_demand/'}`
    * **incore**: this type uses the population dislocation models and household
    unit allocation data from IN-CORE (stored locally) to calculate demand values 
    in each time step of the analysis. To use this type, set the dictionary to:<br>
    `{'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED':'Joplin', 'DIR': 'C:/Users/ht20/Documents/GitHub/NIST_testbeds/'}`
    <br><br>
3. `STM_MODEL_DICT`: contains information about the statistical models approximating INDP,
    which are used for valution methods in auction-based resource allocation. Set it to *None*
    if `VAL_TYPE` does not include *STM* (see below). Otherwise, the dictionary should have
    the following items:
    1. `num_pred`: number of model prediction that are used to calculate each valuation.
    2. `model_dir`: the folder that contains the statistical model files.
    3. `param_folder`: the folder that contains the statistical model parameters.<br>
    Example: <br>
    `MODEL_DIR = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc'
    STM_MODEL_DICT = {'num_pred':1, 'model_dir':MODEL_DIR+'/traces', 'param_folder':MODEL_DIR+'/parameters'}`
'''

# %%
FAIL_SCE_PARAM = {'TYPE':"synthetic", 'SAMPLE_RANGE':range(0, 1), 'MAGS':range(0, 100),
                  'FILTER_SCE':None, 'TOPO':'General', 'BASE_DIR':BASE_DIR,
                  'DAMAGE_DIR':DAMAGE_DIR}
DYNAMIC_PARAMS = None
STM_MODEL_DICT = None

# Adjust output and base dir for sythetic database based on `FAIL_SCE_PARAM`
SYNTH_DIR = None
if FAIL_SCE_PARAM['TYPE'] == 'synthetic':
    SYNTH_DIR = BASE_DIR+FAIL_SCE_PARAM['TOPO']+'Networks/'
    OUTPUT_DIR += FAIL_SCE_PARAM['TOPO']+'/results/'

# %%
''' 
### Set analysis parameters 
1. `RC`: list of resource caps or the number of available reousrces in each step of the 
    analysis. 
    * If `FAIL_SCE_PARAM[TYPE']`=*synthetic*, this item is not necessaary. `R_c` is
    adjusted for each configuration. Set it to to `R_c`=[0]
    * If `FAIL_SCE_PARAM[TYPE']`=*shelby*, you have to options.
        * if, for example, `R_c`= [3, 6, 8, 12], then the analysis is done for the cases
        when threr are 3, 6, 8, and 12 resources available (total reource assignment).
        * if, for example, `R_c`= [[1, 1], [1, 2], [3, 3]] and given there are 2 layers,
        then the analysis is done for the case where each layer gets 1 resource, AND
        the case where layer 1 gets 1 and layer 2 gets 2 resources, AND 
        the case where each layer gets 3 resource (Prescribed resource for each layer).
2. `LAYERS`: list of layers in the analysis. 
    * If `FAIL_SCE_PARAM[TYPE']`=*synthetic*, this item is not necessaary. `LAYERS` is
    adjusted for each configuration. Set it to to `LAYERS`=[0]
3. `JUDGE_TYPE`: list of judgment types that are used in JC method and/or computing valuations
    for auction-based allocation [cite]. Options are *OPTIMISTIC*, *PESSIMISTIC*, *DEMAND*,
    *DET-DEMAND*, and *RANDOM*. 
4. `RES_ALLOC_TYPE`: list of resource allocation types that are used in JC method [cite].
    Options are *MDA*, *MAA*, *MCA*, *UNIFORM*, and *OPTIMAL*. 
5. `VAL_TYPE`: list of valuation types that are used in auction-based resource allocation
    method [cite], i.e. when `RES_ALLOC_TYPE` includes at least one of the options *MDA*,
    *MAA*, or *MCA*.
    Options are *DTC*, *DTC_uniform*, *MDDN*, *STM*, and *DTC-LP*. 
'''

# %%
RC = [0]
LAYERS = [1, 2, 3, 4]
JUDGE_TYPE = ["OPTIMISTIC"]
RES_ALLOC_TYPE = ['UNIFORM', 'OPTIMAL']
VAL_TYPE = ['DTC']

# %%
''' 
### Run method(s)
There are five choices of method:
1. `INDP`: runs Interdependent Network Restoration Problem (INDP) [cite]. To run this method,
    you have to call:<br>
    `runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP', output_dir=OUTPUT_DIR,
    misc = {'DYNAMIC_PARAMS':DYNAMIC_PARAMS})`
2. `TDINDP`: runs time-dependent INDP (td-INDP) [cite]. To run this method,
    you have to call:<br>
    `runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='TDINDP', output_dir=OUTPUT_DIR,
    misc = {'DYNAMIC_PARAMS':DYNAMIC_PARAMS}))`
3. `JC`: runs Judgment Call (JC) method, which is a decentralized version of INDP [cite]. To run this method,
    you have to call:<br>
    `runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='JC', judgment_type=JUDGE_TYPE,
    res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE,
    output_dir=OUTPUT_DIR, dynamic_params=DYNAMIC_PARAMS,
    misc = {'STM_MODEL':STM_MODEL_DICT, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS})`
4. `NORMALGAME`: runs Interdependent Network Restoration Normal Game (INRNG), which is a
    decentralized version of INDP [cite]. To run this method, you have to call:<br>
    `runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='NORMALGAME', judgment_type=JUDGE_TYPE,
    res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR, 
    misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
   'REDUCED_ACTIONS':'EDM'}`<br>
    Here, `misc['REDUCED_ACTIONS']` sets the hueristic method to reduce the number of actions of
    each player to add Bounded Rationality to the analysis. Options are *ER* for exhasuting resources, 
    and *EDM* for expert decision maker.
5. `BAYESGAME`: runs Interdependent Network Restoration Bayesian Game (INRBG), which is a
    decentralized version of INDP [cite]. To run this method, you have to call:<br>
    `runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='BAYESGAME', judgment_type=JUDGE_TYPE,
    res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR,
    misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
    "SIGNALS":{1:'C', 2:'C'}, "BELIEFS":{1:'U', 2:'U'},
    'REDUCED_ACTIONS':'EDM'}`<br>
    Here, `misc['SIGNALS']` sets the actual type of each player in the game. Options are *C* for
    cooperative and *N* for non-cooperative.<br>
    Also, `misc['BELIEFS']` sets the belief of each player in the game. Options are *U* for
    uniformed belief, *F* for false consensus bias, and *I* for inverse false consensus bias.
'''

# %%
runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP', output_dir=OUTPUT_DIR,
                    misc = {'DYNAMIC_PARAMS':DYNAMIC_PARAMS})
# runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='TDINDP', output_dir=OUTPUT_DIR,
#                     misc = {'DYNAMIC_PARAMS':DYNAMIC_PARAMS})
# runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='JC', judgment_type=JUDGE_TYPE,
#                     res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE,
#                     output_dir=OUTPUT_DIR, dynamic_params=DYNAMIC_PARAMS,
#                     misc = {'STM_MODEL':STM_MODEL_DICT, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS})
runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='NORMALGAME', judgment_type=JUDGE_TYPE,
                    res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR, 
                    misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
                    'REDUCED_ACTIONS':'EDM'})
runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='BAYESGAME', judgment_type=JUDGE_TYPE,
            res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR,
            misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
                    "SIGNALS":{1:'C', 2:'C'}, "BELIEFS":{1:'U', 2:'U'},
                    'REDUCED_ACTIONS':'EDM'})

# %%
''' Post-processing '''

# %%
# COST_TYPES = ['Total'] # 'Under Supply', 'Over Supply'
# REF_METHOD = 'indp'
# METHOD_NAMES = ['indp','ng', 'bgCCUU'] 
# # #'ng', 'jc', 'dp_indp', 'tdindp',
# # #'bgNNNNUUUU','bgCCCCUUUU', 'bgCCNCUUUU', 'bgCCCCFFFF', 'bgNNNNFFFF', 'bgCCNCFFFF'
# # #'bgCCCCIIII','bgNNNNIIII', 'bgCCNCIIII',

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
# ANALYZE_NE_DF = gameutils.analyze_NE(objs, COMBS, OPTIMAL_COMBS)

# %%
''' Save Variables to file '''

# %%
# OBJ_LIST = [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF,
#             RES_ALLOC_DF, ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPES, ANALYZE_NE_DF]

# ## Saving the objects ###
# with open(OUTPUT_DIR+'postprocess_dicts.pkl', 'wb') as f:
#     pickle.dump(OBJ_LIST, f)

# %%
''' Plot results '''

# %%
# plt.close('all')
# ### Getting back the objects ###
# with open(OUTPUT_DIR+'postprocess_dicts.pkl', 'rb') as f:
#     [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
#       ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPE, ANALYZE_NE_DF] = pickle.load(f)

# plots.plot_performance_curves(BASE_DF,
#                               cost_type='Total', ci=95,
#                               deaggregate=False, plot_resilience=False)

# # plots.plot_seperated_perform_curves(BASE_DF, x='t', y='cost', cost_type='Total',
# #                                     ci=95, normalize=False)

# plots.plot_relative_performance(LAMBDA_DF, lambda_type='U')
# # plots.plot_auction_allocation(RES_ALLOC_DF, ci=95)
# # plots.plot_relative_allocation(ALLOC_GAP_DF, distance_type='gap')
# # plots.plot_run_time(RUN_TIME_DF, ci=95)
# plots.plot_ne_analysis(ANALYZE_NE_DF, ci=95) #[(ANALYZE_NE_DF['auction_type']!='UNIFORM')]
plots.plot_ne_cooperation(ANALYZE_NE_DF, ci=95)
# plots.plot_payoff_hist(ANALYZE_NE_DF, compute_payoff_numbers=True, outlier=False)

# # [(RUN_TIME_DF['auction_type']!='MDA')&(RUN_TIME_DF['auction_type']!='MAA')]
# [((BASE_DF['decision_type']=='bgCCCCUUUU')|\
#                                       (BASE_DF['decision_type']=='bgCCNCUUUU')|\
#                                       (BASE_DF['decision_type']=='bgNNNNUUUU')|\
#                                       (BASE_DF['decision_type']=='ng')|\
#                                       (BASE_DF['decision_type']=='indp'))&\
#                                       (BASE_DF['auction_type']=='OPTIMAL')&\
#                                       (BASE_DF['no_resources']<7)]