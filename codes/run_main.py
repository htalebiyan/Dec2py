# %%
"""
# Interdependent Network Restoration Decision-making (Complete Analysis Dashboard)
This notebook finds restoration plans for synthetic or real-world interdependent
networks subject to different initial seismic damage scenarios. Various restoration
decision-making models are considered here:

* Centralized methods: These methods solve one optimization problem for the whole
interdependent network, which leads to the optimal restoration plan. Such models
assume that the decision-maker is one entity that has complete information and
authority to restore all layers of the interdependent network. These methods include
Interdependent Network Design Problem (INDP) [cite] and time-dependent INDP (td-INDP) [cite].

* Decentralized methods: These methods model a multi-agent decision-making environment where
 each agent has the authority to restore a single layer, has complete information about
 her respective layer, minimal or no information about other layers. It is assumed that
 agents communicate poorly and not on time. This kind of methods includes Judgment Call (JC)
 method [cite] with and without Auction-based resource allocations [cite] and Interdependent
 Network Restoration Simultaneous Games (INRSG) and Bayesian Games (INRBG) [cite].
"""

# %%
import os
import matplotlib.pyplot as plt
import runutils
import dindputils
import gameutils
import plots
import pickle

# %%
"""
## Run a toy example using different methods 
"""

# %%
# plt.close('all')
# runutils.run_sample_problems()

# %%
"""
## Run different methods for different networks, post-process the outputs, and plot the result
### Input/Output file addresses
1. `BASE_DIR`: the address of the folder where the basic network information (topology, parameters, etc.) are stored
2. `DAMAGE_DIR`: the address of the folder where the damage information are stored
3. `OUTPUT_DIR`: the address of the folder where the output files should be written
4. `FILTER_SCE`(optional): The address of the list of scenarios that should be included in the analyses. It is used to 
remove less damaging scenarios from the list of damage scenarios. Set it to *None* if you don't want to use this option.
5. `PAYOFF_DIR`(optional, only for Games): The address of the folder that contains the objects that store the payoff 
values for the game so that they are read from file and not calculated again. Set it to *None* if you don't want to use 
this option.
"""

# %%
BASE_DIR = "../data/Extended_Shelby_County/"
# '/home/hesam/Desktop/Files/Generated_Network_Dataset_v4.1/'
# "../data/Extended_Shelby_County/"
# 'C:/Users/ht20/Box Sync/Shelby County Database/Node_arc_info'
# "C:/Users/ht20/Documents/Files/Generated_Network_Dataset_v4.1/"
# "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Node_arc_info/"

DAMAGE_DIR = "../data/Wu_Damage_scenarios/"
# '/home/hesam/Desktop/Files/Generated_Network_Dataset_v4.1/'
# ../data/random_disruption_shelby/"
# "../data/Wu_Damage_scenarios/"
# "C:/Users/ht20/Documents/Files/Generated_Network_Dataset_v4.1/"
# 'C:/Users/ht20/Box Sync/Shelby County Database/Damage_scenarios'
# "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Damage_scenarios/eq_1000yr_initial_damage/"

OUTPUT_DIR = '../results/'
# '/home/hesam/Desktop/Files/Game_synthetic/v4.1/results_temp/'
# '/home/hesam/Desktop/Files/Game_Shelby_County/results/'
# 'C:/Users/ht20/Documents/Files/Auction_Extended_Shelby_County_Data/results/'
# '../results/'
# 'C:/Users/ht20/Documents/Files/Game_synthetic/v4.1/results_temp/'
# 'C:/Users/ht20/Documents/Files/Game_Shelby_County/results/'
# 'C:/Users/ht20/Documents/Files/Shelby_data_paper/Restoration_results/'
# FAIL_SCE_PARAM['TOPO']+'/results/'

FILTER_SCE = '../data/damagedElements_sliceQuantile_0.90.csv'
# '../data/damagedElements_sliceQuantile_0.90.csv'

PAYOFF_DIR = None
# OUTPUT_DIR + 'General/results/reduced_action_matrix_100/'
# '/home/hesam/Desktop/Files/Game_Shelby_County/results_NE_only_objs/'

# %%
''' 
### Set analysis dictionaries 
1. `FAIL_SCE_PARAM`: stores information on the type of the damage scenario and network 
dataset. This dictionary should have the following items:
    1. `TYPE`: type of the network. Options are `shelby` for the infrastructure network
    of Shelby County, TN, and `synthetic` for the synthetic interdependent dataset.
    2. `MAGS`: the damage scenarios for all datasets come in a two-level format. The
    implication of each level is different, as explained below. `MAGS` sets the 
    range of the scenarios in the first level that should be included in the analysis.
    3. `SAMPLE_RANGE`: sets the range of the scenarios in the second level that should be 
    included in the analysis.
    4. `BASE_DIR`: sets the folder where the basic network information is stored.
    5. `DAMAGE_DIR`: sets the folder where the damage information is stored
    6. `FILTER_SCE` (optional): sets a given list of scenarios that should be
    included in the analyses and exclude the rest (mostly used with **WU format** below).
    7. `TOPO` (only when `TYPE`=*synthetic*): sets the topology of the synthetic 
    networks that should be analyzed
    <br><br>
    When the network dataset is the infrastructure network of Shelby County, TN,
    there are three formats for network data and damage scenarios files:
    * **ANDRES**: this the old format that Andres Gonzalez employed during the 
    development of INDP, and is based on the input data for Xpress software.
    The network data that are available in this format are the first version Shelby
    County dataset [cite] and the damage data are 1000 realizations of
    hazard maps corresponding to hypothetical earthquakes with Magnitudes 6 to 9
    occurring at a specific epicenter [cite] . To use this format, set the dictionary to:<br>
    `{'TYPE':"ANDRES", 'SAMPLE_RANGE':range(1, 1001), 'MAGS':[6, 7, 8, 9], 'FILTER_SCE':None,
    'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}`<br>
    Here, the range of magnitudes in the analysis is set by `MAGS`, and for each magnitude,
    the range of analyzed samples is set by `SAMPLE_RANGE`.
    * **WU**: this is the new format that is designed by Hesam Talebiyan and
    used in the Shelby County data paper :cite:`Talebiyan2021`. The damage data for this dataset comes
    in a format similar to the hazard maps from Jason Wu :cite:`Wu2017`, which consist of N
    sets (`SAMPLE_RANGE`) of M damage scenarios (`MAGS`). For shelby county, for example,
    N=50 and M=96. To use this format, set the dictionary to:<br>
    `{'TYPE':"WU", 'SAMPLE_RANGE':range(50), 'MAGS':range(96),'FILTER_SCE':FILTER_SCE,
    'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}`
    * **from_csv**: this type uses the same network data format as the **WU format**.
    However, the damage data come in the form of two csv files that contain all damage data
    for nodes and arcs. This is a more compressed representation of damage data. In this
    format, there is only one `MAGS`=0, and `SAMPLE_RANGE` defines all scenarios that
    should be analyzed. To use this format, set the dictionary to:<br>
    `{'TYPE':"from_csv", 'SAMPLE_RANGE':range(100), 'MAGS':range(0, 1), 'FILTER_SCE':None,
    'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}`
    <br><br>
    When the network dataset is synthetic, there is one format for network data and
    damage scenarios files:<br><br>
    * **synthetic**: in this format, network data and damage data are in the same folder, and
    hence, `BASE_DIR`= `DAMAGE_DIR`. Also, `MAGS` represents the range of network configuration, and
    `SAMPLE_RANGE` sets the range of sample networks for each configuration in the analysis.
    To use this format, set the dictionary to:<br>
    `{'TYPE':"synthetic", 'SAMPLE_RANGE':range(0, 1), 'MAGS':range(0, 100), 'FILTER_SCE':None,
    'TOPO':'General', 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}`
    <br><br>
2. `DYNAMIC_PARAMS`: sets the features of the models that incorporate dynamic parameters
into the analysis. Set it to *None* if you want to use static parameters that are
constant for different time steps. So far, we only have one type of dynamic parameters,
which is the dynamic demand that is calculated based on population dislocation models,
for which, the dictionary should have the following items:
    1. `TYPE`: type of the dislocation data (see below).
    2. `RETURN`: type of the model for the return of the dislocated population. Options
    are *step_function* and *linear*.
    3. `DIR`: sets the folder where the dislocation data are stored.
    4. `TESTBED` (only when `TYPE`=*incore*) : sets the name of the testbed in analysis.
    <br><br>
    The are two types of dislocation data:
    * **shelby_adopted**: this is a precalculated dictionary that stores stylized
    dislocation data for the Shelby County dataset, and the code reads those files.
    To use this type, set the dictionary to:<br> 
    `{'TYPE': 'shelby_adopted', 'RETURN': 'step_function',
    'DIR': 'C:/Users/ht20/Documents/Files/dynamic_demand/'}`
    * **incore**: this type uses the population dislocation models and household
    unit allocation data from IN-CORE (stored locally) to calculate demand values 
    in each time step of the analysis. To use this type, set the dictionary to:<br>
    `{'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED':'Joplin',
    'DIR': 'C:/Users/ht20/Documents/GitHub/NIST_testbeds/'}`
    <br><br>
3. `STM_MODEL_DICT`: contains information about the statistical models approximating
INDP used for valuation methods in auction-based resource allocation. Set it to *None*
if `VAL_TYPE` does not include *STM* (see below). Otherwise, the dictionary should have
the following items:
    1. `num_pred`: number of model predictions that are used to calculate each valuation.
    2. `model_dir`: the folder that contains the statistical model files.
    3. `param_folder`: the folder that contains the statistical model parameters.<br>
    Example: <br>
    `MODEL_DIR = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc'
    STM_MODEL_DICT = {'num_pred':1, 'model_dir':MODEL_DIR+'/traces',
    'param_folder':MODEL_DIR+'/parameters'}`
4. `EXTRA_COMMODITY`: Multicommodity parameters dict
'''

# %%
# FAIL_SCE_PARAM = {'TYPE': "synthetic", 'SAMPLE_RANGE': range(5), 'MAGS': range(100),
#                   'FILTER_SCE': FILTER_SCE, 'TOPO': 'General', 'BASE_DIR': BASE_DIR,
#                   'DAMAGE_DIR': DAMAGE_DIR}
FAIL_SCE_PARAM = {'TYPE': "WU", 'SAMPLE_RANGE': range(50), 'MAGS': range(96),
                  'FILTER_SCE': FILTER_SCE, 'BASE_DIR': BASE_DIR, 'DAMAGE_DIR': DAMAGE_DIR}
# FAIL_SCE_PARAM = {'TYPE': "from_csv", 'SAMPLE_RANGE': range(0, 1), 'MAGS': [1000],
#                   'FILTER_SCE': None, 'BASE_DIR': BASE_DIR, 'DAMAGE_DIR': DAMAGE_DIR}

DYNAMIC_PARAMS = None
# DYNAMIC_PARAMS = {'TYPE': 'shelby_adopted', 'RETURN': 'step_function',
#                   'DIR': 'C:/Users/ht20/Documents/Files/dynamic_demand/'}

# ROOT_DISLOC = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Joplin/"
# POP_DISLOC_DATA = ROOT_DISLOC+'Joplin_testbed/pop-dislocation-results.csv'
# DYNAMIC_PARAMS = {'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED':'joplin',
#                   'OUT_DIR': BASE_DIR, 'POP_DISLOC_DATA': POP_DISLOC_DATA ,
#                   'MAPPING': {'POWER': ROOT_DISLOC+'/Power/Joplin interdependency table - buildings,\
#                               substations, and poles/Joplin_interdependency_table.csv'}}

# ROOT_DISLOC = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/"
# DYNAMIC_PARAMS = {'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED': 'seaside',
#                   'OUT_DIR': ROOT_DISLOC + 'Dislocation_models/',
#                   'POP_DISLOC_DATA': ROOT_DISLOC + 'Dislocation_models/',
#                   'MAPPING': {'POWER': ROOT_DISLOC + 'Power/bldgs2elec_Seaside.csv',
#                               'WATER': ROOT_DISLOC + 'Water/bldgs2wter_Seaside.csv'}}

STM_MODEL_DICT = None

# Adjust output and base dir for synthetic database based on `FAIL_SCE_PARAM`
SYNTH_DIR = None
if FAIL_SCE_PARAM['TYPE'] == 'synthetic':
    SYNTH_DIR = BASE_DIR + FAIL_SCE_PARAM['TOPO'] + 'Networks/'
    OUTPUT_DIR += FAIL_SCE_PARAM['TOPO'] + '/results/'

EXTRA_COMMODITY = None
# {1:['PW'], 3:[]}
# %%
''' 
### Set analysis parameters 
1. `RC`: list of resource caps or the number of available resources in each step of the
analysis. Each item of the list is a dictionary whose items show the type of resource and the available number of that
type of resource. For example:
    * If `FAIL_SCE_PARAM[TYPE']`=*synthetic*, this item is not necessary since `R_c` is
    adjusted for each configuration. Set it to to `R_c`=[0]
    * If `FAIL_SCE_PARAM[TYPE']`=*WU* or *ANDRES* or *from_csv*, you have two options:
    * if, for example, `R_c`= [{'budget': 3}, {'budget': 6}], then the analysis is done for the cases
    when there are 3 and 6 resources available of type 'budget'  (total resource assignment). If the name of resource is 
    set to '', the results will be consistent with older version of the code, where only one type of resource was 
    considered.
    * if, for example, `R_c`= [{'budget': {1:1, 2:1}}, {'budget': {1:1, 2:2}}, {'budget': {1:3, 2:3}}] and given there 
    are 2 layers, then the analysis is done for the case where each layer gets 1 resource of type 'budget', AND
    the case where layer 1 gets 1 and layer 2 gets 2 resources of type 'budget', AND 
    the case where each layer gets 3 resource of type 'budget' (Prescribed resource for each layer).
2. `LAYERS`: list of layers in the analysis. 
    * If `FAIL_SCE_PARAM[TYPE']`=*synthetic*, this item is not necessary. `LAYERS` is
    adjusted for each configuration. Set it to to `LAYERS`=[0]
3. `JUDGE_TYPE`: list of judgment types that are used in JC method and/or computing valuations
for auction-based allocation [cite]. Options are *OPTIMISTIC*, *PESSIMISTIC*, *DEMAND*,
*DET-DEMAND*, and *RANDOM*. 
4. `RES_ALLOC_TYPE`: list of resource allocation types that are used in JC method [cite].
Options are *MDA*, *MAA*, *MCA*, *UNIFORM*, and *OPTIMAL*. 
5. `VAL_TYPE`: list of valuation types that are used in auction-based resource allocation
method [cite], i.e. when `RES_ALLOC_TYPE` includes at least one of the options *MDA*,
*MAA*, or *MCA*. Options are *DTC*, *DTC_uniform*, *MDDN*, *STM*, and *DTC-LP*. 
'''

# %%
RC = [{'': 3}, {'': 6}]
# [{'budget': 120000, 'time': 35}], [{'': 3}]
# Prescribed for each layer -> RC = [{'budget':{1:60000, 3:700}, 'time':{1:2, 3:10}}] 

LAYERS = [1, 2, 3, 4]  # [1, 2, 3, 4]
JUDGE_TYPE = ["OPTIMISTIC"]
RES_ALLOC_TYPE = ['OPTIMAL']  # 'OPTIMAL', 'UNIFORM'
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
3. `JC`: runs Judgment Call (JC) method, which is a decentralized version of INDP [cite]. 
To run this method,	you have to call:<br>
`runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='JC', judgment_type=JUDGE_TYPE,
res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE
output_dir=OUTPUT_DIR, dynamic_params=DYNAMIC_PARAMS,
misc = {'STM_MODEL':STM_MODEL_DICT, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS})`
4. `NORMALGAME`: runs Interdependent Network Restoration Normal Game (INRNG), which is a
decentralized version of INDP [cite]. To run this method, you have to call:<br>
`runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='NORMALGAME', judgment_type=JUDGE_TYPE,
res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR,
misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS, 'REDUCED_ACTIONS':'EDM'}`<br>
Here, `misc['REDUCED_ACTIONS']` sets the heuristic method to reduce the number of actions of
each player to add Bounded Rationality to the analysis. Options are *ER* for exhausting resources 
and *EDM* for the expert decision-maker.
5. `BAYESGAME`: runs Interdependent Network Restoration Bayesian Game (INRBG), which is a
decentralized version of INDP [cite]. To run this method, you have to call:<br>
`runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='BAYESGAME', judgment_type=JUDGE_TYPE,
res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR,
misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
"SIGNALS":{1:'C', 2:'C'}, "BELIEFS":{1:'U', 2:'U'}, 'REDUCED_ACTIONS':'EDM'}`<br>
Here, `misc['SIGNALS']` sets the actual type of each player in the game. Options
are *C* for	cooperative and *N* for non-cooperative.<br>
Also, `misc['BELIEFS']` sets the belief of each player in the game. Options are *U* for
uniformed belief, *F* for false consensus bias, and *I* for inverse false consensus bias.
'''

# %%
print('EDM considers 10 actions')
print('CF is 2 ')

# runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP', output_dir=OUTPUT_DIR,
#                     misc={'DYNAMIC_PARAMS': DYNAMIC_PARAMS, 'EXTRA_COMMODITY': EXTRA_COMMODITY, 'TIME_RESOURCE': False})
# runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='TDINDP', output_dir=OUTPUT_DIR,
#                     misc={'DYNAMIC_PARAMS': DYNAMIC_PARAMS, 'EXTRA_COMMODITY': EXTRA_COMMODITY, 'TIME_RESOURCE': True})
# runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='JC', judgment_type=JUDGE_TYPE,
#                     res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR,
#                     misc={'STM_MODEL': STM_MODEL_DICT, 'DYNAMIC_PARAMS': DYNAMIC_PARAMS,
#                           'EXTRA_COMMODITY': EXTRA_COMMODITY, 'TIME_RESOURCE': True})
# runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='NORMALGAME', judgment_type=JUDGE_TYPE,
#             res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR,
#             misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
#                     'EXTRA_COMMODITY':EXTRA_COMMODITY,  'TIME_RESOURCE': False, 'REDUCED_ACTIONS': 'EDM'})
# runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='BAYESGAME', judgment_type=JUDGE_TYPE,
#             res_alloc_type=RES_ALLOC_TYPE, valuation_type=VAL_TYPE, output_dir=OUTPUT_DIR,
#             misc = {'PAYOFF_DIR':PAYOFF_DIR, 'DYNAMIC_PARAMS':DYNAMIC_PARAMS,
#                     'EXTRA_COMMODITY':EXTRA_COMMODITY, 'TIME_RESOURCE': False, 'REDUCED_ACTIONS': 'EDM',
#                     "SIGNALS":{x:'R' for x in LAYERS}, "BELIEFS":{x:'U' for x in LAYERS}})

# %%
''' 
### Post-processing 
First, you have to set a few parameters and then call functions that read outputs
and generate the pandas DataFrames that are needed for plotting the results.

##### Post-processing parameters

1. `COST_TYPES`: type of cost that should be used in processing the outputs. Options 
are *Total*, *Under Supply*, *Over Supply*, *Node*, *Arc*, *Flow*, *Space Prep*, *Under Supply Perc*.
2. `REF_METHOD`: the method served as the reference in computing the relative performance
and allocation gap. Usually, this is an optimal method like `indp` or `tdindp`. However,
it can be any other method like `jc`, `ng`, or else.
3. `METHOD_NAMES`: methods whose output should be read. Options are `indp`, `tdindp`, `jc`,
`ng`, `dp_indp`, `dp_jc`, `bg????` (For example, `bgNCUI` means the Bayesian game with
two players where the first player is non-cooperative and uses uninformative belief,
and the second one is cooperative and uses the inverse false consensus belief).

##### Post-processing functions

1. `generate_combinations`: generate all the combination of outputs that should be read and
save them in `COMBS` and `OPTIMAL_COMBS` lists.
2. `read_results`: read results for combinations in `COMBS` and `OPTIMAL_COMBS` lists.
3. `relative_performance`: computes relative performance measures for different combinations.
4. `read_resource_allocation`: read the resource allocations by different methods and
compute allocation gaps for different combinations.
5. `read_run_time`: compute run time for different combinations.
6. `analyze_NE`: analyze the characteristics of Nash equilibria for different combinations.
7. `relative_actions`: computes the relative usage of different action types compared to the
optimal solution.
8. `cooperation_gain`: computes the gain for each player from chainging thier types.
'''

# %%
COST_TYPES = ['Total']  # 'Under Supply', 'Over Supply'
REF_METHOD = 'indp'
METHOD_NAMES = ['indp', 'bgRRRRUUUU']
# 'ng', 'jc', 'tdindp', 'ng', 'bgCCCCUUUU', 'dp_indp', 'dp_jc', 'bgCNUU',

# COMBS, OPTIMAL_COMBS = dindputils.generate_combinations(FAIL_SCE_PARAM['TYPE'],
#                                                         FAIL_SCE_PARAM['MAGS'], FAIL_SCE_PARAM['SAMPLE_RANGE'], LAYERS,
#                                                         RC, METHOD_NAMES, JUDGE_TYPE, RES_ALLOC_TYPE, VAL_TYPE,
#                                                         list_high_dam_add=FAIL_SCE_PARAM['FILTER_SCE'],
#                                                         synthetic_dir=SYNTH_DIR)

# BASE_DF, objs = dindputils.read_results(COMBS, OPTIMAL_COMBS, COST_TYPES,
#                                         root_result_dir=OUTPUT_DIR, deaggregate=True)

# LAMBDA_DF = dindputils.relative_performance(BASE_DF, COMBS, OPTIMAL_COMBS, ref_method=REF_METHOD,
#                                             cost_type=COST_TYPES[0], deaggregate=True)
# RES_ALLOC_DF, ALLOC_GAP_DF = dindputils.read_resource_allocation(BASE_DF, COMBS, OPTIMAL_COMBS,
#                                                                   objs, root_result_dir=OUTPUT_DIR,
#                                                                   ref_method=REF_METHOD)
# RUN_TIME_DF = dindputils.read_run_time(COMBS, OPTIMAL_COMBS, objs, root_result_dir=OUTPUT_DIR)
# ANALYZE_NE_DF = gameutils.analyze_NE(objs, COMBS, OPTIMAL_COMBS)
# REL_ACTION_DF = gameutils.relative_actions(ANALYZE_NE_DF, COMBS)

# COOP_GAIN, COOP_GAIN_TIME = gameutils.cooperation_gain(BASE_DF, LAMBDA_DF, COMBS, ref_state='bgNNUU',
#                                                         states=['bgCCUU', 'bgCNUU', 'bgNCUU'])

# %%
''' 
### Save Variables to file
All dictionaries that are made in the postprocessing step are saved here.
'''

# %%
# OBJ_LIST = [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
#             ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPES, ANALYZE_NE_DF, REL_ACTION_DF]
# OBJ_LIST = [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF]
# with open(OUTPUT_DIR + 'postprocess_dicts.pkl', 'wb') as f:
#     pickle.dump(OBJ_LIST, f)

# %%
''' 
### Plot results 
Plot functions use the dictionaries that are made in the postprocessing step to 
make output figures:
1. `plot_performance_curves`: plots costs (in `COST_TYPES`) and unmet demand vs. time.
2. `plot_separated_perform_curves`: plots costs (in `COST_TYPES`) vs. time for each layer sepearately.
3. `plot_relative_performance`: plots relative performances.
4. `plot_auction_allocation`: plots resource allocation vs. time.
5. `plot_relative_allocation`: plots allocation gaps.
6. `plot_run_time`: plots run time vs. time.
7. `plot_ne_analysis`: plots NE analysis measures vs. time (for games only).
8. `plot_ne_cooperation`: plots action types vs. time (for games only).
9. `plot_payoff_hist`: plots size of the payoff matrix vs. time (for games only).
10. `plot_relative_actions`: plots relative action usage (for games only).
'''

# %%
plt.close('all')
### Getting back the objects ###
import pickle

results_dir = OUTPUT_DIR
with open(results_dir + 'postprocess_dicts.pkl', 'rb') as f:
    [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
     ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPE, ANALYZE_NE_DF, REL_ACTION_DF] = pickle.load(f)
plots.plot_performance_curves(BASE_DF,
                              cost_type='Total', ci=None,
                              deaggregate=False, plot_resilience=True)
# plots.plot_relative_performance(LAMBDA_DF[(LAMBDA_DF['auction_type'] != 'UNIFORM') & \
#                                           ((LAMBDA_DF['no_resources'] != 8)&(LAMBDA_DF['no_resources'] != 12))],
#                                 lambda_type='U')
# plots.plot_ne_analysis(ANALYZE_NE_DF, ci=None)
# plots.plot_ne_cooperation(ANALYZE_NE_DF, ci=None)
# plots.plot_relative_actions(REL_ACTION_DF)

# plots.plot_cooperation_gain(COOP_GAIN, ref_state = 'bgNNUU',
#                             states = ['bgCCUU', 'bgCNUU', 'bgNCUU'])

# # # plots.plot_separated_perform_curves(BASE_DF, x='t', y='cost', cost_type='Total',
# # #                                     ci=95, normalize=False)
# # plots.plot_auction_allocation(RES_ALLOC_DF, ci=95)
# # plots.plot_relative_allocation(ALLOC_GAP_DF, distance_type='gap')
# # plots.plot_run_time(RUN_TIME_DF, ci=95)
# plots.plot_payoff_hist(ANALYZE_NE_DF, compute_payoff_numbers=True, outlier=False)

# [(REL_ACTION_DF['auction_type']!='UNIFORM')]
