# %%
"""# Interdependent Network Mitigation and Restoration Decision-making (Complete Analysis Dashboard) This notebook
finds mitigation actions and restoration plans for synthetic or infrastructure interdependent networks subject to
different initial seismic damage scenarios. Various restoration decision-making models are considered here:

* Centralized methods: These methods solve one optimization problem for the whole interdependent network, which leads
to the optimal mitigation and  restoration plan. Such models assume that the decision-maker is one entity that has
complete information and authority to restore all layers of the interdependent network. These methods build upon
Interdependent Network Design Problem (INDP) [cite] and time-dependent INDP (td-INDP) [cite].

"""

# %%
import os
import matplotlib.pyplot as plt
import runutils_v2
import dindputils_v2
import plots
import pickle

plt.close('all')

# %%
"""
## Run a toy example using different methods 
"""

# %%

# runutils_v2.run_sample_problems()

# %%
"""
## Run different methods for a given set of networks and a host of initial damage scenarios, post-process the outputs, 
and plot the result

### Input/Output file addresses
1. `BASE_DIR`: the address of the folder where the basic network information (topology, parameters, etc.) are stored
2. `DAMAGE_DIR`: the address of the folder where the damage information are stored
3. `OUTPUT_DIR`: the address of the folder where the output files should be written
4. `FILTER_SCE`(optional): The address of the list of scenarios that should be included in the analyses. It is used to 
remove less damaging scenarios from the list of damage scenarios. Set it to *None* if you don't want to use this option.
"""

# %%
BASE_DIR = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Node_arc_info_v2/"
# '/home/hesam/Desktop/Files/Generated_Network_Dataset_v4.1/'
# "../data/Extended_Shelby_County/"
# "../data/Extended_Shelby_County_dp/"
# 'C:/Users/ht20/Box Sync/Shelby County Database/Node_arc_info'
# "C:/Users/ht20/Documents/Files/Generated_Network_Dataset_v4.1/"
# "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Node_arc_info/"

DAMAGE_DIR = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Damage_scenarios/cumulative_250yr_initial_damage/"
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

FILTER_SCE = None
# '../data/damagedElements_sliceQuantile_0.90.csv'

# %%
''' 
### Set analysis dictionaries 
1. `FAIL_SCE_PARAM`: stores information on the type of initial damage scenarios. This dictionary should have the 
following items:
    1. `TYPE`: type of the initial damage network. Options are: `WU` (for the infrastructure of Shelby County, TN),
    `synthetic` (for the synthetic interdependent networks), and 'from_csv' (for general use).
    2. `L1_RANGE`: the damage scenarios for all datasets come in a two-level format. The implication of each level is 
    different based on the application. For example, for infrastructure damage scenarios, level 1 differentiates between 
    damge scenarios with different magnitudes, while level 2 includes samples of damage scenarios based on events with 
    the same magnitudes. `L1_RANGE` sets the  range of the scenarios in the first level that should be included in the 
    analysis.
    3. `L2_RANGE`: sets the range of the scenarios in the second level that should be included in the analysis.
    4. `BASE_DIR`: sets the folder where the basic network information is stored.
    5. `DAMAGE_DIR`: sets the folder where the damage information is stored
    6. `FILTER_SCE` (optional): sets a given list of scenarios that should be included in the analyses and exclude the 
    rest (mostly used with **WU format** below).
    7. `TOPO` (only when `TYPE`=*synthetic*): sets the topology of the synthetic networks that should be analyzed
    <br><br>
    Examples:
    * **WU**: this is the new format that is designed by Hesam Talebiyan and used in the Shelby County data paper 
    :cite:`Talebiyan2021`. The damage data for this dataset comes in a format similar to the hazard maps from Jason Wu 
    :cite:`Wu2017`, which consist of N sets (`L2_RANGE`) of M damage scenarios (`L1_RANGE`). For shelby county, 
    for example, N=50 and M=96. To use this format, set the dictionary to:<br>
    `{'TYPE':"WU", 'L1_RANGE':range(96), 'L2_RANGE':range(50),'FILTER_SCE':FILTER_SCE, 'BASE_DIR':BASE_DIR,
    'DAMAGE_DIR':DAMAGE_DIR}`
    * **from_csv**: for this type, the damage data come in the form of two csv files that contain all damage data
    for nodes and arcs. This is a more compressed representation of damage data. In this format, there is only one
    `L1_RANGE`=0, and `L2_RANGE` defines all scenarios that should be analyzed. To use this format, set the dictionary 
    to:<br>
    `{'TYPE':"from_csv", 'L2_RANGE':range(100), 'L1_RANGE':range(0, 1), 'FILTER_SCE':None, 'BASE_DIR':BASE_DIR,
    'DAMAGE_DIR':DAMAGE_DIR}`
    <br><br>
    :<br><br>
    * **synthetic**: this type is employed only for the synthetic network dataset. In this format, network data and 
    damage data are in the same folder, and hence, `BASE_DIR`= `DAMAGE_DIR`. Also, `L1_RANGE` represents the range of 
    network configuration, and `L2_RANGE` sets the range of sample networks of each configuration in the analysis.
    To use this format, set the dictionary to:<br>
    `{'TYPE':"synthetic", 'L2_RANGE':range(0, 1), 'L1_RANGE':range(0, 100), 'FILTER_SCE':None, 'TOPO':'General',
    'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}`
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
4. `EXTRA_COMMODITY`: Multi-commodity parameters dict
'''

# %%
# FAIL_SCE_PARAM = {'TYPE': "synthetic", 'L2_RANGE': range(5), 'L1_RANGE': range(100), 'TOPO': 'General',
#                   'BASE_DIR': BASE_DIR, 'FILTER_SCE': FILTER_SCE, 'DAMAGE_DIR': DAMAGE_DIR}
# FAIL_SCE_PARAM = {'TYPE': "WU", 'L2_RANGE': range(7), 'L1_RANGE': range(3), 'BASE_DIR': BASE_DIR,
#                   'DAMAGE_DIR': DAMAGE_DIR, 'FILTER_SCE': FILTER_SCE}
FAIL_SCE_PARAM = {'TYPE': "from_csv", 'L2_RANGE': range(0, 30), 'L1_RANGE': [250],
                  'FILTER_SCE': None, 'BASE_DIR': BASE_DIR, 'DAMAGE_DIR': DAMAGE_DIR}

DYNAMIC_PARAMS = None
# ROOT_DISLOC = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Joplin/"
# POP_DISLOC_DATA = ROOT_DISLOC+'Joplin_testbed/pop-dislocation-results.csv'
# DYNAMIC_PARAMS = {'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED':'joplin',
#                   'OUT_DIR': BASE_DIR, 'POP_DISLOC_DATA': POP_DISLOC_DATA ,
#                   'MAPPING': {'POWER': ROOT_DISLOC+'/Power/Joplin interdependency table - buildings,\
#                               substations, and poles/Joplin_interdependency_table.csv'}}
ROOT_DISLOC = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/"
DYNAMIC_PARAMS = {'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED': 'seaside', 'OUT_DIR': OUTPUT_DIR,
                  'POP_DISLOC_DATA': ROOT_DISLOC + 'Seaside_notebook/output/250yr/',
                  'MAPPING': {'POWER': ROOT_DISLOC + 'Power/bldgs2elec_Seaside.csv',
                              'WATER': ROOT_DISLOC + 'Water/bldgs2wter_Seaside.csv'}}

# Adjust output and base dir for synthetic database based on `FAIL_SCE_PARAM`
SYNTH_DIR = None
if FAIL_SCE_PARAM['TYPE'] == 'synthetic':
    SYNTH_DIR = BASE_DIR + FAIL_SCE_PARAM['TOPO'] + 'Networks/'
    OUTPUT_DIR += FAIL_SCE_PARAM['TOPO'] + '/results/'

EXTRA_COMMODITY = None  # {1: ['PW'], 3: []}

# %%
''' 
### Set analysis parameters 
1. `T`: number of time steps of the analysis. 
2. `RC`: list of resource caps or the number of available resources in each step of the
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
3. `LAYERS`: list of layers in the analysis. 
    * If `FAIL_SCE_PARAM[TYPE']`=*synthetic*, this item is not necessary. `LAYERS` is
    adjusted for each configuration. Set it to to `LAYERS`=[0]
'''

# %%
T = 15
RC = [{'budget': {t: 245733 for t in range(T)}, 'time': {t: 65 for t in range(T)}}]  # 349215*(+1/13*(t-1)+.5)
RC[0]['budget'][0] = 0
RC[0]['time'][0] = 0
LAYERS = [3]

# %%
''' 
### Run method(s)
There are ??? choices of method:
1. `INMRP`: runs Interdependent Network Mitigation and Restoration Problem (INMRP) with is based on time-dependent 
Interdependent Network Design Problem INDP (td-INDP) [cite]. To run this method,
you have to call:<br>
`runutils_v2.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INMRP', output_dir=OUTPUT_DIR, T=T,
misc = {'DYNAMIC_PARAMS':DYNAMIC_PARAMS, 'EXTRA_COMMODITY': EXTRA_COMMODITY, 'TIME_RESOURCE': False}))`
'''

# %%
runutils_v2.run_method(FAIL_SCE_PARAM, RC, T, LAYERS, method='INMRP', output_dir=OUTPUT_DIR,
                       misc={'DYNAMIC_PARAMS': DYNAMIC_PARAMS, 'EXTRA_COMMODITY': EXTRA_COMMODITY,
                             'TIME_RESOURCE': True})

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
# COST_TYPES = ['Total']  # 'Under Supply', 'Over Supply'
# REF_METHOD = 'dp_inmrp'
# METHOD_NAMES = ['dp_inmrp']
# # 'ng', 'jc', 'tdindp', 'ng', 'bgCCCCUUUU', 'dp_indp', 'dp_jc', 'bgCNUU', 'inmrp', 'dp_inmrp'
#
# COMBS, OPTIMAL_COMBS = dindputils_v2.generate_combinations(FAIL_SCE_PARAM['TYPE'], FAIL_SCE_PARAM['L1_RANGE'],
#                                                            FAIL_SCE_PARAM['L2_RANGE'], LAYERS, RC, METHOD_NAMES,
#                                                            list_high_dam_add=FAIL_SCE_PARAM['FILTER_SCE'],
#                                                            synthetic_dir=SYNTH_DIR)
#
# BASE_DF, objs = dindputils_v2.read_results(COMBS, OPTIMAL_COMBS, COST_TYPES, root_result_dir=OUTPUT_DIR, deaggregate=True)
#
# # LAMBDA_DF =  dindputils_v2.relative_performance(BASE_DF, COMBS, OPTIMAL_COMBS, ref_method=REF_METHOD,
# #                                             cost_type=COST_TYPES[0], deaggregate=True)
# # RES_ALLOC_DF, ALLOC_GAP_DF =  dindputils_v2.read_resource_allocation(BASE_DF, COMBS, OPTIMAL_COMBS,
# #                                                                   objs, root_result_dir=OUTPUT_DIR,
# #                                                                   ref_method=REF_METHOD)
# # RUN_TIME_DF =  dindputils_v2.read_run_time(COMBS, OPTIMAL_COMBS, objs, root_result_dir=OUTPUT_DIR)
# # ANALYZE_NE_DF = gameutils.analyze_NE(objs, COMBS, OPTIMAL_COMBS)
# # REL_ACTION_DF = gameutils.relative_actions(ANALYZE_NE_DF, COMBS)
#
# # COOP_GAIN, COOP_GAIN_TIME = gameutils.cooperation_gain(BASE_DF, LAMBDA_DF, COMBS, ref_state='bgNNUU',
# #                                                         states=['bgCCUU', 'bgCNUU', 'bgNCUU'])

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
# import pickle
#
# results_dir = OUTPUT_DIR
# with open(results_dir + 'postprocess_dicts.pkl', 'rb') as f:
#     [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES, LAMBDA_DF, RES_ALLOC_DF,
#      ALLOC_GAP_DF, RUN_TIME_DF, COST_TYPE, ANALYZE_NE_DF, REL_ACTION_DF] = pickle.load(f)
# plots.plot_performance_curves(BASE_DF,
#                               cost_type='Total', ci=95,
#                               deaggregate=False, plot_resilience=True)
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
