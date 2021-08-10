# %%
"""
# Interdependent Network Restoration Decision-making
This notebook finds restoration plans for interdependent power and water in Seaside, OR, subject to different initial
seismic damage scenarios. Centralized decision-making models are employed here. These methods solve one optimization
problem for the whole interdependent network, which leads to the optimal restoration plan. Such models assume that the
decision-maker is one entity that has complete information and authority to restore all layers of the interdependent
network. These methods include Interdependent Network Design Problem (INDP) and time-dependent INDP (td-INDP).
"""

# %%
import runutils
import dindputils
import plots
import pickle

# %%
"""
## Input/Output file addresses
1. `BASE_DIR`: the address of the folder where the basic network information (topology, parameters, etc.) are stored
2. `DAMAGE_DIR`: the address of the folder where the damage information are stored
3. `OUTPUT_DIR`: the address of the folder where the output files should be written
"""

# %%
BASE_DIR = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Node_arc_info/"
DAMAGE_DIR = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Damage_scenarios/eq_1000yr_initial_damage/"
OUTPUT_DIR = '../results/'

# %%
""" 
## Set analysis dictionaries 
1. `FAIL_SCE_PARAM`: stores information on the type of the damage scenario and network 
dataset. This dictionary should have the following items:
    1. `TYPE`: type of the network, which is set to `from_csv` for Seaside networks.
    2. `MAGS`: sets the earthquake return period.
    3. `SAMPLE_RANGE`: sets the range of sample scenarios to be analyzed.
    4. `BASE_DIR`: sets the folder where the basic network information is stored.
    5. `DAMAGE_DIR`: sets the folder where the damage information is stored
2. `DYNAMIC_PARAMS`: sets the features of the models that incorporate dynamic demand values (per dislocation models)
into the analysis. Set it to *None* if you want to use static demand values (equal to the pre-event values) that are
constant for different time steps. The dictionary should have the following items:
    1. `TYPE`: type of the dislocation data (see below).
    2. `RETURN`: type of the model for the return of the dislocated population. Options: *step_function* and *linear*.
    3. `DIR`: sets the folder where the dislocation data are stored.
    4. `TESTBED`: sets the name of the testbed in analysis.
3. `EXTRA_COMMODITY`: Multi-commodity parameters dict
"""

# %%
FAIL_SCE_PARAM = {'TYPE': "from_csv", 'SAMPLE_RANGE': range(0, 1), 'MAGS': [1000],
                  'FILTER_SCE': None, 'BASE_DIR': BASE_DIR, 'DAMAGE_DIR': DAMAGE_DIR}
ROOT_DISLOC = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/"
DYNAMIC_PARAMS = {'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED': 'seaside',
                  'OUT_DIR': ROOT_DISLOC + 'Dislocation_models/',
                  'POP_DISLOC_DATA': ROOT_DISLOC + 'Dislocation_models/',
                  'MAPPING': {'POWER': ROOT_DISLOC + 'Power/bldgs2elec_Seaside.csv',
                              'WATER': ROOT_DISLOC + 'Water/bldgs2wter_Seaside.csv'}}
EXTRA_COMMODITY = {1: ['PW'], 3: []}

# %%
"""
## Set analysis parameters 
1. `RC`: list of resource caps or the number of available resources in each step of the analysis. Each item of the list 
is a dictionary whose items show the type of resource and the available number of that type of resource. For example:
    * If `FAIL_SCE_PARAM[TYPE']`=*from_csv*, you have two options:
    * if, for example, `R_c`= [{'budget': 3}, {'budget': 6}], then the analysis is done for the cases
    when there are 3 and 6 resources available of type 'budget' (total resource assignment).
    * if, for example, `R_c`= [{'budget': {1:1, 2:1}}, {'budget': {1:1, 2:2}}, {'budget': {1:3, 2:3}}] and given there 
    are 2 layers, then the analysis is done for the case where each layer gets 1 resource of type 'budget', AND
    the case where layer 1 gets 1 and layer 2 gets 2 resources of type 'budget', AND 
    the case where each layer gets 3 resource of type 'budget' (Prescribed resource for each layer).
2. `LAYERS`: list of layers in the analysis. 
"""

# %%
RC = [{'budget': 120000, 'time': 70}]
# Prescribed for each layer -> RC = [{'budget':{1:60000, 3:700}, 'time':{1:2, 3:10}}]
LAYERS = [1, 3]

# %%
""" 
## Run method(s)
There are two choices of method:
1. `INDP`: runs Interdependent Network Restoration Problem (INDP). To run this method, you have to call:<br>
`runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP', output_dir=OUTPUT_DIR,
misc = {'DYNAMIC_PARAMS':DYNAMIC_PARAMS})`
2. `TDINDP`: runs time-dependent INDP (td-INDP). To run this method, you have to call:<br>
`runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='TDINDP', output_dir=OUTPUT_DIR,
misc = {'DYNAMIC_PARAMS':DYNAMIC_PARAMS}))`

In both cases, if 'TIME_RESOURCE' is True, then the repair time for each element is considered in devising the 
restoration plans 
"""

# %%
runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP', output_dir=OUTPUT_DIR,
                    misc={'DYNAMIC_PARAMS': None, 'EXTRA_COMMODITY': EXTRA_COMMODITY, 'TIME_RESOURCE': True})
runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='INDP', output_dir=OUTPUT_DIR,
                    misc={'DYNAMIC_PARAMS': DYNAMIC_PARAMS, 'EXTRA_COMMODITY': EXTRA_COMMODITY, 'TIME_RESOURCE': True})
# runutils.run_method(FAIL_SCE_PARAM, RC, LAYERS, method='TDINDP', output_dir=OUTPUT_DIR,
#                     misc={'DYNAMIC_PARAMS': DYNAMIC_PARAMS, 'EXTRA_COMMODITY': EXTRA_COMMODITY, 'TIME_RESOURCE': True})

# %%
""" 
## Post-processing 
First, you have to set a few parameters and then call functions that read outputs
and generate the pandas DataFrames that are needed for plotting the results.

### Post-processing parameters 
1. `COST_TYPES`: type of cost that should be used in processing the outputs. Options 
are *Total*, *Under Supply*, *Over Supply*, *Node*, *Arc*, *Flow*, *Space Prep*, *Under Supply Perc*. 
2. `METHOD_NAMES`: methods whose output should be read. Options: `indp` (with static demand), `dp_indp` (with dynamic 
demand), `tdindp`, `dp_tdindp`. 

### Post-processing functions
1. `generate_combinations`: generate all the combination of outputs that should be read and
save them in `COMBS` and `OPTIMAL_COMBS` lists.
2. `read_results`: read results for combinations in `COMBS` and `OPTIMAL_COMBS` lists.
3. `read_run_time`: compute run time for different combinations.
"""

# %%
COST_TYPES = ['Total']
METHOD_NAMES = ['indp', 'dp_indp']
COMBS, OPTIMAL_COMBS = dindputils.generate_combinations(FAIL_SCE_PARAM['TYPE'], FAIL_SCE_PARAM['MAGS'],
                                                        FAIL_SCE_PARAM['SAMPLE_RANGE'], LAYERS, RC, METHOD_NAMES)
BASE_DF, objs = dindputils.read_results(COMBS, OPTIMAL_COMBS, COST_TYPES, root_result_dir=OUTPUT_DIR, deaggregate=True)

# %%
"""
### Save Variables to file
All dictionaries that are made in the postprocessing step are saved here.
"""

# %%
OBJ_LIST = [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES]
with open(OUTPUT_DIR + 'postprocess_dicts.pkl', 'wb') as f:
    pickle.dump(OBJ_LIST, f)

# %%
"""
### Plot results 
Plot functions use the dictionaries that are made in the postprocessing step to 
make output figures:
1. `plot_performance_curves`: plots costs (in `COST_TYPES`) and unmet demand vs. time.
2. `plot_separated_perform_curves`: plots costs (in `COST_TYPES`) vs. time for each layer sepearately.
"""

# %%
### Getting back the objects ###
with open(OUTPUT_DIR + 'postprocess_dicts.pkl', 'rb') as f:
    [COMBS, OPTIMAL_COMBS, BASE_DF, METHOD_NAMES] = pickle.load(f)
# plots.plot_performance_curves(BASE_DF, cost_type='Total', ci=None, deaggregate=False, plot_resilience=True)
plots.plot_separated_perform_curves(BASE_DF, x='t', y='cost', cost_type='Total', ci=95, normalize=False)
