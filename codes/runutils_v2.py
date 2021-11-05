# %%
"""
Functions that are used to run different types of analysis on the mitigation and restoration of interdependent networks
considering repair time and dynamic parameters
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import inmrp
import dislocationutils
import infrastructure_v2


# %%
def batch_run(params, fail_sce_param):
    """
    Batch run different methods for a given list of damage scenarios,
    given global parameters.

    Parameters
    ----------
    params : dict
        Dictionary of parameters needed to run the analyses.
    fail_sce_param : dict
        Dictionary of information regarding the initial damage scenarios.

    Returns
    -------
    None. Writes to file

    """
    # Set directories and paramters
    base_dir = fail_sce_param['BASE_DIR']
    damage_dir = fail_sce_param['DAMAGE_DIR']
    topology = None
    infrastructure_data = True
    if fail_sce_param['TYPE'] == 'WU':
        if fail_sce_param['FILTER_SCE'] is not None:
            list_high_dam = pd.read_csv(fail_sce_param['FILTER_SCE'])
    elif fail_sce_param['TYPE'] == 'from_csv':
        pass
    elif fail_sce_param['TYPE'] == 'synthetic':
        infrastructure_data = False
        topology = fail_sce_param['TOPO']

    print('----Running for resources: ' + str(params['V']))
    for m in fail_sce_param['L1_RANGE']:
        for i in fail_sce_param['L2_RANGE']:
            params["L2_INDEX"] = i
            params["L1_INDEX"] = m
            try:
                list_high_dam
                if len(list_high_dam.loc[(list_high_dam.set == i) & (list_high_dam.sce == m)].index) == 0:
                    continue
            except NameError:
                pass

            # Check if the results exist 
            # ..todo: move it after initializing network for synthetic nets since L is identified there
            output_dir_full = ''
            if params["ALGORITHM"] in ["INMRP"]:
                out_dir_suffix_res = inmrp.get_resource_suffix(params)
                output_dir_full = params["OUTPUT_DIR"] + '_L' + str(len(params["L"])) + '_m' + str(
                    params["L1_INDEX"]) + "_v" + out_dir_suffix_res + '/actions_' + str(i) + '_.csv'
            if os.path.exists(output_dir_full):
                print('results are already there\n')
                continue

            print('---Running Magnitude ' + str(m) + ' sample ' + str(i) + '...')
            if params['TIME_RESOURCE']:
                print('Computing repair times...')
                inmrp.time_resource_usage_curves(base_dir, damage_dir, i, params['T'])
            if params['DYNAMIC_PARAMS']:
                print("Computing dislocation data...")
                dyn_dmnd = dislocationutils.create_dynamic_param(base_dir, params)
                dislocationutils.apply_dynamic_demand(base_dir, dyn_dmnd, extra_commodity=params["EXTRA_COMMODITY"])

            print("Initializing network...")
            if infrastructure_data:
                params["N"], _, _ = inmrp.initialize_network(base_dir=base_dir, T=params['T'],
                                                             infrastructure_data=infrastructure_data,
                                                             extra_commodity=params["EXTRA_COMMODITY"])
            else:
                params["N"], params["V"], params['L'] = inmrp.initialize_network(base_dir=base_dir, l1_index=m,
                                                                                 l2_index=i, topology=topology,
                                                                                 infrastructure_data=infrastructure_data)
            if fail_sce_param['TYPE'] == 'WU':
                infrastructure_v2.add_wu_failure_scenario(params["N"], dam_dir=damage_dir, no_set=i, no_sce=m)
            elif fail_sce_param['TYPE'] == 'from_csv':
                infrastructure_v2.add_from_csv_failure_scenario(params["N"], sample=i, dam_dir=damage_dir)
            elif fail_sce_param['TYPE'] == 'synthetic':
                infrastructure_v2.add_synthetic_failure_scenario(params["N"], dam_dir=base_dir, topology=topology,
                                                                 config=m, sample=i)
            if params["ALGORITHM"] == "INMRP":
                inmrp.run_inmrp(params, save_model=True, print_cmd_line=False, co_location=False)


def run_inmrp_sample(layers):
    T = 3
    interdep_net = inmrp.initialize_sample_network(layers=layers, T=T)
    resource = {'': {t: len(layers) for t in range(T)}}
    resource[''][0] = resource[''][0] * 2
    params = {"OUTPUT_DIR": '../results/inmrp_sample_12Node_results', "V": resource, "T": T, "L": layers,
              "ALGORITHM": "INMRP", "N": interdep_net, "L1_INDEX": 0, "L2_INDEX": 0}#, "WINDOW_LENGTH": 2
    inmrp.run_inmrp(params, layers=layers, T=params["T"], suffix="", save_model=True, print_cmd_line=True)
    print('\n\nPlot restoration plan by INDP')
    inmrp.plot_indp_sample(params, T=T)
    plt.show()


def run_method(fail_sce_param, v_r, T, layers, method, output_dir='..', misc=None):
    """
    This function runs a given method for different numbers of resources,
    and a given judge, auction, and valuation type in the case of JC.

    Parameters
    ----------
    fail_sce_param : dict
        information about damage scenarios.
    v_r : float, list of float, or list of lists of floats
        number of resources,
        if this is a list of floats, each float is interpreted as a different total
        number of resources, and indp is run given the total number of resources.
        It only works when auction_type != None.
        If this is a list of lists of floats, each list is interpreted as fixed upper
        bounds on the number of resources each layer can use (same for all time step).
    T : int
        Number of time steps of the analysis.
    layers : list
        List of layers.
    method : str
        Name of the analysis method. Options are: `INMRP`
    output_dir : str, optional
        Address of the output directory. The default is '..'.
    misc: dict
        Dictionary containing additional information for analysis.
    Returns
    -------
    None.  Writes to file

    """
    for v in v_r:
        if method == 'INMRP':
            params = {"OUTPUT_DIR": output_dir + '/inmrp_results', "V": v, "T": T, 'L': layers, "ALGORITHM": method,
            "WINDOW_LENGTH": 3}
        else:
            sys.exit('Wrong method name: ' + method)

        params['EXTRA_COMMODITY'] = misc['EXTRA_COMMODITY']
        params['TIME_RESOURCE'] = misc['TIME_RESOURCE']
        params['DYNAMIC_PARAMS'] = misc['DYNAMIC_PARAMS']
        if misc['DYNAMIC_PARAMS']:
            prefix = params['OUTPUT_DIR'].split('/')[-1]
            params['OUTPUT_DIR'] = params['OUTPUT_DIR'].replace(prefix, 'dp_' + prefix)
        batch_run(params, fail_sce_param)


def run_sample_problems():
    layers = [1, 2]  # ,3]
    run_inmrp_sample(layers)
