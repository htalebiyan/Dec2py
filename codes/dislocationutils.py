"""
These function compute the change of demand values over time based on population dislocation models for Seaside and
dislocation time model for Lumberton.
"""

import pickle
import pandas as pd
import os
import numpy as np
import math


def create_dynamic_param(base_dir, params, extend=5):
    """
    This function computes the change of demand values over time based on population dislocation models.

    Parameters
    ----------
    base_dir : str
        Address of the directory where network data is stored
    params : dict
        Parameters that are needed to run the INDP optimization.\
    extend: int
        number of time steps beyond the analysis length for which the dislocation should be calculated

    Returns
    -------
     dynamic_params : dict
         Dictionary of dynamic demand value for nodes
    """
    T = params['T'] + extend
    dynamic_param_dict = params['DYNAMIC_PARAMS']
    return_type = dynamic_param_dict['RETURN']
    dp_dict_col = ['time', 'node', 'current pop', 'total pop']
    net_names = {'WATER': 1, 'GAS': 2, 'POWER': 3, 'TELECOME': 4}
    dynamic_params = {}

    output_file = dynamic_param_dict['OUT_DIR'] + dynamic_param_dict['TESTBED'] + '_pop_dislocation_demands_' + \
                  str(params['L1_INDEX']) + 'yr.pkl'
    if os.path.exists(output_file):
        print("Reading from file...")
        with open(output_file, 'rb') as f:
            dynamic_params = pickle.load(f)
        return dynamic_params
    pop_dislocation_file = dynamic_param_dict['POP_DISLOC_DATA'] + 'PopDis_results.csv'
    pop_dislocation = pd.read_csv(pop_dislocation_file, low_memory=False)
    for net in dynamic_param_dict['MAPPING'].keys():
        nn = net_names[net]
        mapping_data = pd.read_csv(dynamic_param_dict['MAPPING'][net], low_memory=False)
        dynamic_params[nn] = pd.DataFrame(columns=dp_dict_col)
        if net == 'POWER':
            node_data = pd.read_csv(base_dir + net + 'Nodes.csv')
            for idx, row in node_data.iterrows():
                guid = row['guid']
                # Find building in the service area of the node/substation
                try:
                    serv_area = mapping_data[mapping_data['substations_guid'] == guid]
                except KeyError:
                    serv_area = mapping_data[mapping_data['node_guid'] == guid]
                # compute dynamic_params
                num_dilocated = {t: 0 for t in range(T + 1)}
                total_pop_node = 0
                for _, bldg in serv_area.iterrows():
                    try:
                        pop_bldg_dict = pop_dislocation[pop_dislocation['guid'] == bldg['buildings_guid']]
                    except KeyError:
                        pop_bldg_dict = pop_dislocation[pop_dislocation['guid'] == bldg['bldg_guid']]
                    for _, hh in pop_bldg_dict.iterrows():
                        total_pop_node += hh['numprec'] if ~np.isnan(hh['numprec']) else 0
                        if hh['dislocated']:
                            # ..todo Lumebrton dislocation time model. Replace with that of Seaside when available
                            return_time = lumberton_disloc_time_mode(hh)
                            for t in range(return_time):
                                if t <= T and return_type == 'step_function':
                                    num_dilocated[t] += hh['numprec'] if ~np.isnan(hh['numprec']) else 0
                                elif t <= T and return_type == 'linear':
                                    pass  # ..todo Add linear return here
                for t in range(T + 1):
                    values = [t, row['nodenwid'], total_pop_node - num_dilocated[t], total_pop_node]
                    dynamic_params[nn] = dynamic_params[nn].append(dict(zip(dp_dict_col, values)),
                                                                   ignore_index=True)
        elif net == 'WATER':
            node_pop = {}
            arc_data = pd.read_csv(base_dir + net + 'Arcs.csv')
            for idx, row in arc_data.iterrows():
                guid = row['guid']
                # Find building in the service area of the pipe
                serv_area = mapping_data[mapping_data['edge_guid'] == guid]
                start_node = row['fromnode']
                if start_node not in node_pop.keys():
                    node_pop[start_node] = {'total_pop_node': 0, 'num_dilocated': {t: 0 for t in range(T + 1)}}
                end_node = row['tonode']
                if end_node not in node_pop.keys():
                    node_pop[end_node] = {'total_pop_node': 0, 'num_dilocated': {t: 0 for t in range(T + 1)}}
                # compute dynamic_params
                for _, bldg in serv_area.iterrows():
                    pop_bldg_dict = pop_dislocation[pop_dislocation['guid'] == bldg['bldg_guid']]
                    for _, hh in pop_bldg_dict.iterrows():
                        # half of the arc's demand is assigned to each node
                        # also, each arc is counted twice both as (u,v) and (v,u)
                        node_pop[start_node]['total_pop_node'] += hh['numprec'] / 4 if ~np.isnan(hh['numprec']) else 0
                        node_pop[end_node]['total_pop_node'] += hh['numprec'] / 4 if ~np.isnan(hh['numprec']) else 0
                        if hh['dislocated']:
                            # ..todo Lumebrton dislocation time model. Replace with that of Seaside when available
                            return_time = lumberton_disloc_time_mode(hh)
                            for t in range(return_time):
                                if t <= T and return_type == 'step_function':
                                    node_pop[start_node]['num_dilocated'][t] += hh['numprec'] / 4 if ~np.isnan(
                                        hh['numprec']) else 0
                                    node_pop[end_node]['num_dilocated'][t] += hh['numprec'] / 4 if ~np.isnan(
                                        hh['numprec']) else 0
                                elif t <= T and return_type == 'linear':
                                    pass  # ..todo Add linear return here
            for n, val in node_pop.items():
                for t in range(T + 1):
                    values = [t, n, val['total_pop_node'] - val['num_dilocated'][t], val['total_pop_node']]
                    dynamic_params[nn] = dynamic_params[nn].append(dict(zip(dp_dict_col, values)),
                                                                   ignore_index=True)
    with open(output_file, 'wb') as f:
        pickle.dump(dynamic_params, f)
    return dynamic_params


def apply_dynamic_demand(base_dir, dfs, extra_commodity=None):
    net_names = {1: 'Water', 2: 'Gas', 3: 'Power', 4: 'Telecome'}
    for idx, val in dfs.items():
        net_name = net_names[idx]
        net_data = pd.read_csv(base_dir + net_name + 'Nodes.csv')
        for idxn, row in net_data.iterrows():
            node_id = row['nodenwid']
            original_demand = float(row['Demand_t' + str(0)])
            if original_demand < 0:
                for t in val['time'].unique():
                    disloc_row = val[(val['node'] == node_id) & (val['time'] == t)]
                    current_pop = float(disloc_row['current pop'])
                    total_pop = float(disloc_row['total pop'])
                    if 'Demand_t' + str(int(t) + 1) in net_data.columns:
                        net_data.loc[idxn, 'Demand_t' + str(int(t) + 1)] = original_demand * current_pop / total_pop
                    if extra_commodity:
                        for ec in extra_commodity[idx]:
                            original_demand_ec = float(row['Demand_' + ec + '_t' + str(0)])
                            if original_demand_ec < 0 and 'Demand_' + ec + '_t' + str(int(t) + 1) in net_data.columns:
                                net_data.loc[idxn, 'Demand_' + ec + '_t' + str(
                                    int(t) + 1)] = original_demand_ec * current_pop / total_pop
        net_data.to_csv(base_dir + net_name + 'Nodes.csv', index=False)


def lumberton_disloc_time_mode(household_data):
    dt_params = {'DS0': 1.00, 'DS1': 2.33, 'DS2': 2.49, 'DS3': 3.62,
                 'white': 0.78, 'black': 0.88, 'hispanic': 0.83,
                 'income': -0.00, 'insurance': 1.06}
    race_white = 1 if household_data['race'] == 1 else 0
    race_balck = 1 if household_data['race'] == 2 else 0
    hispan = household_data['hispan'] if ~np.isnan(household_data['hispan']) else 0
    # ..todo verify that the explanatory variable correspond to columns in dt_params
    # ..todo Replace random insurance assumption
    linear_term = household_data['DS_0'] * dt_params['DS0'] + household_data['DS_1'] * dt_params['DS1'] + \
                  household_data['DS_2'] * dt_params['DS2'] + household_data['DS_3'] * dt_params['DS3'] + \
                  race_white * dt_params['white'] + race_balck * dt_params['black'] + hispan * dt_params['hispanic'] + \
                  np.random.choice([0, 1], p=[.15, .85]) * dt_params['insurance']
    # household_data['randincome']/1000*dt_params['income']+\#!!! income data
    disloc_time = np.exp(linear_term)
    return_time = math.ceil(disloc_time / 7)  # !!! assuming each time step is one week
    return return_time
