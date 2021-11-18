import pickle
import pandas as pd
import math
import re
import numpy as np

sample = 0
net_names = {1: 'Water', 3: 'Power'}
initial_damage_folder = 'C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Damage_scenarios' \
                        '/cumulative_1000yr_initial_damage/'
results_folder = '../../results/dp_inmrp_results_L2_m1000_vb5587439t1764_uneven_importance/'

repair_data = pd.read_csv(results_folder + 'actions_' + str(sample) + '_.csv', low_memory=False)
repair_dict = pd.DataFrame(columns=['name', 'element', 'net', 'guid', 'damaged', 'protected', 'repair_time'])

for element in ['node', 'link']:
    initial_damage_data = pd.read_csv(initial_damage_folder + 'initial_' + element + '.csv', low_memory=False, header=0)
    for _, val in initial_damage_data.iterrows():
        temp_dict = {'name': val['name'], 'element': element, 'guid': val['guid'], 'protected': False,
                     'damaged': True if val[str(sample)] == 0 else False,
                     'repair_time': -1 if val[str(sample)] == 0 else -2}
        try:
            temp_dict['net'] = net_names[int(val['name'][-2])]
        except ValueError:
            temp_dict['net'] = net_names[int(val['name'][-3])]
        repair_dict = repair_dict.append(temp_dict, ignore_index=True)
for _, val in repair_data.iterrows():
    repair_time = val['t']
    net_name = net_names[int(val['action'][-1])]
    if '/' not in val['action']:
        action = val['action'].split('.')
        name = '(' + action[0] + ',' + action[1] + ')'
    else:
        action = val['action'].split('.')
        action_s = action[1].split('/')
        name = '((' + action[0] + ',' + action_s[0] + '),(' + action_s[1] + ',' + action[2] + ')'
        name_dup = '((' + action_s[1] + ',' + action[2] + '),(' + action[0] + ',' + action_s[0] + '))'
    if sum(repair_dict.name == name) != 0:
        repair_dict.loc[repair_dict.name == name, 'repair_time'] = int(repair_time)
        if repair_time == 0:
            repair_dict.loc[repair_dict.name == name, 'protected'] = True
    else:
        repair_dict.loc[repair_dict.name == name_dup, 'repair_time'] = int(repair_time)
        if repair_time == 0:
            repair_dict.loc[repair_dict.name == name_dup, 'protected'] = True
repair_dict.to_csv(results_folder+'repair_dict.csv', sep=',', header=True)
