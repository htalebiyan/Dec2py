import pandas as pd

T = 20
results_folder = '../../results/'
power_node_file = 'C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Node_arc_info_v2/PowerNodes.csv'
power_node = pd.read_csv(power_node_file, low_memory=False)
columns = ['guid'] + ['Demand_t' + str(x) for x in range(T)]
dislocation_dict = power_node.drop(columns=[x for x in power_node.columns if x not in columns])
dislocation_dict = dislocation_dict[dislocation_dict['Demand_t0'] <= 0]
dislocation_dict.to_csv(results_folder + 'disloc_dict.csv', sep=',', header=True)
