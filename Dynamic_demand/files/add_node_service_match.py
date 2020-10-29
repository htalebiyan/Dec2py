import pickle
import pandas as pd

return_type = 'step_function'
net_type = 'telecom'
filename = 'C:/Users/ht20/Documents/Files/dynamic_demand/dynamic_demand_'+return_type+'_'+net_type+'.pkl'
with open(filename, 'rb') as f:
    dynamic_demand_df = pickle.load(f)
aaaa = dynamic_demand_df[(dynamic_demand_df['sce']==34)&(dynamic_demand_df['time']==0)]

# service_node_file = 'service_to_node_'+net_type+'.csv'
# service_node_data = pd.read_csv(service_node_file)
# # dynamic_demand_df = dynamic_demand_df.rename(columns={'node': 'service area'})
# for idx, row in dynamic_demand_df.iterrows():
#     dynamic_demand_df.loc[idx,'node'] = service_node_data.loc[service_node_data['Service Area']==int(row['service area']),
#                                         'Node'].iloc[0]
    
# with open('dynamic_demand_'+return_type+'_'+net_type+'.pkl', 'wb') as f:
#     pickle.dump(dynamic_demand_df, f)

