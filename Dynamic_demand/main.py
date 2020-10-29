from disloc_utils import *
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

intersect_file = 'files/Hazard_XY_intersect_census_tract.txt'
hazard_file = 'C:/Users/ht20/Documents/Files/DataBase_tdINDP/WuData/Wu_HazardAnalysis/Stochastic GMs/'
race_file = 'files/ACSDP5Y2018.DP05_2020-10-26T004214/ACSDP5Y2018.DP05_data_with_overlays_2020-10-26T004209.csv'
income_file = 'files/ACSST5Y2018.S1901_2020-10-26T112837/ACSST5Y2018.S1901_data_with_overlays_2020-10-26T112834.csv'
netowrk_type = 'telecom'
service_intersect_file = 'files/VT_intersect_'+netowrk_type+'.txt'
service_node_file = 'files/service_to_node_'+netowrk_type+'.csv'
node_column = 'FID_TelecomVT'
time_steps = range(3)
return_type = 'linear' #'step_function' 'linear'

# census_tract_data = census_pga_values(intersect_file, hazard_file, set_range=range(1,2))
# census_tract_data = damage_state(census_tract_data)
# census_tract_data = race_data(census_tract_data, race_file)
# census_tract_data = median_income(census_tract_data, income_file)
# census_tract_data = insurance(census_tract_data)
# with open('input_data.pkl', 'wb') as f:
#     pickle.dump(census_tract_data, f)

# disloc_results = dislocation_models(census_tract_data)
# with open('disloc_results.pkl', 'wb') as f:
#     pickle.dump(disloc_results, f)

dynamic_demand_df = pd.DataFrame()
for set_num in range(1,2):
    for sce_num in range(34, 35):
        dd_df = dynamic_demand(disloc_results, service_intersect_file, node_column,
                                            time_steps, set_num, sce_num, return_type,
                                            service_node_file)
        dynamic_demand_df = pd.concat([dynamic_demand_df, dd_df])
with open('dynamic_demand_'+return_type+'.pkl', 'wb') as f:
    pickle.dump(dynamic_demand_df, f)



# # Getting back the input data ###
# root = "C:/Users/ht20/Documents/Files/dynamic_demand/"
# with open(root+'input_data.pkl', 'rb') as f:
#     input_data = pickle.load(f)
# with open(root+'disloc_results.pkl', 'rb') as f:
#     disloc_results = pickle.load(f)
# with open(root+'dynamic_demand_'+return_type+'.pkl', 'rb') as f:
#     dynamic_demand_df = pickle.load(f)

sns.set(context='notebook',style='darkgrid')
ax = sns.lineplot(x="time", y="current pop perc", hue="node", data=dynamic_demand_df)
plt.savefig('demand.png',dpi=300,bbox_inches='tight')