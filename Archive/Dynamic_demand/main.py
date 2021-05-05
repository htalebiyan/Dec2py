from disloc_utils import *
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

''' census-level dislocation data '''
intersect_file = 'files/Hazard_XY_intersect_census_tract.txt'
hazard_root = 'C:/Users/ht20/Documents/Files/DataBase_tdINDP/WuData/Wu_HazardAnalysis/Stochastic GMs/'
list_scenarios_file = '../data/damagedElements_sliceQuantile_0.95.csv'
list_scenarios = pd.read_csv(list_scenarios_file)
race_file = 'files/ACSDP5Y2018.DP05_2020-10-26T004214/ACSDP5Y2018.DP05_data_with_overlays_2020-10-26T004209.csv'
income_file = 'files/ACSST5Y2018.S1901_2020-10-26T112837/ACSST5Y2018.S1901_data_with_overlays_2020-10-26T112834.csv'

# census_tract_data = census_pga_values(intersect_file, hazard_root, list_scenarios)
# census_tract_data = damage_state(census_tract_data)
# census_tract_data = race_data(census_tract_data, race_file)
# census_tract_data = median_income(census_tract_data, income_file)
# census_tract_data = insurance(census_tract_data)
# with open('input_data.pkl', 'wb') as f:
#     pickle.dump(census_tract_data, f)

# disloc_results = dislocation_models(census_tract_data)
# with open('disloc_results.pkl', 'wb') as f:
#     pickle.dump(disloc_results, f)

''' Population trend over time '''
netowrk_type = 'water'
# service_intersect_file = 'files/VT_intersect_'+netowrk_type+'.txt'
# node_column = 'FID_TelecomVT'
# service_node_file = 'files/service_to_node_'+netowrk_type+'.csv'
# time_steps = 10
return_type = 'step_function' #'step_function' 'linear'
# dynamic_demand_df = pd.DataFrame()
# for _, sce_data in list_scenarios.iterrows():
#     dd_df = dynamic_demand(disloc_results, service_intersect_file, node_column,
#                            time_steps, int(sce_data['set']), int(sce_data['sce']),
#                            return_type, service_node_file)
#     dynamic_demand_df = pd.concat([dynamic_demand_df, dd_df])
# with open('dynamic_demand_'+return_type+'_'+netowrk_type+'.pkl', 'wb') as f:
#     pickle.dump(dynamic_demand_df, f)



''' Getting back the input data '''
# root = "C:/Users/ht20/Documents/Files/dynamic_demand/"
# with open(root+'input_data.pkl', 'rb') as f:
#     input_data = pickle.load(f)
# with open(root+'disloc_results.pkl', 'rb') as f:
#     disloc_results = pickle.load(f)
# with open(root+'dynamic_demand_'+return_type+'_telecom.pkl', 'rb') as f:
#     dynamic_demand_df = pickle.load(f)

''' Plot '''
# dynamic_demand_df['temp'] = dynamic_demand_df['current pop']/dynamic_demand_df['total pop']
# plt.close('all')
# sns.set(context='notebook',style='darkgrid')
# ax = sns.lineplot(x="time", y="temp", hue="node", data=dynamic_demand_df, ci=65)
# # plt.savefig('demand.png',dpi=300,bbox_inches='tight')

time_steps = 10
dislocation_trend = disloc_results[(disloc_results['set']==4)&(disloc_results['sce']==46)]
for t in range(time_steps+1): 
    dislocation_trend[t] = 0
    if t==0:
        dislocation_trend['total pop'] = 0
    for idx, row in dislocation_trend.iterrows():
        total_pop_tract = row['disloc pop']/row['prob disloc']
        if t==0:
            dislocation_trend.loc[idx,'total pop'] = total_pop_tract
        if t < row['disloc time']:
            dislocation_trend.loc[idx,t] = total_pop_tract - row['disloc pop']
        else:
            dislocation_trend.loc[idx,t] = total_pop_tract