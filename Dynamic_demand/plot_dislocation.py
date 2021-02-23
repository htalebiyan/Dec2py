import pickle
import seaborn as sns    
import pandas as pd
import matplotlib.pyplot as plt

file_dir = 'C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Node_arc_info/'
with open(file_dir+'seaside_pop_dislocation_demands.pkl', 'rb') as f:
    dynamic_params = pickle.load(f)

demand_data = dynamic_params[1]
demand_data['temp'] = demand_data['current pop']/(demand_data['total pop']+1e-8)
demand_data = demand_data.apply(pd.to_numeric)
ax = sns.lineplot(x="time", y="temp", hue="node", data=demand_data)
