import pickle
file_dir = 'C:/Users/ht20/Documents/GitHub/NIST_testbeds/Joplin/Damage_scenarios/'
with open(file_dir+'pop_dislocation_data.pkl', 'rb') as f:
    dynamic_params = pickle.load(f)
import seaborn as sns    
import pandas as pd
dynamic_params[3]['temp'] = dynamic_params[3]['current pop']/(dynamic_params[3]['total pop']+1e-8)
dynamic_params[3] = dynamic_params[3].apply(pd.to_numeric)
ax = sns.lineplot(x="time", y="temp", hue="node", data=dynamic_params[3])