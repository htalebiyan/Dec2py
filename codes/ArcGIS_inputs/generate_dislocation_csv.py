import pickle
import pandas as pd
import math
import numpy as np

def lumberton_disloc_time_mode(household_data):
    dt_params = {'DS1':1.00, 'DS2':2.33,'DS3':2.49, 'DS4':3.62,
                 'white':0.78, 'black':0.88, 'hispanic':0.83,
                 'income':-0.00, 'insurance':1.06}
    race_white = 1 if household_data['race']==1 else 0
    race_balck = 1 if household_data['race']==2 else 0
    hispan = household_data['hispan'] if ~np.isnan(household_data['hispan']) else 0
    #!!! verfy that the explanatory variable correspond to columns in dt_params
    linear_term = household_data['insignific']*dt_params['DS1']+\
        household_data['moderate']*dt_params['DS2']+\
        household_data['heavy']*dt_params['DS3']+\
        household_data['complete']*dt_params['DS4']+\
        race_white*dt_params['white']+\
        race_balck*dt_params['black']+\
        hispan*dt_params['hispanic']+\
        np.random.choice([0,1], p=[.15, .85])*dt_params['insurance'] #!!! insurance data
        # household_data['randincome']/1000*dt_params['income']+\#!!! income data
    disloc_time = np.exp(linear_term)
    return_time = math.ceil(disloc_time/7) #!!! assuming each time step is one week
    return return_time

return_period = 1000
net_name = 'POWER'
T = 20
return_type = 'step_function'

pop_dislocation_file = '../housingunit_eq_'+ str(return_period)+'yr_popdis_result.csv'
pop_dislocation = pd.read_csv(pop_dislocation_file, low_memory=False)

mappinf_file = '../../power/bldgs2elec_Seaside.csv'
mapping_data = pd.read_csv(mappinf_file, low_memory=False)

# compute dynamic_params
dp_dict_col = ['guid', 'total population bldg']+[x for x in range(T+1)]
dynamic_params = pd.DataFrame(columns=dp_dict_col)
for _, bldg in mapping_data.iterrows():
    total_pop_bldg = 0
    num_dilocated = {t:0 for t in range(T+1)}
    try:
        guid = bldg['buildings_guid']
    except KeyError:
        guid = bldg['bldg_guid']
    pop_bldg_dict = pop_dislocation[pop_dislocation['guid']==guid] 
    for _, hh in pop_bldg_dict.iterrows():
        total_pop_bldg += hh['numprec'] if ~np.isnan(hh['numprec']) else 0
        if hh['dislocated']:
            #!!! Lumebrton dislocation time paramters
            return_time = lumberton_disloc_time_mode(hh)
            for t in range(return_time):
                if t <= T and return_type == 'step_function':
                    num_dilocated[t] += hh['numprec'] if ~np.isnan(hh['numprec']) else 0
                elif t <= T and return_type == 'linear':
                    pass #!!! Add linear return here
    if total_pop_bldg != 0:
        values = [guid, total_pop_bldg]+\
            [(total_pop_bldg-num_dilocated[t])/total_pop_bldg for t in range(T+1)]
    else:
        values = [guid, total_pop_bldg]+[0 for t in range(T+1)]        
    dynamic_params = dynamic_params.append(dict(zip(dp_dict_col, values)), ignore_index=True)

dynamic_params.to_csv('bldg_dislocation_'+str(return_period)+'.csv', sep=',', header = False)
