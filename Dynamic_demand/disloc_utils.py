import sys
import pandas as pd
import numpy as np
from scipy.stats import norm
import random

def census_pga_values(intersect_file, hazard_file, set_range=range(1,2)):
    print('Finding PGA values for each census tract')
    mean_pga_census_df = pd.DataFrame()
    intersect_data = pd.read_csv(intersect_file)
    census_names = intersect_data['censustractclip_NAME'].unique()
    for set_num in set_range:
        print('Reading set',set_num,'...')
        set_file = hazard_file+'PGA_Set'+str(set_num)+'.txt'
        hazard_data = pd.read_csv(set_file, delimiter='\t', header=None).dropna(axis=1)
        for ct in census_names:
            data = intersect_data[intersect_data['censustractclip_NAME']==ct]
            census_id = list(data['CensusData_Id'].unique())
            if len(census_id) != 1:
                sys.exit ('duplicate ID for the census tract')
            pga_col = 0
            num_poins = len(data.index)
            for idx, row in data.iterrows():
                pga_point = row['FID_XYHazardSce95']
                pga_col += hazard_data[pga_point].to_numpy()
            for idx, pga in enumerate(pga_col):
                temp = {'census':ct, 'census id': census_id[0], 'set':set_num,
                        'sce':idx, 'mean_pga':pga/num_poins}
                mean_pga_census_df = mean_pga_census_df.append(temp, ignore_index=True)
    return mean_pga_census_df

def damage_state(census_tract_data):
    print('Computing exceedance probabilities of damage state...')
    fragility = {'w2':{'Slight':[0.26, 0.64], 'Moderate':[0.56, 0.64],
                   'Extensive':[1.15, 0.64], 'Complete':[2.08, 0.64]}}#per HAZUS

    buidling_type = 'w2' #per HAZUS
    for idx, row in census_tract_data.iterrows():
        census_tract_data.loc[idx,'buidling type'] = buidling_type
        for key, val in fragility[buidling_type].items():
            exceed_prob = norm.cdf(np.log(row['mean_pga']/val[0])/val[1], 0, 1)
            census_tract_data.loc[idx, key+' EP'] = exceed_prob
    return census_tract_data

def race_data(census_tract_data, race_file):
    print('Finding race data...')
    race_data = pd.read_csv(race_file, delimiter=',', header=1)
    for idx, row in census_tract_data.iterrows():
        census_id = row['census id']
        race_row = race_data[race_data['id']==census_id]
        census_tract_data.loc[idx, 'total population'] = race_row['Estimate!!RACE!!Total population'].values
        census_tract_data.loc[idx, 'white'] = race_row['Estimate!!RACE!!Total population!!One race!!White'].values
        census_tract_data.loc[idx, 'black'] = race_row['Estimate!!RACE!!Total population!!One race!!Black or African American'].values
        census_tract_data.loc[idx, 'hispanic'] = race_row['Estimate!!HISPANIC OR LATINO AND RACE!!Total population!!Hispanic or Latino (of any race)'].values
    return census_tract_data

def median_income(census_tract_data, income_file):
    print('Finding median income data...')
    income_data = pd.read_csv(income_file, delimiter=',', header=1)
    for idx, row in census_tract_data.iterrows():
        census_id = row['census id']
        race_row = income_data[income_data['id']==census_id]
        census_tract_data.loc[idx, 'median income'] = race_row['Estimate!!Households!!Median income (dollars)'].values
    return census_tract_data

def insurance(census_tract_data):
    print('Finding insurance data...')
    for idx, row in census_tract_data.iterrows():
        census_tract_data.loc[idx, 'insurance'] = random.randint(0, 1)
    return census_tract_data

def dislocation_models(input_data):
    print('Computing dislocation probability and time...')
    initil_disloc_params = {'DS1':2.20, 'DS2+':4.83, 'white':-.68, 'black':.23,
                            'hispanic':-1.06, 'income':-.01, 'insurance':.78}
    disloc_time_params = {'DS1':1.00, 'DS2':2.33,'DS3':2.49, 'DS4':3.62,
                            'white':0.78, 'black':0.88, 'hispanic':0.83,
                            'income':-0.00, 'insurance':1.06}

    disloc_results = pd.DataFrame()
    for idx, row in input_data.iterrows():
        temp = {'census':row['census'], 'census id': row['census id'], 'set':row['set'],
                'sce':row['sce']}
        temp['DS1'] = row['Slight EP'] - row['Moderate EP']
        temp['DS2'] = row['Moderate EP'] - row['Extensive EP']
        temp['DS3'] = row['Extensive EP'] - row['Complete EP']
        temp['DS4'] = row['Complete EP']
        if row['total population']>0:
            temp['white'] = row['white']/row['total population']
            temp['black'] = row['black']/row['total population']
            temp['hispanic'] = row['hispanic']/row['total population']
        else:
            temp['white'] = 0
            temp['black'] = 0
            temp['hispanic'] = 0
        if row['median income'] != '-':
            temp['income'] = float(row['median income'])
        else:
            temp['income'] = 0
        temp['insurance'] = row['insurance']
        
        linear_term = temp['DS1']*initil_disloc_params['DS1']+\
            (temp['DS2']+temp['DS3']+temp['DS4'])*initil_disloc_params['DS2+']+\
            temp['white']*initil_disloc_params['white']+\
            temp['black']*initil_disloc_params['black']+\
            temp['hispanic']*initil_disloc_params['hispanic']+\
            temp['income']/1000*initil_disloc_params['income']+\
            temp['insurance']*initil_disloc_params['insurance']
        temp['prob disloc'] = 1/(1+np.exp(-1.0*linear_term))
        temp['disloc pop'] = temp['prob disloc']*row['total population']
    
        linear_term = temp['DS1']*disloc_time_params['DS1']+\
            temp['DS2']*disloc_time_params['DS2']+\
            temp['DS3']*disloc_time_params['DS3']+\
            temp['DS4']*disloc_time_params['DS4']+\
            temp['white']*disloc_time_params['white']+\
            temp['black']*disloc_time_params['black']+\
            temp['hispanic']*disloc_time_params['hispanic']+\
            temp['income']/1000*disloc_time_params['income']+\
            temp['insurance']*disloc_time_params['insurance']
        temp['disloc time'] = np.exp(linear_term)
        disloc_results = disloc_results.append(temp, ignore_index=True)
    return disloc_results

def dynamic_demand(disloc_results, service_intersect_file, node_column, time_steps,
                   set_num, sce_num, return_type):
    print('Demand calculation for sce', sce_num, ', set', set_num)
    service_intersect_data = pd.read_csv(service_intersect_file)
    nodes = service_intersect_data[node_column].unique()
    dynamic_demand_df = pd.DataFrame()
    for t in time_steps:
        temp_df = pd.DataFrame()
        for n in nodes:
            data = service_intersect_data[service_intersect_data[node_column]==n]
            temp = {'time':t, 'node':n, 'set': set_num, 'sce':sce_num, 'total pop':0, 'current pop':0}
            for idx, row in data.iterrows():
                disloc_data = disloc_results[(disloc_results['census id']==row['CensusData_Id'])&\
                                             (disloc_results['sce']==sce_num)&\
                                             (disloc_results['set']==set_num)]
                if disloc_data.empty:
                    pass
                    # print('No census tract:', row['CensusData_Id'])
                else:
                    total_pop_tract = disloc_data['disloc pop'].values[0]/disloc_data['prob disloc'].values[0]
                    overlap_perc = row['Shape_Area']/row['censustractclip_Area']
                    temp['total pop'] += total_pop_tract*overlap_perc
                    if return_type == 'step_function':
                        if t < disloc_data['disloc time'].values[0]:
                            temp['current pop'] += (total_pop_tract - disloc_data['disloc pop'].values[0])*overlap_perc
                        else:
                            temp['current pop'] += total_pop_tract*overlap_perc
                    elif return_type == 'linear':
                        if t < disloc_data['disloc time'].values[0]:
                            initial_disloc_pop = disloc_data['disloc pop'].values[0]
                            temp['current pop'] += (initial_disloc_pop +\
                                (total_pop_tract - initial_disloc_pop)/disloc_data['disloc time'].values[0]*t)*overlap_perc
                        else:
                            temp['current pop'] += total_pop_tract*overlap_perc
            temp_df = temp_df.append(temp, ignore_index=True)
        county_pop = sum(temp_df['total pop'])
        county_cur_pop = sum(temp_df['current pop'])
        temp_df['total pop perc'] = temp_df['total pop']/county_pop
        temp_df['current pop perc'] = temp_df['current pop']/county_cur_pop
        dynamic_demand_df = pd.concat([dynamic_demand_df, temp_df])
    return dynamic_demand_df