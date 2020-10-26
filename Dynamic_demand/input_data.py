import sys
import pandas as pd
import numpy as np
import pickle
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
    print('Computing damage state exceddance probabilities...')
    fragility = {'w2':{'Slight':[0.26, 0.64], 'Moderate':[0.56, 0.64],
                   'Extensive':[1.15, 0.64], 'Complete':[2.08, 0.64]}}#per HAZUS

    buidling_type = 'w2' #per HAZUS
    for idx, row in census_tract_data.iterrows():
        census_tract_data.loc[idx,'buidling type'] = buidling_type
        for key, val in fragility[buidling_type].items():
            exceed_prob = norm.cdf(np.log(row['mean_pga']/val[0])/val[1], 0, 1)
            census_tract_data.loc[idx, key+' EP'] = exceed_prob
    return census_tract_data

def race_data(census_tract_data):
    print('Finding race data...')
    race_file = 'ACSDP5Y2018.DP05_2020-10-26T004214/ACSDP5Y2018.DP05_data_with_overlays_2020-10-26T004209.csv'
    race_data = pd.read_csv(race_file, delimiter=',', header=1)
    for idx, row in census_tract_data.iterrows():
        census_id = row['census id']
        race_row = race_data[race_data['id']==census_id]
        census_tract_data.loc[idx, 'total population'] = race_row['Estimate!!RACE!!Total population'].values
        census_tract_data.loc[idx, 'white'] = race_row['Estimate!!RACE!!Total population!!One race!!White'].values
        census_tract_data.loc[idx, 'black'] = race_row['Estimate!!RACE!!Total population!!One race!!Black or African American'].values
        census_tract_data.loc[idx, 'hispanic'] = race_row['Estimate!!HISPANIC OR LATINO AND RACE!!Total population!!Hispanic or Latino (of any race)'].values
    return census_tract_data

def median_income(census_tract_data):
    print('Finding median income data...')
    income_file = 'ACSST5Y2018.S1901_2020-10-26T112837/ACSST5Y2018.S1901_data_with_overlays_2020-10-26T112834.csv'
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

intersect_file = 'Hazard_XY_intersect_census_tract.txt'
hazard_file = 'C:/Users/ht20/Documents/Files/DataBase_tdINDP/WuData/Wu_HazardAnalysis/Stochastic GMs/'
census_tract_data = census_pga_values(intersect_file, hazard_file, set_range=range(1,2))
census_tract_data = damage_state(census_tract_data)
census_tract_data = race_data(census_tract_data)
census_tract_data = median_income(census_tract_data)
census_tract_data = insurance(census_tract_data)

with open('input_data.pkl', 'wb') as f:
    pickle.dump(census_tract_data, f)



