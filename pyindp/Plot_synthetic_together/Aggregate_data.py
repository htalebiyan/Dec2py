import numpy as np
import pandas as pd
import pickle

lambda_df_old = pd.read_pickle('Archive/temp_synthetic_v3_1')
lambda_df_old = lambda_df_old.replace({'Uniform': 'UNIFORM', 'judgeCall_OPTIMISTIC': 'jc',
                                       '': 'nan'})
lambda_df_old = lambda_df_old.assign(judgment_type='OPTIMISTIC')

alloc_df_old = pd.read_pickle('Archive/temp_res')
alloc_df_old = alloc_df_old.replace({'Uniform': 'UNIFORM', 'judgeCall_OPTIMISTIC': 'jc',
                                     '': 'nan'})
alloc_df_old = alloc_df_old.assign(judgment_type='OPTIMISTIC')
alloc_df_old = alloc_df_old.rename({'distance_to_optimal': 'gap',
                                    'norm_distance_to_optimal': 'norm_gap'}, axis='columns')

time_df_old = pd.read_pickle('Archive/temp_run')
time_df_old = time_df_old.replace({'Uniform': 'UNIFORM', 'judgeCall_OPTIMISTIC': 'jc',
                                   '': 'nan'})
time_df_old = time_df_old.assign(judgment_type='OPTIMISTIC')


#Read new results and config data
Topo = ['Random', 'ScaleFree', 'Grid']
config_info = pd.DataFrame()
lambda_df_new = pd.DataFrame()
alloc_df_new = pd.DataFrame()
time_df_new = pd.DataFrame()
for to in Topo:
    config_dir = "C:\\Users\\ht20\\Documents\\Files\\Generated_Network_Dataset_v3.1\\"
    df_config = pd.read_csv(config_dir+to+'Networks\List_of_Configurations.txt',
                            header=0, sep="\t")
    df_config = df_config.assign(topology=to)
    config_info = pd.concat([config_info,df_config])
    
    results_dir = "C:\\Users\\ht20\\Documents\\Files\\Auction_synthetic_networks_v3.1\\"
    rslt = pd.read_pickle(results_dir+to+'\postprocess_dicts.pkl')
    rslt[4] = rslt[4].assign(topology=to)
    lambda_df_new = pd.concat([lambda_df_new, rslt[4]])
    rslt[6] = rslt[6].assign(topology=to)
    alloc_df_new = pd.concat([alloc_df_new, rslt[6]])
    rslt[7] = rslt[7].assign(topology=to)
    time_df_new = pd.concat([time_df_new, rslt[7]])

LAMBDA_DF = pd.concat([lambda_df_old, lambda_df_new])
LAMBDA_DF=pd.merge(LAMBDA_DF, config_info,
              left_on=['Magnitude','topology'],
              right_on=['Config Number','topology'])
ALLOC_GAP_DF = pd.concat([alloc_df_old, alloc_df_new])
ALLOC_GAP_DF=pd.merge(ALLOC_GAP_DF, config_info,
              left_on=['Magnitude','topology'],
              right_on=['Config Number','topology']) 
RUN_TIME_DF = pd.concat([time_df_old, time_df_new])
RUN_TIME_DF=pd.merge(RUN_TIME_DF, config_info,
              left_on=['Magnitude','topology'],
              right_on=['Config Number','topology']) 

with open('postprocess_dicts_all_topo.pkl', 'wb') as f:
    pickle.dump([LAMBDA_DF, ALLOC_GAP_DF, RUN_TIME_DF], f)

