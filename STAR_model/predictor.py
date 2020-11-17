"""Predicts restoration scenarios"""
import numpy as np
import pandas as pd
from predictorutils import *
import STAR_utils
import plot_star
import itertools
import pickle

SAMPLE_RANGE = range(50) #range(1, 50) #range(50, 551, 100) #[50, 70, 90]
MAGS = range(96)
Rc = [3, 6, 8, 12]
MODEL_DIR = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc_only_nodes_damaged'
OPT_DIR = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/results_only_nodes_damaged/'
FAIL_SCE_PARAM = {"type":"WU", "sample_range":None, "mags_range":None,
                  'Base_dir':"../data/Extended_Shelby_County/",
                  'Damage_dir':"../data/Wu_Damage_scenarios/", 'topology':None,
                  'filtered_List':None}
# FAIL_SCE_PARAM = {"type":"random", "sample_range":None, "mags":None, 'filtered_List':None,
#                   'Base_dir':"../data/Extended_Shelby_County/",
#                   'Damage_dir':"../data/random_disruption_shelby/"}
PRED_DICT = {'num_pred':5, 'model_dir':MODEL_DIR+'/traces',
             'param_folder':MODEL_DIR+'/parameters', 'output_dir':'./results'}
PARAMS = {"NUM_ITERATIONS":10, "V":None, "ALGORITHM":"INDP", 'L':[1, 2, 3, 4]}

''' Run models '''
prediction_error = pd.DataFrame(columns=['magnitude', 'sample', 'Rc', 'name',
                                          'pred sample', 'pred rep time',
                                          'real rep time', 'prediction error'])
repair_precentage = pd.DataFrame(columns=['magnitude', 'sample', 'Rc', 't',
                                          'pred sample', 'pred rep prec', 'real rep prec'])
cost_all_df = pd.DataFrame()
for res, mag_num, sample_num in itertools.product(Rc, MAGS, SAMPLE_RANGE):
    FAIL_SCE_PARAM['sample_range'] = sample_num
    FAIL_SCE_PARAM['mags'] = mag_num
    PARAMS['V'] = res
    
    ### Find actual repair sequence and costs ###
    if FAIL_SCE_PARAM['type'] == 'random':
        miss_sce_dir = MODEL_DIR+'/data/missing_scenarios.txt'
        real_result_dir = MODEL_DIR+'/data/'
        samples_all, costs_all, _ = pickle.load(open(real_result_dir+'/initial_data.pkl', "rb" ))
        data_specific = {'samples_all':samples_all, 'costs_all':costs_all, 'miss_sce_dir':miss_sce_dir}
    if FAIL_SCE_PARAM['type'] == 'WU':
        PARAMS['OUTPUT_DIR'] = 'C:/Users/ht20/Documents//Files/STAR_models/Shelby_final_all_Rc_only_nodes_damaged/results_shelby/actual_results/indp_results'
        data_specific = {}
    
    real_rep_seq, real_cost = actual_repair_data(FAIL_SCE_PARAM, PARAMS, data_specific)
    
    if not(real_rep_seq and real_cost):
        continue
    
    ### Predict repair sequence ###
    pred_results, pred_error, rep_prec = predict_resotration(PRED_DICT, FAIL_SCE_PARAM,
                                                              PARAMS, real_rep_seq)
    cost_df, pred_error, rep_prec = plot_star.plot_df(mag_num, sample_num, len(PARAMS['L']),
                                                      res, pred_results, pred_error,
                                                      rep_prec, real_cost)
    ### Save results ###
    prediction_error = pd.concat([prediction_error, pred_error])
    repair_precentage = pd.concat([repair_precentage, rep_prec])
    cost_all_df = pd.concat([cost_all_df, cost_df])
    
    folder_name = PRED_DICT['output_dir']
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    pickle.dump([cost_all_df, prediction_error, repair_precentage],
                open(folder_name+'/results.pkl', "wb" ),
                protocol=pickle.HIGHEST_PROTOCOL)

''' Plot results '''
plot_star.plot_cost(cost_all_df)
plot_star.plot_results(prediction_error, repair_precentage)
