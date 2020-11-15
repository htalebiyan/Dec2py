"""Predicts restoration scenarios"""
import numpy as np
import pandas as pd
from predictorutils import *
import STAR_utils
import plot_star
import pickle

rooy_folder = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc_only_nodes_damaged/data'
samples_all, costs_all, _ = pickle.load(open(rooy_folder+'/initial_data.pkl', "rb" ))
SAMPLE_RANGE = [50] #range(50, 551, 100) #[50, 70, 90]
MAGS = range(0, 1)
Rc = [2]
MODEL_DIR = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc_only_nodes_damaged'
OPT_DIR = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/results_only_nodes_damaged/'
# FAIL_SCE_PARAM = {"type":"WU", "sample":1, "mag":52,
#                   'Base_dir':"../data/Extended_Shelby_County/",
#                   'Damage_dir':"../data/Wu_Damage_scenarios/", 'topology':None}
FAIL_SCE_PARAM = {"type":"random", "sample":None, "mag":None, 'filtered_List':None,
                  'Base_dir':"../data/Extended_Shelby_County/",
                  'Damage_dir':"../data/random_disruption_shelby/"}
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
for res in Rc:
    for mag_num in MAGS:
        for sample_num in SAMPLE_RANGE:
            FAIL_SCE_PARAM['sample'] = sample_num
            FAIL_SCE_PARAM['mag'] = mag_num
            PARAMS['V'] = res
            ### Find actual repair sequence and costs ###
            real_rep_sequence = {}
            cost_opt = {}
            column_number = STAR_utils.find_sce_index(sample_num, res, 
                                                      MODEL_DIR+'./data/missing_scenarios.txt')
            for key, val in samples_all[res].items():
                real_rep_sequence[key] = val[:PARAMS["NUM_ITERATIONS"]+1, column_number]
            for key, val in costs_all[res].items():
                cost_opt[key] = val[:PARAMS["NUM_ITERATIONS"]+1, column_number]
            ### Predict repair sequence ###
            pred_results, pred_error, rep_prec = predict_resotration(PRED_DICT,
                                                                     FAIL_SCE_PARAM,
                                                                     PARAMS,
                                                                     real_rep_sequence)
            cost_df, pred_error, rep_prec = plot_star.plot_df(mag_num, sample_num,
                                                              len(PARAMS['L']), res,
                                                              pred_results, pred_error,
                                                              rep_prec, cost_opt)
            prediction_error = pd.concat([prediction_error, pred_error])
            repair_precentage = pd.concat([repair_precentage, rep_prec])
            cost_all_df = pd.concat([cost_all_df, cost_df])
            ### Save results ###
            folder_name = PRED_DICT['output_dir']
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            pickle.dump([cost_all_df, prediction_error, repair_precentage],
                        open(folder_name+'/results.pkl', "wb" ),
                        protocol=pickle.HIGHEST_PROTOCOL)

''' Run models in parallel '''
# import multiprocessing
# import run_parallel
# with multiprocessing.Pool(processes=len(SAMPLE_RANGE)) as p:
#     p.map(run_parallel.run_parallel, SAMPLE_RANGE)
#     p.join()

''' Plot results '''
plot_star.plot_cost(cost_all_df)
plot_star.plot_results(prediction_error, repair_precentage)
