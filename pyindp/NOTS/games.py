"""Predicts restoration scenarios"""
import copy
import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
import gambit
import indp
import gameclasses

# layers=[1,2,3]
# interdep_net= indp.initialize_sample_network(layers=layers)
# params={"NUM_ITERATIONS":5, "OUTPUT_DIR":'../results/ng_sample_12Node_results',
#         "V":6, "T":1, "L":layers, "WINDOW_LENGTH":1, "ALGORITHM":"NORMALGAME",
#         'EQUIBALG':'gnm_solve'}
# params["N"]=interdep_net
# params["JUDGMENT_TYPE"]="OPTIMISTIC"
# params["MAGNITUDE"] = 0
# params["SIM_NUMBER"] = 0
# params["RES_ALLOC_TYPE"]= 'UNIFORM'
# # params["VALUATION_TYPE"]= 'DTC'

# obj = gameclasses.InfrastructureGame(params)
# obj.run_game(compute_optimal=True, plot=True)

# indp.plot_indp_sample(params, folderSuffix="_"+params["RES_ALLOC_TYPE"])


BASE_DIR = "../data/Extended_Shelby_County/"
DAMAGE_DIR = "../data/Wu_Damage_scenarios/" 
OUTPUT_DIR = '../results/'

layers=[1,3]
FAIL_SCE_PARAM = {'TYPE':"WU", 'SAMPLE_RANGE':range(0, 50), 'MAGS':range(0, 95),
                  'FILTER_SCE':None, 'BASE_DIR':BASE_DIR, 'DAMAGE_DIR':DAMAGE_DIR}
params = {"NUM_ITERATIONS":1, "OUTPUT_DIR":OUTPUT_DIR+'/ng_results',
          "V":10, "T":1, 'L':layers, "ALGORITHM":"NORMALGAME", 'EQUIBALG':'gnm_solve',
          "RES_ALLOC_TYPE":'UNIFORM'}

params["N"], _, _ = indp.initialize_network(BASE_DIR=BASE_DIR,
            external_interdependency_dir=None, sim_number=0, magnitude=6,
            sample=0, v=params["V"], shelby_data=True)
params["SIM_NUMBER"] = 1
params["MAGNITUDE"] = 13
indp.add_Wu_failure_scenario(params["N"], DAM_DIR=DAMAGE_DIR, noSet=1, noSce=13)

obj = gameclasses.InfrastructureGame(params)
obj.run_game(compute_optimal=True, plot=True)

# # Getting back the objects ###
# address = '../results/ng_results_L2_m13_v10_UNIFORM/objs_1.pkl'
# with open(address, 'rb') as f:
#     objs_read = pickle.load(f)