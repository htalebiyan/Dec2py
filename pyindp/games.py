"""Predicts restoration scenarios"""
import copy
import os
import sys
import time
import numpy as np
import pandas as pd
import gambit
import indp
import gameclasses

layers=[1,2,3]
interdep_net= indp.initialize_sample_network(layers=layers)
params={"NUM_ITERATIONS":5, "OUTPUT_DIR":'../results/ng_sample_12Node_results',
        "V":6, "T":1, "L":layers, "WINDOW_LENGTH":1, "ALGORITHM":"NORMALGAME",
        'EQUIBALG':'gnm_solve'}
params["N"]=interdep_net
params["JUDGMENT_TYPE"]="OPTIMISTIC"
params["MAGNITUDE"] = 0
params["SIM_NUMBER"] = 0
params["RES_ALLOC_TYPE"]= 'UNIFORM'
# params["VALUATION_TYPE"]= 'DTC'

obj = gameclasses.InfrastructureGame(params)
obj.run_game(compute_optimal=True, plot=True)

indp.plot_indp_sample(params, folderSuffix="_"+params["RES_ALLOC_TYPE"])