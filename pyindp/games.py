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

interdep_net= indp.initialize_sample_network()
params={"NUM_ITERATIONS":5, "OUTPUT_DIR":'../results/ng_sample_12Node_results',
        "V":9, "T":1, "L":[1,2], "WINDOW_LENGTH":1, "ALGORITHM":"NORMALGAME",
        'EQUIBALG':'enummixed_solve'}
params["N"]=interdep_net
params["JUDGMENT_TYPE"]="OPTIMISTIC"
params["MAGNITUDE"] = 0
params["SIM_NUMBER"] = 0
params["SIM_NUMBER"]=0
params["RES_ALLOC_TYPE"]= 'UNIFORM'
params["VALUATION_TYPE"]= 'DTC'

obj = gameclasses.InfrastructureGame(params)
obj.run_game(compute_optimal=True, plot=True)

   



# g = gambit.Game.read_game("e02.nfg")
# p = g.mixed_strategy_profile()
# # g.write(format='native')[2:-1].replace('\\n', '\n')
# # print(p)
# # print(p.payoff())
# # print(p.strategy_value(g.players[1].strategies[1]))
# gambit.nash.enumpure_solve(g)
# gambit.nash.enummixed_solve(g)
# gambit.nash.lcp_solve(g)
# # gambit.nash.lp_solve(g)
# gambit.nash.simpdiv_solve(g)
# gambit.nash.ipa_solve(g)
# gambit.nash.gnm_solve(g)