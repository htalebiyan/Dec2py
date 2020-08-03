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
import gameplots

# def run_indp_sample():
interdep_net= indp.initialize_sample_network()
params={"NUM_ITERATIONS":7, "OUTPUT_DIR":'../results/ng_sample_12Node_results',
        "V":2, "T":1, "L":[1,2], "WINDOW_LENGTH":1, "ALGORITHM":"NORMALGAME"}
params["N"]=interdep_net
params["JUDGMENT_TYPE"]="OPTIMISTIC"
params["MAGNITUDE"]=0
params["SIM_NUMBER"]=0
params["RES_ALLOC_TYPE"]= 'UNIFORM'

# obj = gameclasses.infrastructureGame(params)

game = gameclasses.NormalGame(params['L'], params['N'], [2,2])
game.compute_payoffs()
game.build_game(write_to_file_dir='./games', file_name='test')
game.solve_game(method='enumpure', print_to_cmd=False)
gameplots.plot_ne_sol_2player(game)
 
   



# g = gambit.Game.new_table([2,2])
# m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=gambit.Rational)
# m2 = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]], dtype=gambit.Rational)
# m3 = np.array([[[21, 22], [23, 24]], [[25, 26], [27, 28]]], dtype=gambit.Rational)
# g = gambit.Game.from_arrays(m, m2,m3)#np.transpose(m))

# g.title = "A prisoner's dilemma game"
# # g.players[0].label = "Alphonse"
# # g.players[1].label = "Gaston"
# # g.players[0].strategies[0].label = "Cooperate"
# # g.players[0].strategies[1].label = "Defect"
# # g.players[1].strategies[0].label = "Cooperate"
# # g.players[1].strategies[1].label = "Defect"
# # g.players[1].strategies[2].label = "extra"
# print("outcomes")
# for x in g.outcomes:
#         print([float(y) for y in x]) 
# # for profile in g.contingencies:
# #     print(profile, g[profile][0], g[profile][1], g[profile][2])

# list(g.players[0].strategies)[0]

# p = g.mixed_strategy_profile()
# list(p)
# p.payoff(g.players[0])
# p.strategy_value(g.players[0].strategies[0])

# a = gambit.nash.enumpure_solve(g)
# # a = gambit.nash.lcp_solve(g, rational=True, use_strategic=True)

# print("Solution")
# for x in a:
#     print([float(y) for y in x])
#     print([float(y) for y in x.payoff()])

# # g = gambit.Game.read_game("games/2x2x2.nfg")
# # for x in g.outcomes:
# #         print([y for y in x])

# # # list(g.players[0].strategies)[0]

# # # p = g.mixed_strategy_profile()
# # # list(p)
# # # p.payoff(g.players[0])
# # # p.strategy_value(g.players[0].strategies[0])

# # a = gambit.nash.enumpure_solve(g)
# # for x in a:
# #     print([float(y) for y in x])
# #     print([float(y) for y in x.payoff()])

# # g = gambit.Game.read_game("games/e02.nfg")
# # solver = gambit.nash.ExternalEnumPureSolver()
# # solver.solve(g)

# # solver = gambit.nash.ExternalEnumMixedSolver()
# # solver.solve(g)

# # g = gambit.Game.read_game("games/e02.efg")
# # p = g.mixed_behavior_profile()
# # list(p)