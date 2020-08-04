'''
This module contain the classes used for game theoritical analysis of interdpendent
network restoration
'''
import sys
import os
import time
import copy
import itertools
import fractions
import pickle
import random
# import numpy as np
# import pandas as pd
import gambit
import dindpclasses
import indpalt
import indp
import indputils
import gameplots

class NormalGame:
    '''
    This class models a normal restoration game for a given time step
    '''
    def __init__(self, L, net, v_r):
        #: Basic attributes
        self.players = L
        self.net = net
        self.v_r = v_r
        self.actions = self.find_actions()
        self.payoffs = {}
        self.payoff_time = {}
        self.normgame = []
        self.solution = []
        self.chosen_equilibrium = {}
        self.optimal_solution = {}
        self.temp_storage = {}

    def find_actions(self):
        '''
        This function finds all possible restoration actions for each player

        Returns
        -------
        actions : dict
            DESCRIPTION.

        '''
        actions = {}
        for l in self.players:
            damaged_nodes = [n for n, d in self.net.G.nodes(data=True) if\
                        d['data']['inf_data'].functionality == 0.0 and n[1] == l]
            damaged_arcs = [(u, v) for u, v, a in self.net.G.edges(data=True) if\
                        a['data']['inf_data'].functionality == 0.0 and u[1] == l and v[1] == l]
            basic_actions = damaged_nodes + damaged_arcs
            actions[l] = []
            for v in range(self.v_r[l]):
                actions[l].extend(list(itertools.combinations(basic_actions, v+1)))
        return actions

    def compute_payoffs(self):
        '''
        This function finds all possible combinations of actions and their corresponding
        payoffs considering resource limitations

        Returns
        -------
        None.

        '''
        actions_super_set = []
        for l in self.players:
            actions_super_set.append([])
            actions_super_set[-1].extend(self.actions[l])
        action_comb = list(itertools.product(*actions_super_set))
        # compute payoffs for each possible combinations of actions
        for idx, ac in enumerate(action_comb):
            self.payoffs[idx] = {}
            flow_results = self.flow_problem(ac)
            for idxl, l in enumerate(self.players):
                # Minus sign because we want to minimize the cost
                payoff_layer = -flow_results[1].results_layer[l][0]['costs']['Total']
                self.payoffs[idx][l] = [ac[idxl], payoff_layer]
                self.temp_storage[idx] = flow_results
                self.payoff_time[idx] = flow_results[1].results[0]['run_time']
                # self.payoffs['full solution']

    def flow_problem(self, action):
        '''
        solves a flow problem for a given combination of actions

        Parameters
        ----------
        action : tuple
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        for idxl, l in enumerate(self.players):
            for act in action[idxl]:
                self.net.G.nodes[act]['data']['inf_data'].repaired=1.0
                self.net.G.nodes[act]['data']['inf_data'].functionality=1.0
        flow_results = indp.indp(self.net, v_r=0, layers=self.players,
                                 controlled_layers=self.players,
                                 print_cmd=True, time_limit=None)
        for idxl, l in enumerate(self.players):
            for act in action[idxl]:
                self.net.G.nodes[act]['data']['inf_data'].repaired=0.0
                self.net.G.nodes[act]['data']['inf_data'].functionality=0.0
        for idxl, l in enumerate(self.players):
            for act in action[idxl]:
                action_conv=str(act[0])+"."+str(act[1])
                flow_results[1].add_action(0,action_conv)
        return flow_results
        
    def build_game(self, save_model=None, suffix=''):
        '''
        This function constructs the normal restoratuion game

        Parameters
        ----------
        write_to_file_dir : str, optional
            DESCRIPTION. The default is None.
        file_name : str, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        dimensions = [len(val) for key, val in self.actions.items()]
        self.normgame = gambit.Game.new_table(dimensions)
        for idxl, l in enumerate(self.players):
            for idxal, al in enumerate(self.actions[l]):
                self.normgame.players[idxl].strategies[idxal].label = str(al)

        for _, ac in self.payoffs.items():
            payoff_cell_corrdinate = []
            for keyl, l in ac.items():
                payoff_cell_corrdinate.append(str(l[0]))
            for keyl, l in ac.items():
                index = self.players.index(keyl)
                self.normgame[tuple(payoff_cell_corrdinate)][index] = fractions.Fraction(l[1])
        if save_model:
            if not os.path.exists(save_model):
                os.makedirs(save_model)
            with open(save_model+'/ne_game_'+suffix+".txt", "w") as text_file:
                text_file.write(self.normgame.write(format='native')[2:-1].replace('\\n', '\n'))

    def solve_game(self, method='enumpure', print_to_cmd=False):
        '''
        This function solves the normal restoration games given a solving method

        Parameters
        ----------
        method : str, optional
            DESCRIPTION. The default is 'enumpure'.
        print_to_cmd : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        if method == 'enumpure':
            gambit_solution = gambit.nash.enumpure_solve(self.normgame)
        else:
            sys.exit('The solution method is not valid')
            
        self.solution = GameSolution(self.players, gambit_solution, self.actions)
        # Find the INDP results correpsonding to solutions
        for key, ac in self.payoffs.items():
            for _, sol in self.solution.sol.items():
                pay_vec = []
                sol_vec = []
                for l in self.players:
                    pay_vec.append(ac[l][0])
                    sol_vec.append(sol['P'+str(l)+' actions']) 
                if pay_vec == sol_vec:
                    sol['full results'] = self.temp_storage[key]
        self.temp_storage = {} #Empty the temprory attribute
        # Print to console
        if print_to_cmd:
            print("NE Solutions(s)")
            for idx, x in enumerate(gambit_solution):
                print('%d.'%(idx+1))
                print([float(y) for y in x])
                print([float(y) for y in x.payoff()])

    def choose_equilibrium(self):
        total_cost_dict = {}
        for key, sol in self.solution.sol.items():
            total_cost_dict[key] = sol['total cost']
        min_val = min(total_cost_dict.items())[1]
        min_keys = [k for k, v in total_cost_dict.items() if v == min_val]
        if len(min_keys) == 1:
            self.chosen_equilibrium = self.solution.sol[min_keys[0]]
        else:
            self.chosen_equilibrium = self.solution.sol[random.choice(min_keys)]
            
    def find_optimal_solution(self):
        '''
        This function finds the centralized, optimal solution corresponding to
        the normal restoration game using INDP

        Returns
        -------
        opt_sol : dict
            DESCRIPTION.

        '''
        v_r = sum([x for key,x in self.v_r.items()])
        indp_res = indp.indp(self.net, v_r=v_r, T=1, layers=self.players,
                             controlled_layers=self.players, functionality={},
                             print_cmd=False, time_limit=None)[1]

        for _, l in enumerate(self.players):
            self.optimal_solution['P'+str(l)+' actions'] = ()
            for act in indp_res.results_layer[l][0]['actions']:
                act = [int(y) for y in act.split(".")]
                self.optimal_solution['P'+str(l)+' actions'] += ((act[0], act[1]), )
            self.optimal_solution['P'+str(l)+' payoff'] = indp_res.results_layer[l][0]['costs']['Total']
        self.optimal_solution['full result'] = indp_res

class GameSolution:
    '''
    This class extract and save the solution of the normal game
    '''
    def __init__(self, L, sol, actions):
        self.players = L
        self.gambit_sol = sol
        self.sol = self.extract_solution(actions)

    def extract_solution(self, actions):
        '''
        This function extract the solution of the normal game from the solution
        structure from gambit

        Parameters
        ----------
        actions : dict
            DESCRIPTION.

        Returns
        -------
        sol : dict
            DESCRIPTION.

        '''
        sup_action = []
        sol = {}
        for l in self.players:
            sup_action.extend(actions[l])
        for idx, x in enumerate(self.gambit_sol):
            sol[idx] = {}
            c = -1
            for y in x:
                c += 1
                if y > 0.0:
                    act = sup_action[c]
                    for l in self.players:
                        if act in actions[l]:
                            plyr = l
                    sol[idx]['P'+str(plyr)+' actions'] = act
            total_cost = 0
            for idxl, l in enumerate(self.players):
                sol[idx]['P'+str(l)+' payoff'] = float(x.payoff()[idxl])
                total_cost -= float(x.payoff()[idxl])
            sol[idx]['total cost'] = total_cost
        return sol

class InfrastructureGame:
    '''
    This class is employed to find the restoration strategy for an interdepndent network
    using game theoretic methods over a given time horizon
    '''
    def __init__(self, params):
        #: Basic attributes
        self.layers = params['L']
        self.game_type = params['ALGORITHM']
        self.judge_type = params["JUDGMENT_TYPE"]
        self.magnitude = params['MAGNITUDE']
        self.sample = params["SIM_NUMBER"]
        self.net = self.set_network(params)
        self.time_steps = self.set_time_steps(params['T'], params['NUM_ITERATIONS'])
        self.objs = {t:0 for t in range(1,self.time_steps+1)}
        self.results = indputils.INDPResults(self.layers)
        #: Resource allocation attributes
        self.resource = dindpclasses.ResourceModel(params, self.time_steps)
        self.res_alloc_type = self.resource.type
        self.v_r = self.resource.v_r
        self.output_dir = self.set_out_dir(params['OUTPUT_DIR'], self.magnitude)

    def set_out_dir(self, root, mag):
        '''
        Parameters
        ----------
        root : TYPE
            DESCRIPTION.
        mag : TYPE
            DESCRIPTION.

        Returns
        -------
        output_dir : TYPE
            DESCRIPTION.

        '''
        output_dir = root+'_L'+str(len(self.layers))+'_m'+str(mag)+"_v"+\
            str(self.resource.sum_resource)+'_'+self.res_alloc_type
        if self.res_alloc_type == 'AUCTION':
            output_dir += '_'+self.resource.auction_model.auction_type+\
                '_'+self.resource.auction_model.valuation_type
        return output_dir

    def set_network(self, params):
        '''

        Parameters
        ----------
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if "N" not in params:
            sys.exit('No initial network object for: '+self.judge_type+', '+\
                     self.res_alloc_type)
        else:
            return copy.deepcopy(params["N"]) #!!! deepcopy

    @staticmethod
    def set_time_steps(T, num_iter):
        '''

        Parameters
        ----------
        T : TYPE
            DESCRIPTION.
        num_iter : TYPE
            DESCRIPTION.

        Returns
        -------
        num_iter : TYPE
            DESCRIPTION.

        '''
        if T != 1:#!!! to be modified in futher expansions
            sys.exit('ERROR: T!=1, JC currently only supports iINDP, not td_INDP.')
        else:
            return num_iter
        
    def run_game(self, compute_optimal=False, save_results=True, plot=False):
        print('--Running Game: '+self.game_type+', resource allocation: '+self.res_alloc_type)
        if self.game_type == 'NORMALGAME':
            for t in self.objs.keys():
                print("-Time Step", t, "/", self.time_steps)
                self.objs[t] = NormalGame(self.layers, self.net, self.v_r[t])
                self.objs[t].compute_payoffs()
                game_start = time.time()
                if self.objs[t].payoffs:
                    self.objs[t].build_game(save_model=self.output_dir, suffix=str(t))
                    self.objs[t].solve_game(method='enumpure', print_to_cmd=True)
                    self.objs[t].choose_equilibrium()
                    game_time = time.time()-game_start
                    if compute_optimal:
                        self.objs[t].find_optimal_solution()
                    if plot:
                        gameplots.plot_ne_sol_2player(self.objs[t], suffix=str(t))
                    
                    ne_results = self.objs[t].chosen_equilibrium['full results']
                    ne_results[1].results[0]['run_time'] = game_time
                    self.results.extend(ne_results[1], t_offset=t)
                    indp.apply_recovery(self.net, self.results, t)
                    self.results.add_components(t, indputils.INDPComponents.\
                                                calculate_components(ne_results[0],
                                                self.net, layers=self.layers))
                else:
                    print('No further action is feasible')
        if save_results:
            # self.save_object_to_file()
            self.save_results_to_file()
            
        
    def save_object_to_file(self):
        '''

        Parameters
        ----------
        sample : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self.output_dir+'/objs_'+str(self.sample)+'.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    
    def save_results_to_file(self):
        '''

        Parameters
        ----------
        sample : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        output_dir_agents = self.output_dir+'/agents'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(output_dir_agents):
            os.makedirs(output_dir_agents)
        self.results.to_csv(self.output_dir, self.sample)
        self.results.to_csv_layer(output_dir_agents, self.sample)