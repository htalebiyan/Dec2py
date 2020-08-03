'''
This module contain the classes used for game theoritical analysis of interdpendent
network restoration
'''
import sys
import os
# import time
import copy
import itertools
import fractions
# import numpy as np
# import pandas as pd
import gambit
import dindpclasses
import indpalt
import indp

class NormalGame:
    '''
    This class models a normal restoration game
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
        self.optimal_solution = self.find_optimal_solution()

    def find_actions(self):
        '''
        This function finds all possible restoration actions for each player

        Returns
        -------
        actions : dict
            DESCRIPTION.

        '''
        actions = {}
        for idx, l in enumerate(self.players):
            damaged_nodes = [n for n, d in self.net.G.nodes(data=True) if\
                        d['data']['inf_data'].functionality == 0.0 and n[1] == l]
            damaged_arcs = [(u, v) for u, v, a in self.net.G.edges(data=True) if\
                        a['data']['inf_data'].functionality == 0.0 and u[1] == l and v[1] == l]
            basic_actions = damaged_nodes + damaged_arcs
            actions[l] = []
            for v in range(self.v_r[idx]):
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
        for idx, l in enumerate(self.players):
            actions_super_set.append([])
            actions_super_set[-1].extend(self.actions[l])
        action_comb = list(itertools.product(*actions_super_set))
        # compute payoffs for each possible combinations of actions
        for idx, ac in enumerate(action_comb):
            self.payoffs[idx] = {}
            decision_vars = {0:{}}
            for n, d in self.net.G.nodes(data=True):
                decision_vars[0]['w_'+str(n)] = d['data']['inf_data'].functionality
            for idxl, l in enumerate(self.players):
                for act in ac[idxl]:
                    decision_vars[0]['w_'+str(act)] = 1.0
            flow_results = indpalt.flow_problem(self.net, v_r=0, layers=self.players,
                                                controlled_layers=self.players,
                                                decision_vars=decision_vars,
                                                print_cmd=True, time_limit=None)
            for idxl, l in enumerate(self.players):
                # Minus sign because we want to minimize the cost
                payoff_layer = -flow_results[1].results_layer[l][0]['costs']['Total']
                self.payoffs[idx][idxl] = [ac[idxl], payoff_layer]
                self.payoff_time[idx] = flow_results[1].results[0]['run_time']

    def build_game(self, write_to_file_dir=None, file_name=None):
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
                self.normgame[tuple(payoff_cell_corrdinate)][keyl] = fractions.Fraction(l[1])
        if write_to_file_dir:
            if file_name:
                if not os.path.exists(write_to_file_dir):
                    os.makedirs(write_to_file_dir)
                with open(write_to_file_dir+'/'+file_name+".txt", "w") as text_file:
                    text_file.write(self.normgame.write(format='native')[2:-1].replace('\\n', '\n'))
            else:
                sys.exit('Provide a file name')

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
            sol = gambit.nash.enumpure_solve(self.normgame)
            self.solution = GameSolution(self.players, sol, self.actions)
            if print_to_cmd:
                print("NE Solutions(s)")
                for idx, x in enumerate(sol):
                    print('%d.'%(idx+1))
                    print([float(y) for y in x])
                    print([float(y) for y in x.payoff()])
        else:
            sys.exit('The solution method is not valid')

    def find_optimal_solution(self):
        '''
        This function finds the centralized, optimal solution corresponding to
        the normal restoration game using INDP

        Returns
        -------
        opt_sol : dict
            DESCRIPTION.

        '''
        opt_sol = {}
        indp_res = indp.indp(self.net, v_r=sum(self.v_r), T=1, layers=self.players,
                             controlled_layers=self.players, functionality={},
                             print_cmd=False, time_limit=None)[1]

        for _, l in enumerate(self.players):
            opt_sol['P'+str(l)+' actions'] = ()
            for act in indp_res.results_layer[l][0]['actions']:
                act = [int(y) for y in act.split(".")]
                opt_sol['P'+str(l)+' actions'] += ((act[0], act[1]), )
            opt_sol['P'+str(l)+' payoff'] = indp_res.results_layer[l][0]['costs']['Total']
        return opt_sol

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
            for idxl, l in enumerate(self.players):
                sol[idx]['P'+str(l)+' payoff'] = float(x.payoff()[idxl])
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
        self.net = self.set_network(params)
        self.time_steps = self.set_time_steps(params['T'], params['NUM_ITERATIONS'])
        #: Resource allocation attributes
        self.resource = dindpclasses.ResourceModel(params, self.time_steps)
        self.res_alloc_type = self.resource.type
        self.v_r = self.resource.v_r
        self.output_dir = self.set_out_dir(params['OUTPUT_DIR'], params['MAGNITUDE'])

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
