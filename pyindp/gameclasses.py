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
        self.dependee_nodes = {}
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
        This function finds all relevant restoration actions for each player

        Returns
        -------
        actions : dict
            DESCRIPTION.

        '''
        for u,v,a in self.net.G.edges(data=True):
            if a['data']['inf_data'].is_interdep:
                if u not in self.dependee_nodes:
                    self.dependee_nodes[u]=[]    
                self.dependee_nodes[u].append((v,a['data']['inf_data'].gamma))
        actions = {}
        for l in self.players:
            damaged_nodes = [n for n, d in self.net.G.nodes(data=True) if\
                        d['data']['inf_data'].functionality == 0.0 and n[1] == l]
            damaged_arcs = [(u, v) for u, v, a in self.net.G.edges(data=True) if\
                        a['data']['inf_data'].functionality == 0.0 and u[1] == l and v[1] == l]
            '''
            Arc repaire actions are collected under "other action (OA)" since
            the arc actions do not affect the decision of other palyers
            '''
            other_action = False
            if len(damaged_arcs) > 0:
                other_action = True
            '''
            Non-depndeee node repaire actions are collected under "other action (OA)" 
            since these actions do not affect the decision of other palyers
            '''
            rel_actions = []
            for n in damaged_nodes:
                if n in self.dependee_nodes.keys():
                    rel_actions.append(n)
                else:
                    other_action = True
            if other_action:
                rel_actions.extend([('OA',l)])
            actions[l] = []
            for v in range(self.v_r[l]):
                actions[l].extend(list(itertools.combinations(rel_actions, v+1)))
            '''
            "No Action (NA)" is added to possible actions
            '''
            actions[l].extend([('NA',l)])
        return actions

    def compute_payoffs(self, save_model=None):
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
            if save_model:
                indp.save_INDP_model_to_file(flow_results[0], save_model[0],
                                             save_model[1], suffix=str(ac))
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
        
        Damaged arc repairs and damage, non-dependee node repairs are removed 
        from action and collected under "Other Action (OA)" since they do not 
        affect the action of other agents.
        
        To find OA payoff, I solve an INDP problem with fixed node values. For 
        example, assume player 1's relevant damaged nodes (damaged, dependee node)
        [(1,1), (2,1), (3,1)]. Also, player 2's relevant damaged nodes are
        [(1,2), (2,2), (3,2)]. For an action profile {[(1,1), (OA,1)],
        [(2,2), (3,2), (OA,2)]}, I set (1,1) to 1 and (2,1), (3,1) to 0. Similarly,
        I set (2,2) and (3,2) to 1, and (1,2) to 0. Then, I solve INDP by given 
        these restrictions.
        
        Also, I restrict the number of resources for each layer. Say, if Rc=2 
        for each layer, I impose this restriction too. Effectively, in solving 
        INDP for player 1, I assume that (1,1) is repaired, (2,1), (3,1) must not 
        be repaired, and the repair of non-relevant elements are decided by INDP.

        Furthermore, if there are, for example, 3 resources available (Rc=3) for
        each player but the action profile is {[(1,1),[(2,2), (3,2)]}, then I solve
        INDP by restricting Rc to 1 for player 1 and to 2  for player 2. Therefore,
        I make sure that only the nodes in the action profile are considered repaired.
        
        Also, by removing arcs from the actions, I am ignoring the geographical
        interdependency.

        Parameters
        ----------
        action : tuple
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        adjusted_v_r = {key:val for key, val in self.v_r.items()}
        fixed_nodes = {}
        for u in self.dependee_nodes.keys():
            if self.net.G.nodes[u]['data']['inf_data'].functionality != 1.0:
                fixed_nodes[u] = 0.0
        for idxl, l in enumerate(self.players):
            if action[idxl] != ('NA',l):
                for act in action[idxl]:
                    if act != ('OA',l):
                        fixed_nodes[act] = 1.0
                if ('OA',l) not in action[idxl]:
                    adjusted_v_r[l] = len(action[idxl])
            else:
                adjusted_v_r[l] = 0

        adjusted_v_r = [val for _,val in adjusted_v_r.items()]
        flow_results = indp.indp(self.net, v_r=adjusted_v_r, layers=self.players,
                                 controlled_layers=self.players, print_cmd=True,
                                 time_limit=None, fixed_nodes=fixed_nodes)
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

    def solve_game(self, method='enumpure_solve', print_to_cmd=False):
        '''
        This function solves the normal restoration games given a solving method

        Parameters
        ----------
        method : str, optional
            DESCRIPTION. The default is 'enumpure_solve'.
        print_to_cmd : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        if method == 'enumpure_solve':
            gambit_solution = gambit.nash.enumpure_solve(self.normgame)
        elif method == 'enummixed_solve':
            gambit_solution = gambit.nash.enummixed_solve(self.normgame)
        elif method == 'lcp_solve':
            gambit_solution = gambit.nash.lcp_solve(self.normgame)
        elif method == 'lp_solve':
            gambit_solution = gambit.nash.lp_solve(self.normgame)
        elif method == 'simpdiv_solve':
            gambit_solution = gambit.nash.simpdiv_solve(self.normgame)
        elif method == 'ipa_solve':
            gambit_solution = gambit.nash.ipa_solve(self.normgame)
        elif method == 'gnm_solve':
            gambit_solution = gambit.nash.gnm_solve(self.normgame)
        else:
            sys.exit('The solution method is not valid')

        self.solution = GameSolution(self.players, gambit_solution, self.actions)
        # Find the INDP results correpsonding to solutions
        for _, sol in self.solution.sol.items():
            sol['full results'] = []
            # Find all co,bination of action in the case of mixed strategy
            sol_super_set = []
            for l in self.players:
                sol_super_set.append(sol['P'+str(l)+' actions'])
            sol['solution combination'] = list(itertools.product(*sol_super_set))
            for key, ac in self.payoffs.items():
                pay_vec = []
                for l in self.players:
                    pay_vec.append(ac[l][0])
                for sol_vec in sol['solution combination']:
                    if pay_vec == list(sol_vec):
                        sol['full results'].append(self.temp_storage[key])
        self.temp_storage = {} #Empty the temprory attribute
        # Print to console
        if print_to_cmd:
            print("NE (pure or mixed) Solutions(s)")
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
        # Chosse the lowest cost NE
        if len(min_keys) == 1:
            sol_key = min_keys[0]
        else:
            sol_key = random.choice(min_keys)
        # Chosse a ranodm action from the action profile if it is a mixed NE
        num_profile_actions = len(self.solution.sol[sol_key]['solution combination'])    
        if num_profile_actions == 1:
            mixed_index = 0
        else:
            max_prob_idx = {}
            chosen_profile = ()
            for l in self.players:  
                probs = self.solution.sol[sol_key]['P'+str(l)+' action probs']
                max_prob = max(probs)[1]
                max_keys = [c for c in range(len(probs)) if probs[c] == max_prob]
                if len(max_keys) == 1:
                    max_prob_idx[l] = max_keys[0]
                else:
                    max_prob_idx[l] = random.choice(max_keys) 
                chosen_profile += self.solution.sol[sol_key]['P'+str(l)+' action'][max_prob_idx[l]]
            mixed_index = self.solution.sol[sol_key]['solution combination'].index(chosen_profile)
            
        self.chosen_equilibrium = self.solution.sol[sol_key]
        self.chosen_equilibrium['chosen mixed profile action'] = mixed_index

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
                if ((act[0], act[1])) in self.dependee_nodes.keys():
                    self.optimal_solution['P'+str(l)+' actions'] += ((act[0], act[1]),)
                elif ('OA',l) not in self.optimal_solution['P'+str(l)+' actions']:
                    self.optimal_solution['P'+str(l)+' actions'] += (('OA', l),)
            if len(indp_res.results_layer[l][0]['actions']) == 0:
                self.optimal_solution['P'+str(l)+' actions'] += ('NA', l)
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
            for l in self.players:
                sol[idx]['P'+str(l)+' actions'] = []
                sol[idx]['P'+str(l)+' action probs'] = []
            c = -1
            for y in x:
                c += 1
                if y > 1e-3:
                    act = sup_action[c]
                    for l in self.players:
                        if act in actions[l]:
                            plyr = l
                    sol[idx]['P'+str(plyr)+' actions'].append(act)
                    sol[idx]['P'+str(plyr)+' action probs'].append(y)
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
        self.equib_alg = params['EQUIBALG']
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

    def run_game(self, compute_optimal=False, save_results=True, plot=False):
        print('--Running Game: '+self.game_type+', resource allocation: '+self.res_alloc_type)
        if self.game_type == 'NORMALGAME':
            for t in self.objs.keys():
                print("-Time Step", t, "/", self.time_steps)
                self.objs[t] = NormalGame(self.layers, self.net, self.v_r[t])
                self.objs[t].compute_payoffs(save_model=[self.output_dir+'/payoff_models', t])
                game_start = time.time()
                if self.objs[t].payoffs:
                    self.objs[t].build_game(save_model=self.output_dir+'/games',
                                            suffix=str(t))
                    self.objs[t].solve_game(method=self.equib_alg, print_to_cmd=True)
                    self.objs[t].choose_equilibrium()
                    mixed_index = self.objs[t].chosen_equilibrium['chosen mixed profile action']
                    game_time = time.time()-game_start
                    if compute_optimal:
                        self.objs[t].find_optimal_solution()
                    if plot:
                        gameplots.plot_ne_sol_2player(self.objs[t], suffix=str(t))
                    
                    ne_results = self.objs[t].chosen_equilibrium['full results'][mixed_index]
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