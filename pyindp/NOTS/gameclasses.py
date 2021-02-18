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
#import plots

class NormalGame:
    '''
    This class models a normal (strategic) restoration game for a given time step 
    and finds pure and mixed strategy nash equilibria.

    Attributes
    ----------
    players : list
        List of players, set based on 'L' from input variables of :class:`NormalGame`
    net : :class:`~infrastructure.InfrastructureNetwork`
        Object that stores network information, set based on 'net' from input variables
        of :class:`NormalGame`
    v_r : dict
        Dictionary that stores the number of available resources, :math:`R_c`, for
        the current time step, set based on 'v_r' from input variables of :class:`NormalGame` 
    dependee_nodes : dict
        Dictionary of all dependee nodes in the network
    actions : dict
        Dictionary of all relavant restoration actions (including 'No Action (NA)' and
        possibly 'Other Action (OA)'), which are used as the possible moves by players,
        set by :meth:`find_actions`
    first_actions : list
        List of first action of each player in :attr:`actions`, which is sed to check if
        any action is left for any one the players.
    actions_reduced : bool
        Provisional: If true, actions for at least one agent are more than 1000 and hence are reduced
    payoffs : dict
        Dictionary of payoffs for all possible action profiles, calculated by solving 
        INDP or the corresponfing flow problem. It is populated by :meth:`compute_payoffs`.
    payoff_time : dict
        Time to compute each entry in :attr:`payoffs`
    solving_time : float
        Time to solve the game using :meth:`solve_game`
    normgame : :class:`gambit.Game`
        Normal game object defined by gambit. It is populated by :meth:`build_game`.
    solution : :class:`GameSolution`
        Solution of the normal game. It is populated by :meth:`solve_game`.
    chosen_equilibrium : dict
        Action chosen from Nash equilibria (if there are more than one) as
        the action of the current time step to proceed to the next step. It is
        populated by :meth:`choose_equilibrium`.\n
        The solution is the one with lowest total cost assuming that after many games
        (as supposed in NE) people know which one has the overall lowest total cost.
        If there are more than one minimum total cost equilibrium, one of them is 
        chosen randomly. \n
        If the chosen NE is a mixed strategy, the mixed strategy with the highest
        probability is chosen. If there are more than one of such mixed strategy,
        then one of them is chosen randomly

        .. todo::
            Games: refine the way the game solution is chosen in each time step
    optimal_solution : dict
        Optimal Solution from INDP. It is populated by :meth:`find_optimal_solution`.
    '''
    def __init__(self, L, net, v_r):
        self.players = L
        self.net = net
        self.v_r = v_r
        self.dependee_nodes = {}
        self.actions = self.find_actions()
        self.first_actions = [self.actions[l][0][0] for l in self.players]
        self.actions_reduced = False
        self.payoffs = {}
        self.payoff_time = {}
        self.solving_time = 0.0
        self.normgame = []
        self.solution = []
        self.chosen_equilibrium = {}
        self.optimal_solution = {}

    def __getstate__(self):
        """
        Return state values to be pickled. Gambit game object is deleted when
        pickling. To retirve it, user should save the game to file when building 
        the game and read it later and add it to to the unpickled object
        """
        state = self.__dict__.copy()
        state["normgame"] =  {}
        return state

    def __setstate__(self, state):
        """
        Restore state from the unpickled state values. Gambit game object is deleted when
        pickling. To retirve it, user should save the game to file when building 
        the game and read it later and add it to to the unpickled object
        """
        self.__dict__.update(state)

    def find_actions(self):
        '''
        This function finds all relevant restoration actions for each player
        
        .. todo::
            Games: Add the geographical interdependency to find actions, which means
            to consider the arc repaires as indepndent actions rather than aggregate 
            them in the 'OA' action.

        Returns
        -------
        actions : dict
            Dictionary of all relavant restoration actions.

        '''
        for u,v,a in self.net.G.edges(data=True):
            if a['data']['inf_data'].is_interdep and u[1] in self.players and v[1] in self.players:
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

    def compute_payoffs(self, save_model=None, payoff_dir=None):
        '''
        This function finds all possible combinations of actions and their corresponding
        payoffs considering resource limitations

        Parameters
        ----------
        save_model : list, optional
            The folder name and the current time step, which are needed to create
            the folder that contains INDP models for computing payoffs. The default
            is None, which prevent saving models to file.
        payoff_dir : list, optional
            Address to the file containing past results including payoff values
            for the first time step. For other time steps, the payoff value may 
            have been computed based on different initial conditions. The default
            is None.
        Returns
        -------
        :
            None.

        '''
        actions_super_set = []
        memory_threshold =  5000
        for l in self.players:
            actions_super_set.append([])
            actions_super_set[-1].extend(self.actions[l])
        action_comb = list(itertools.product(*actions_super_set))
        if payoff_dir:
            # Read payoffs for each possible combinations of actions
            with open(payoff_dir, 'rb') as obj_file:
                past_obj = pickle.load(obj_file)
            obj_1 = past_obj.objs[1]
            self.actions = obj_1.actions
            self.v_r = obj_1.v_r
            self.payoffs = obj_1.payoffs
            self.payoff_time = obj_1.payoff_time
        else:
            # compute payoffs for each possible combinations of actions
            for idx, ac in enumerate(action_comb):
                self.payoffs[idx] = {}
                flow_results = self.flow_problem(ac)
                if flow_results:
                    if save_model:
                        indp.save_INDP_model_to_file(flow_results[0], save_model[0],
                                                     save_model[1], suffix=str(ac))
                    for idxl, l in enumerate(self.players):
                        # Minus sign because we want to minimize the cost
                        payoff_layer = -flow_results[1].results_layer[l][0]['costs']['Total']
                        self.payoffs[idx][l] = [ac[idxl], payoff_layer]
                        self.payoff_time[idx] = flow_results[1].results[0]['run_time']
                else:
                    for idxl, l in enumerate(self.players):
                        payoff_layer = -1e100 #!!! Change this to work for general case
                        self.payoffs[idx][l] = [ac[idxl], payoff_layer]
                        self.payoff_time[idx] = 0.0
                if len(action_comb)>memory_threshold and idx%memory_threshold==0 and idx!=0:
                    print(str(idx//memory_threshold), '*', str(memory_threshold))
                #     temp_dir = './temp_payoff_objs'
                #     if not os.path.exists(temp_dir):
                #         os.makedirs(temp_dir)
                #     with open(temp_dir+'/temp_payoff_obj_'+str(idx//memory_threshold)+'.pkl', 'wb') as output:
                #         pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
                #         self.payoffs = {}
                #         self.payoff_time = {}
    def flow_problem(self, action):
        '''
        Solves a flow problem for a given combination of actions

        Damaged arc repairs and damage, non-dependee node repairs are removed
        from actions, and collected under "Other Action (OA)" since they do not
        affect other agents' actions.

        To find OA payoff, I solve an INDP problem with fixed node values. For
        example, assume player 1's relevant damaged nodes (damaged, dependee node)
        are :math:`[1, 2, 3]`. Also, player 2's relevant damaged nodes are
        :math:`[4, 5, 6]`. For an action profile :math:`\{[1, OA], [5, 6, OA]\}`,
        I set :math:`1` to be repaired and :math:`2,3` to stay damaged. Similarly,
        I set :math:`5,6` to to be repaired, and :math:`4` to stay damaged.
        Then, I solve INDP under these restrictions.

        Also, I restrict the number of resources for each layer. Say, if 2 resources
        are available for each layer (i.e. :math:`R_c=2`), I impose this restriction too.
        Effectively, in solving INDP for player 1, I assume that :math:`1` is repaired,
        :math:`2,3` must not be repaired, and the repair of non-relevant elements are
        decided by INDP.

        Furthermore, if there are, for example, :math:`R_c=3` for each player, but
        the action profile is :math:`\{[1], [5, 6]\}`, then I solve INDP by restricting
        :math:`R_c` to 1 for player 1 and to 2 for player 2. Therefore, I make sure that
        only the nodes in the action profile are considered repaired.

        Moreover, by removing arcs from the actions, I am ignoring the geographical
        interdependency among layers.

        .. todo::
            Games: Add the geographical interdependency to computation of payoff values

        Parameters
        ----------
        action : tuple
            Action profile for which the payoff is computed.

        Returns
        -------
        :
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
        flow_results = indp.indp(self.net, v_r=adjusted_v_r, layers=self.players,
                                 controlled_layers=self.players, print_cmd=True,
                                 time_limit=None, fixed_nodes=fixed_nodes,
                                 co_location=False)
        return flow_results

    def build_game(self, save_model=None, suffix=''):
        '''
        This function constructs the normal restoratuion game

        Parameters
        ----------
        save_model : str, optional
            Directory to which the game should be written as a .txt file. The default
            is None, which prevents writing to file.
        suffix : str, optional
            An optional suffix that is added to the file name. The default is ''.

        Returns
        -------
        :
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
            with open(save_model+'/norm_game_'+suffix+".txt", "w") as text_file:
                text_file.write(self.normgame.write(format='native')[2:-1].replace('\\n', '\n'))

    def solve_game(self, method='enumerate_pure', print_to_cmd=False, 
                   game_info=None):
        '''
        This function solves the normal restoration game given a solving method

        Parameters
        ----------
        method : str, optional
            Method to solve the normal game. The default is 'enumerate_pure'.
            Options: enumerate_pure, enumerate_mixed_2p, linear_complementarity_2p,
            linear_programming_2p, simplicial_subdivision,
            iterated_polymatrix_approximation, global_newton_method
        print_to_cmd : bool, optional
            Should the found equilibria be written to console. The default is False.
        game_info :list, optional
            The list of information about the game that should be solved. The first
            item in the list is a class:`~gambit.Game` object similar to
            :py:attr:`~NormalGame.normgame`. The second item is  a list of players of
            the game similar to :py:attr:`~NormalGame.players`. The third item is a
            dictionary that contatins the actions of each player similar to
            :py:attr:`~NormalGame.actions`. The default is 'None', which is the
            equivalent of passing [self.normgame, self.players, self.actions].

        Returns
        -------
        :
            None.

        '''
        game = self.normgame
        player_list = self.players
        action_list = self.actions
        if game_info:
            game = game_info[0]
            player_list = game_info[1]
            action_list = game_info[2]

        start_time = time.time()
        if method == 'enumerate_pure':
            gambit_solution = gambit.nash.enumpure_solve(game)
        elif method == 'enumerate_mixed_2p':
            gambit_solution = gambit.nash.enummixed_solve(game)
        elif method == 'linear_complementarity_2p':
            gambit_solution = gambit.nash.lcp_solve(game)
        elif method == 'linear_programming_2p':
            gambit_solution = gambit.nash.lp_solve(game)
        elif method == 'simplicial_subdivision':
            gambit_solution = gambit.nash.simpdiv_solve(game)
        elif method == 'iterated_polymatrix_approximation':
            gambit_solution = gambit.nash.ipa_solve(game)
        elif method == 'global_newton_method':
            gambit_solution = gambit.nash.gnm_solve(game)
        else:
            sys.exit('The solution method is not valid')
        if len(gambit_solution)==0:
            print('No solution found: switching to pure enumeratiom method')
            gambit_solution = gambit.nash.enumpure_solve(game)
        self.solving_time = time.time()-start_time

        self.solution = GameSolution(player_list, gambit_solution, action_list)
        for _, sol in self.solution.sol.items():
            # Find all combination of action in the case of mixed strategy
            sol_super_set = []
            for l in player_list:
                sol_super_set.append(sol['P'+str(l)+' actions'])
            sol['solution combination'] = list(itertools.product(*sol_super_set))
        # Print to console
        if print_to_cmd:
            print("NE (pure or mixed) Solutions(s)")
            for idx, x in enumerate(gambit_solution):
                print('%d.'%(idx+1))
                print([float(y) for y in x])
                print([float(y) for y in x.payoff()])

    def choose_equilibrium(self, preferred_players=None):
        '''
        Choose one action frm pure or mixed strtegies

        Parameters
        ----------
        excluded_players : dict, optional
            The dictionary of player that should be included and excluded from NE
            choosing process. The default is None, which means all
            players are considered in choosing payoffs.

        Returns
        -------
        None.

        '''
        if preferred_players:
            included_palyers = preferred_players['included']
            excluded_palyers = preferred_players['excluded']
        else:
            included_palyers = self.players
            excluded_palyers = []

        total_cost_dict = {}
        for key, sol in self.solution.sol.items():
            exclude_payoffs = sum([-sol['P'+str(x)+' payoff'] for x in excluded_palyers])
            total_cost_dict[key] = sol['total cost'] - exclude_payoffs
        min_val = min(total_cost_dict.items())[1]
        min_keys = [k for k, v in total_cost_dict.items() if v == min_val]
        # Chosse the lowest cost NE and randomize among several minimum values
        if len(min_keys) == 1:
            sol_key = min_keys[0]
        else:
            sol_key = random.choice(min_keys)
        # Chosse a ranodm action from the action profile if the chosen solution is a mixed NE
        num_profile_actions = len(self.solution.sol[sol_key]['solution combination'])
        if num_profile_actions == 1:
            mixed_index = 0
        else:
            max_prob_idx = {}
            chosen_profile = ()
            for l in included_palyers:
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
        # Compute complete results for the chosen equilibrium
        self.chosen_equilibrium['full result'] = []
        for idx, ac in enumerate(self.chosen_equilibrium['solution combination']):
            if preferred_players:
                # Rewrite the chosen equlibrium to be cmpatible with the basic normal game and players
                if self.chosen_equilibrium['total cost'] is list:
                    self.chosen_equilibrium['total cost'][idx] = total_cost_dict[sol_key]
                else:
                    self.chosen_equilibrium['total cost'] = total_cost_dict[sol_key]
                ac_old = []
                for x in ac:
                    if x[0]=='NA':
                        if x[1] in included_palyers:
                            ac_old.append(x)
                    elif x[0][1] in included_palyers:
                        ac_old.append(x)
                ac_old = tuple(ac_old)
                ac = ()
                for x in ac_old:
                    new_a = ()
                    if x[0] =='NA':
                        new_a += ((x[0], x[1][1]))
                    else:
                        for y in x:
                            new_a += ((y[0], y[1][1]),)
                    ac += (new_a,)
                self.chosen_equilibrium['solution combination'][idx] = ac
            self.chosen_equilibrium['full result'].append(self.flow_problem(ac)[1])
        if len(self.chosen_equilibrium['full result']) == 1 and not preferred_players:
            original_tc = self.chosen_equilibrium['total cost']
            re_comp_tc = self.chosen_equilibrium['full result'][0].results[0]['costs']['Total']
            assert abs((re_comp_tc-original_tc)/original_tc)<0.01,\
            'Error: the re-computed total cost does not match the original one.'

    def find_optimal_solution(self):
        '''
        Computes the centralized, optimal solution corresponding to
        the normal restoration game using INDP

        Returns
        -------
        :
            None.

        '''
        v_r = sum([x for key,x in self.v_r.items()])
        indp_res = indp.indp(self.net, v_r=v_r, T=1, layers=self.players,
                             controlled_layers=self.players, functionality={},
                             print_cmd=False, time_limit=None)[1]

        for _, l in enumerate(self.players):
            self.optimal_solution['P'+str(l)+' actions'] = ()
            for act in indp_res.results_layer[l][0]['actions']:
                if '/' in act: 
                    if ('OA',l) not in self.optimal_solution['P'+str(l)+' actions']:
                        self.optimal_solution['P'+str(l)+' actions'] += (('OA', l),)
                else:
                    act = [int(y) for y in act.split(".")]
                    if ((act[0], act[1])) in self.dependee_nodes.keys():
                        self.optimal_solution['P'+str(l)+' actions'] += ((act[0], act[1]),)
                    elif ('OA',l) not in self.optimal_solution['P'+str(l)+' actions']:
                        self.optimal_solution['P'+str(l)+' actions'] += (('OA', l),)
            if len(indp_res.results_layer[l][0]['actions']) == 0:
                self.optimal_solution['P'+str(l)+' actions'] += ('NA', l)
            self.optimal_solution['P'+str(l)+' payoff'] = indp_res.results_layer[l][0]['costs']['Total']
        self.optimal_solution['total cost'] = indp_res.results[0]['costs']['Total']
        self.optimal_solution['full result'] = indp_res

class BayesianGame(NormalGame):
    '''
    This class models a Bayesian restoration game for a given time step 
    and finds bayes nash equilibria. This class inherits from :class:`NormalGame`.

    Attributes
    ----------
    fundamental_types : list
        List of fundamental types of players. Currently, it consists of two types:

        - Cooperative(C) player, which prefers cooperative and partially cooperative actions.
        - Non-cooperative (N) player, which prefers non-cooperative actions.
    states : dict
        List of state(s) of the game.
    states_payoffs : dict
        List of payoff matrices of state(s) of the game.
    types : dict
        List of players' type(s).
    bayesian_players : list
        List of bayesian players of the game comprising combiantion of players 
        and their types.
    '''
    def __init__(self, L, net, v_r):
        super().__init__(L, net, v_r)
        self.fundamental_types = ['C', 'N']
        self.states = {}
        self.states_payoffs = {}
        self.types = {}
        self.bayesian_players = []
        self.bayesian_payoffs = {}
        self.bayesian_game = []

    def __getstate__(self):
        """
        Return state values to be pickled. Gambit game object is deleted when
        pickling. To retirve it, user should save the game to file when building 
        the game and read it later and add it to to the unpickled object
        """
        state = self.__dict__.copy()
        state["normgame"] =  {}
        state["bayesian_game"] =  {}
        return state

    def label_actions(self, action):
        '''
        This function return the type of an input action in accordance with
        :attr:`fundamental_types`:

            - 'C' (cooperative) action consists of one or several actions that
              some of them are relevent---i.e., reparing damaged nodes that on which
              other players depend. A 'C' action may include 'OA' action as well,
              but it cannot be only one 'OA' action. 
            - 'N' (non-cooperative) action is either 'NA' or 'OA'.

        Parameters
        ----------
        action : tuple
            An action.

        Returns
        -------
        label : str
            The action type.
        '''

        label = None
        if len(action) == 1 and action[0][0] == 'OA':
            label = 'N'
        elif action[0] == 'NA':
            label = 'N'
        else:
            label = 'C'
            for a in action:
                if a[0] == 'OA':
                    label = 'C' #'P'
                    break
        return label

    def set_states(self):
        '''
        This function set the states based on :attr:`fundamental_types` and for
        each state compute the payoff matrix of all players by doubling the payoff
        of actions that are not consistant with the player's type.

        .. todo::
            Games: refine how to reduce importance of action not consistant with
            the player's type.

        Returns
        -------
        None.

        '''
        comb_w_rep = list(itertools.combinations(self.fundamental_types*len(self.players),
                                                 len(self.players)))
        self.states = list(set(comb_w_rep))
        # Assign payoff matrix for each state
        for s in self.states:
            self.states_payoffs[s] = copy.deepcopy(self.payoffs) #!!!deepcopy
            for key, val in self.states_payoffs[s].items():
                for idx, l in enumerate(self.players):
                    label = self.label_actions(val[l][0])
                    if label != s[idx]:
                        val[l][1] *= 2 #!!!refine how to reduce importance of other types

    def set_types(self, beliefs):
        '''
        This function set players type based on the beliefs it receives. Currently,
        it can interpret the follwoing signnals:

            - Uninformed ('U'): Players  do not have any information about other players,
              and assign equal probability to other players' types.

        Parameters
        ----------
        beliefs : dict
            The collection of beliefs for all players .

        Returns
        -------
        None.

        '''
        for idx, l in enumerate(self.players):
            self.types[l] = {x:{} for x in self.fundamental_types}
            if beliefs[l]=="U":
                for t in self.fundamental_types:
                    for s in self.states:
                        if s[idx]==t:
                            self.types[l][t][s] = 1/len(self.fundamental_types)
                        else:
                            self.types[l][t][s] = 0.0
            else:
                sys.exit('Error: wrong signal name')

    def create_bayesian_players(self):
        '''
        This function create one player for each combination of player and its types.

        Returns
        -------
        None.

        '''
        for lyr, val in self.types.items():
            for typ, valtyp in val.items():
                self.bayesian_players.append((typ, lyr))

    def compute_bayesian_payoffs(self):
        bayes_actions = []
        actions_super_set = []
        for b in self.bayesian_players:
            actions_super_set.append([])
            actions_super_set[-1].extend(self.actions[b[1]])
        action_comb = list(itertools.product(*actions_super_set))
        # compute payoffs for each possible combinations of actions
        for idx, ac in enumerate(action_comb):
            self.bayesian_payoffs[idx] = {}
            for idxb, b in enumerate(self.bayesian_players):
                payoff = 0
                for s in self.states:
                    prob_state = self.types[b[1]][b[0]][s]
                    if prob_state>0:
                        payoff_dict = self.states_payoffs[s]
                        payoff_keys = list(payoff_dict.keys())
                        for idxl, l in enumerate(self.players):
                            temp = []
                            player_index = self.bayesian_players.index((s[idxl], l))
                            for key in payoff_keys:
                                if payoff_dict[key][l][0] == ac[player_index]:
                                    temp.append(key)
                            payoff_keys = [x for x in temp]
                        if len(payoff_keys) != 1:
                            sys.exit('Error: cannot find the payoff value for', ac)
                        utility_val = payoff_dict[payoff_keys[0]][b[1]][1]
                        payoff += prob_state*utility_val
                self.bayesian_payoffs[idx][b] = [ac[idxb], payoff]

    def build_bayesian_game(self, save_model=None, suffix=''):
        '''
        This function constructs the bayesian restoratuion game

        Parameters
        ----------
        save_model : str, optional
            Directory to which the game should be written as a .txt file. The default
            is None, which prevents writing to file.
        suffix : str, optional
            An optional suffix thta is added to the file name. The default is ''.

        Returns
        -------
        :
            None.

        '''
        dimensions = []
        for b in self.bayesian_players:
            dimensions.append(len(self.actions[b[1]]))
        self.bayesian_game = gambit.Game.new_table(dimensions)
        for idxl, l in enumerate(self.bayesian_players):
            for idxal, al in enumerate(self.actions[l[1]]):
                self.bayesian_game.players[idxl].strategies[idxal].label = str(al)
        for _, ac in self.bayesian_payoffs.items():
            payoff_cell_corrdinate = []
            for keyl, l in ac.items():
                payoff_cell_corrdinate.append(str(l[0]))
            for keyl, l in ac.items():
                index = self.bayesian_players.index(keyl)
                self.bayesian_game[tuple(payoff_cell_corrdinate)][index] = fractions.Fraction(l[1])
        if save_model:
            if not os.path.exists(save_model):
                os.makedirs(save_model)
            with open(save_model+'/bayes_game_'+suffix+".txt", "w") as text_file:
                text_file.write(self.bayesian_game.write(format='native')[2:-1].replace('\\n', '\n'))

class GameSolution:
    '''
    This class extracts (from the gambit solution) and saves the solution of the normal game

    Attributes
    ----------
    players : list
        List of players, set based on 'L' from input variables of :class:`GameSolution`
    gambit_sol : list
        Solutions to the game computed by gambit, set based on 'sol' from input variables
        of :class:`GameSolution`
    sol : dict
        Dictionary of solutions of the normal game, including actions, their probabilities,
        payoffs, and the total cost,  set by :meth:`extract_solution`.
    '''

    def __init__(self, L, sol, actions):
        self.players = L
        self.gambit_sol = sol
        self.sol = self.extract_solution(actions)

    def __getstate__(self):
        """
        Return state values to be pickled. gambit_sol is deleted when
        pickling. To retirve it, user should save the game to file when building 
        the game and read it later and add it to to the unpickled object
        """
        state = self.__dict__.copy()
        state["gambit_sol"] = {}
        return state

    def __setstate__(self, state):
        """
        Restore state from the unpickled state values. gambit_sol is deleted when
        pickling. To retirve it, user should save the game to file when building 
        the game and read it later and add it to to the unpickled object
        """
        self.__dict__.update(state)
        state["gambit_sol"] = {}
        
    def extract_solution(self, actions):
        '''
        This function extracts the solution of the normal game from the solution
        structure from gambit

        Parameters
        ----------
        actions : dict
            Possible actions for each palyer.

        Returns
        -------
        sol : dict
            Dictionary of solutions of the normal game, including actions, their probabilities,
            payoffs, and the total cost.

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
                    sol[idx]['P'+str(plyr)+' action probs'].append(float(y))
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

    .. todo::
        Games: Add bayesian games

    Attributes
    ----------
    layers : list
        List of layers (= players), set based on ['L'] in :class:`params <InfrastructureGame>`
    game_type : str
        Type of the game, set based on ['ALGORITHM'] in :class:`params <InfrastructureGame>` \n
            Options: NORMALGAME
    equib_alg : str
        Algorithm to solve the game, set based on ['EQUIBALG'] in :class:`params <InfrastructureGame>` \n
            Options: enumpure_solve, enummixed_solve, lcp_solve, lp_solve, simpdiv_solve,
            ipa_solve, gnm_solve
    magnitude : int
        Magnitude parameter of the current simulation, set based on ['MAGNITUDE'] in
        :class:`params <InfrastructureGame>`
    sample : int
        Sample parameter of the current simulation, set based on ['SIM_NUMBER'] in
        :class:`params <InfrastructureGame>`
    net : :class:`~infrastructure.InfrastructureNetwork`
        Object that stores network information, set based on :class:`params <InfrastructureGame>`
        using :meth:`set_network`
    time_steps : int
        Number of time steps pf the restoration process, set based on
        :class:`params <InfrastructureGame>` using :meth:`set_time_steps`
    objs : dict of :class:`NormalGame`
        Dictionary of game objects (:class:`NormalGame`) for all time steps of the restoration
    judgments : :class:`~dindpclasses.JudgmentModel`
        Object that stores the judgment attitude of agents, which is only needed 
        for computing the resource allocation when using auction. So, it is not
        used in building or solving the games
    results : :class:`~indputils.INDPResults`
        Object that stores the restoration strategies for all time steps of the
        restoration process
    resource : :class:`~dindpclasses.ResourceModel`
        Model that allocates resources and stores the resource allocation data for
        all time steps of the restoration process
    res_alloc_type : str
        Resource allocation method
    v_r : dict
        Dictionary that stores the number of available resources, :math:`R_c`, for
        all time steps of the restoration process
    output_dir : str
        Directory to which the results are written, set by :meth:`set_out_dir`
    '''
    def __init__(self, params):
        self.layers = params['L']
        self.game_type = params['ALGORITHM']
        self.beliefs = params['BELIEFS']
        self.signals = params['SIGNALS']
        self.equib_alg = params['EQUIBALG']
        self.magnitude = params['MAGNITUDE']
        self.sample = params["SIM_NUMBER"]
        self.net = self.set_network(params)
        self.time_steps = self.set_time_steps(params['T'], params['NUM_ITERATIONS'])
        self.objs = {t:0 for t in range(1,self.time_steps+1)}
        self.judgments = dindpclasses.JudgmentModel(params, self.time_steps)
        self.results = indputils.INDPResults(self.layers)
        self.resource = dindpclasses.ResourceModel(params, self.time_steps)
        self.res_alloc_type = self.resource.type
        self.v_r = self.resource.v_r
        self.output_dir = self.set_out_dir(params['OUTPUT_DIR'])
        self.payoff_dir = self.set_payoff_dir(params['PAYOFF_DIR'])

    def run_game(self, print_cmd=True, compute_optimal=False, save_results=True,
                 plot=False, save_model=False,):
        '''
        Runs the infrastructure restoarion game for a given number of :attr:`time_steps`

        Parameters
        ----------
        print_cmd : bool, optional
            Should the game solution be printed to the command line. The default is True.
        compute_optimal : bool, optional
            Should the optimal restoarion action be found in each time step. The default is False.
        save_results : bool, optional
            Should the results and game be written to file. The default is True.
        plot : bool, optional
            Should the payoff matrix be plotted (only for 2-players games). The default is False.
        save_model : bool, optional
            Should the games and indp models to compute payoffs be written to file. The default is False.

        Returns
        -------
        :
            None

        '''
        save_model_dir = save_model
        save_payoff_info = save_model
        for t in self.objs.keys():
            print("-Time Step", t, "/", self.time_steps)
            #: Resource Allocation
            res_alloc_time_start = time.time()
            if self.resource.type == 'AUCTION':
                self.resource.auction_model.auction_resources(obj=self,
                                                              time_step=t,
                                                              print_cmd=print_cmd,
                                                              compute_poa=True)
            self.resource.time[t] = time.time()-res_alloc_time_start
            # Create game object
            if self.game_type == 'NORMALGAME':
                self.objs[t] = NormalGame(self.layers, self.net, self.v_r[t])
            elif self.game_type == 'BAYESGAME':
                self.objs[t] = BayesianGame(self.layers, self.net, self.v_r[t])
            else:
                sys.exit('Error: wrong algorithm name for Infrastructure Game.')
            # Compute payoffs
            if print_cmd:
                print("Computing (or reading) payoffs...")
            if save_payoff_info:
                save_payoff_info = [self.output_dir+'/payoff_models', t]
            if set(self.objs[t].first_actions) == {'NA'}:
                pass
            elif t==1 and self.payoff_dir:
                self.objs[t].compute_payoffs(save_model=save_payoff_info,
                                             payoff_dir=self.payoff_dir)
                if self.v_r[t] != self.objs[t].v_r:
                    if self.resource.type != 'UNIFORM':
                        sys.exit('Error: read obj changes v_r invalidly')
                    else:
                        self.v_r[t] = self.objs[t].v_r
                        self.resource.v_r[t] = self.objs[t].v_r
            else:
                self.objs[t].compute_payoffs(save_model=save_payoff_info)
                if save_results: #!!! removewhen don't need to save after payoff
                    self.save_object_to_file()
            # Solve game
            if self.objs[t].payoffs:
                if print_cmd:
                    print("Building and Solving the game...")
                if compute_optimal:
                    self.objs[t].find_optimal_solution()

                if self.game_type == 'NORMALGAME':
                    if save_model_dir:
                        save_model_dir = self.output_dir+'/games'
                    self.objs[t].build_game(save_model=save_model_dir,
                                            suffix=str(self.sample)+'_t'+str(t))
                    game_start = time.time()
                    self.objs[t].solve_game(method=self.equib_alg, print_to_cmd=print_cmd)
                    game_time = time.time()-game_start
                    self.objs[t].choose_equilibrium()
                    if plot and len(self.layers)==2:
                        if not os.path.exists(self.output_dir+'/payoff_matrix'):
                            os.makedirs(self.output_dir+'/payoff_matrix')
                        plots.plot_ne_sol_2player(self.objs[t], suffix=str(self.sample)+'_t'+str(t),
                                                  plot_dir=self.output_dir+'/payoff_matrix')
                elif self.game_type == 'BAYESGAME':
                    self.objs[t].set_states()
                    self.objs[t].set_types(self.beliefs)
                    self.objs[t].create_bayesian_players()
                    self.objs[t].compute_bayesian_payoffs()
                    if save_model:
                        save_model_dir = self.output_dir+'/bayesian_games'
                    self.objs[t].build_bayesian_game(save_model=save_model_dir,
                                                     suffix=str(self.sample)+'_t'+str(t))
                    ### Game info needed to pass to the solver
                    game_info = [self.objs[t].bayesian_game, self.objs[t].bayesian_players]
                    action_list = {}
                    for b in self.objs[t].bayesian_players:
                        action_list[b] = copy.deepcopy(self.objs[t].actions[b[1]])
                        for idx, ac in enumerate(action_list[b]):
                            if ac[0] == 'NA':
                                action_list[b][idx] = (ac[0], b)
                            else:
                                temp = ()
                                for a in ac:
                                    temp += ((a[0], b),)
                                action_list[b][idx] = temp
                    game_info.append(action_list)
                    game_start = time.time()
                    self.objs[t].solve_game(method=self.equib_alg, print_to_cmd=print_cmd,
                                            game_info=game_info)
                    game_time = time.time()-game_start
                    ### Use signals to find interim NE as the chosen solution
                    preferred_players = {'included': [(x,i) for i,x in self.signals.items()]}
                    preferred_players['excluded'] =  [x for x in self.objs[t].bayesian_players\
                                                      if x not in preferred_players['included']]
                    self.objs[t].choose_equilibrium(preferred_players=preferred_players)

                mixed_index = self.objs[t].chosen_equilibrium['chosen mixed profile action']
                ne_results = self.objs[t].chosen_equilibrium['full result'][mixed_index]
                ne_results.results[0]['run_time'] = game_time
                self.results.extend(ne_results, t_offset=t)
                indp.apply_recovery(self.net, self.results, t)
            else:
                print('No further action for any of players')
                ne_results = indputils.INDPResults(self.layers)
                ne_results.results[0] = copy.deepcopy(self.results[t-1]) #!!!deepcopy
                costs = ne_results.results[0]['costs']
                costs['Total'] -= costs['Arc']+costs['Node']+costs['Space Prep']
                costs['Total no disconnection'] -= costs['Arc']+costs['Node']+costs['Space Prep']
                costs['Arc'] = 0.0
                costs['Node'] = 0.0
                costs['Space Prep'] = 0.0
                ne_results.results[0]['actions'] = []
                ne_results.results[0]['run_time'] = 0
                for l in self.layers:
                    ne_results.results_layer[l][0] = copy.deepcopy(self.results.results_layer[l][t-1]) #!!!deepcopy
                    costs = ne_results.results_layer[l][0]['costs']
                    costs['Total'] -= costs['Arc']+costs['Node']+costs['Space Prep']
                    costs['Total no disconnection'] -= costs['Arc']+costs['Node']+costs['Space Prep']
                    costs['Arc'] = 0.0
                    costs['Node'] = 0.0
                    costs['Space Prep'] = 0.0
                    ne_results.results_layer[l][0]['actions'] = []
                    ne_results.results_layer[l][0]['run_time'] = 0
                self.results.extend(ne_results, t_offset=t)
        if save_results:
            self.save_object_to_file()
            self.save_results_to_file()

    def set_out_dir(self, root):
        '''
        This function generates and sets the directory to which the results are written

        Parameters
        ----------
        root : str
            Root directory to write results

        Returns
        -------
        output_dir : str
            Directory to which the results are written

        '''
        output_dir = root+'_L'+str(len(self.layers))+'_m'+str(self.magnitude)+"_v"+\
            str(self.resource.sum_resource)+'_'+self.judgments.judgment_type+\
            '_'+self.res_alloc_type
        if self.res_alloc_type == 'AUCTION':
            output_dir += '_'+self.resource.auction_model.auction_type+\
                '_'+self.resource.auction_model.valuation_type
        return output_dir

    def set_payoff_dir(self, root):
        '''
        This function generates and sets the directory to which the past results
        were written, from which the payoffs for the first time step are read

        Parameters
        ----------
        root : str
            Root directory to read past results

        Returns
        -------
        payoff_dir : str
            Directory from which the payoffs are read

        '''
        if root:
            payoff_dir = root+'_L'+str(len(self.layers))+'_m'+str(self.magnitude)+"_v"+\
                str(self.resource.sum_resource)+'_'+self.judgments.judgment_type+\
                '_'+self.res_alloc_type
            if self.res_alloc_type == 'AUCTION':
                payoff_dir += '_'+self.resource.auction_model.auction_type+\
                    '_'+self.resource.auction_model.valuation_type
            return payoff_dir+'/objs_'+str(self.sample)+'.pkl'
        else:
            return None

    def set_network(self, params):
        '''
        Checks if the network object exists, and if so, make a deepcopy of it to
        preserve the initial netowrk object as the initial netowrk object is used for
        all simulations , and must not be altered

        Parameters
        ----------
        params : dict
            Dictionary of input paramters

        Returns
        -------
        :class:`~infrastructure.InfrastructureNetwork`
            Network Object

        '''
        if "N" not in params:
            sys.exit('No initial network object for: '+self.judge_type+', '+\
                     self.res_alloc_type)
        else:
            return copy.deepcopy(params["N"]) #!!! deepcopy

    @staticmethod
    def set_time_steps(T, num_iter):
        '''
        Checks if the window length is equal to one as the current version of
        games is devised based on itrative INDP

        .. todo::
            Games: Exapnd the code to imitate td-INDP

        Parameters
        ----------
        T : int
            Window lenght
        num_iter : TYPE
            Number of time steps

        Returns
        -------
        num_iter : int
            Number of time steps.

        '''
        if T != 1:#!!! to be modified in futher expansions
            sys.exit('ERROR: T!=1, JC currently only supports iINDP, not td_INDP.')
        else:
            return num_iter

    def save_object_to_file(self):
        '''
        Writes the object to file using pickle

        Parameters
        ----------
        sample : int
            Sample paramter of the current simulation

        Returns
        -------
        :
            None.

        '''
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self.output_dir+'/objs_'+str(self.sample)+'.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_results_to_file(self):
        '''
        Writes results to file

        Parameters
        ----------
        sample : int
            Sample paramter of the current simulation

        Returns
        -------
        :
            None.

        '''
        output_dir_agents = self.output_dir+'/agents'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(output_dir_agents):
            os.makedirs(output_dir_agents)
        self.results.to_csv(self.output_dir, self.sample)
        self.results.to_csv_layer(output_dir_agents, self.sample)