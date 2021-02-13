import gameclasses
import infrastructure
import copy

def bayesian_game_BoS():
    '''
    This module builds and solves the Bayesian gane \'BoS\' per Example 276.2
    in Chapter 9 (page 276) of \"M. J. Osborne, 2004, An introduction to game
    theory.\"" :cite:`osborne2004introduction` to verify the  the accuracy of
    algorithms of :class:`~gameclasses.BayesianGame`.

    Returns
    -------
    solutions : list
        The solution(s) of the game.

    '''
    dummy_net = infrastructure.InfrastructureNetwork("dummy_network")
    obj = gameclasses.BayesianGame([1,2], dummy_net, {1:0, 2:0})
    
    obj.actions = {}
    obj.bayesian_players = [('y',1), ('n',1), ('y',2), ('n',2)]
    obj.actions[1] = [('B',1), ('S',1)]
    obj.actions[2] = [('B',2), ('S',2)]
    
    obj.fundamental_types = ['y','n']
    obj.types = {1:{'y':{('y','y'):1/2, ('y','n'):1/2, ('n','y'):0, ('n','n'):0},
                    'n':{('y','y'):0, ('y','n'):0, ('n','y'):1/2, ('n','n'):1/2}},
                 2:{'y':{('y','y'):2/3, ('y','n'):0, ('n','y'):1/3, ('n','n'):0},
                    'n':{('y','y'):0, ('y','n'):2/3, ('n','y'):0, ('n','n'):1/3}}}
    
    obj.states = [('y','y'), ('y','n'), ('n','y'), ('n','n')]
    obj.states_payoffs[('y','y')] = {0:{1:[('B',1), 2], 2:[('B',2), 1]},
                                     1:{1:[('B',1), 0], 2:[('S',2), 0]},
                                     2:{1:[('S',1), 0], 2:[('B',2), 0]},
                                     3:{1:[('S',1), 1], 2:[('S',2), 2]}}
    obj.states_payoffs[('y','n')] = {0:{1:[('B',1), 2], 2:[('B',2), 0]},
                                     1:{1:[('B',1), 0], 2:[('S',2), 2]},
                                     2:{1:[('S',1), 0], 2:[('B',2), 1]},
                                     3:{1:[('S',1), 1], 2:[('S',2), 0]}}
    obj.states_payoffs[('n','y')] = {0:{1:[('B',1), 0], 2:[('B',2), 1]},
                                     1:{1:[('B',1), 2], 2:[('S',2), 0]},
                                     2:{1:[('S',1), 1], 2:[('B',2), 0]},
                                     3:{1:[('S',1), 0], 2:[('S',2), 2]}}
    obj.states_payoffs[('n','n')] = {0:{1:[('B',1), 0], 2:[('B',2), 0]},
                                     1:{1:[('B',1), 2], 2:[('S',2), 2]},
                                     2:{1:[('S',1), 1], 2:[('B',2), 1]},
                                     3:{1:[('S',1), 0], 2:[('S',2), 0]}}
    obj.compute_bayesian_payoffs()
    obj.build_bayesian_game(save_model='', suffix='testc')
    ### Game info needed to pass to the solver
    game_info = [obj.bayesian_game, obj.bayesian_players]
    action_list = {}
    for b in obj.bayesian_players:
        action_list[b] = copy.deepcopy(obj.actions[b[1]])
        for idx, ac in enumerate(action_list[b]):
            if ac[0] in ['B', 'S']:
                action_list[b][idx] = (ac[0], b)
    game_info.append(action_list)
    obj.solve_game(method='enumerate_pure', print_to_cmd=True, game_info=game_info)
    
    solutions = []
    for idx, val in obj.solution.sol.items():
        solutions.append([])
        for x in val['solution combination'][0]:
            solutions[-1].append(x[0])
    return solutions

print('###Building the Bayesian gane \'BoS\'###')
solutions = bayesian_game_BoS()
if (['B','B','B','S'] in solutions) and (['S','B','S','S'] in solutions) and len(solutions)==2:
    print('The Bayesian game \'BoS\' was built and solved correctly')
else:
    print('The Bayesian game \'BoS\' was Not built or solved correctly')
