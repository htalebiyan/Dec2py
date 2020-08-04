'''
This modul contains functions that plot the results of game theoretical restoration
planning of interdepndent networks
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_ne_sol_2player(game, suffix=''):
    '''
    This function plot the payoff functions with nath equilibria amd optimal solution
    marked on it (currently for 2-player games)

    Parameters
    ----------
    game : NormalGame object
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    payoff_dict_cols = ['P'+str(l)+' actions' for l in game.players]
    payoff_dict_cols += ['P'+str(l)+' payoff' for l in game.players]
    payoff_dict = pd.DataFrame(columns=payoff_dict_cols)
    for _, ac in game.payoffs.items():
        acts = []
        pays = []
        for _, l in ac.items():
            acts.append(l[0])
            pays.append(l[1])
        payoff_dict = payoff_dict.append(dict(zip(payoff_dict_cols, acts+pays)),
                                         ignore_index=True)
    dpi = 300
    _, axs = plt.subplots(2, 1, sharex=True, figsize=(2000/dpi, 3000/dpi))
    for idxl, l in enumerate(game.players):
        pivot_dict = payoff_dict.pivot(index='P'+str(2)+' actions',
                                       columns='P'+str(1)+' actions',
                                       values='P'+str(l)+' payoff')
        sns.heatmap(pivot_dict, annot=False, linewidths=.2, cmap="YlGnBu", ax=axs[idxl])
        axs[idxl].set_title('Player %d\'s payoffs'%l)
        for _, val in game.solution.sol.items():
            p1_act_idx = list(pivot_dict).index(val['P1 actions'])
            p2_act_idx = list(pivot_dict.index.values).index(val['P2 actions'])
            axs[idxl].add_patch(Rectangle((p1_act_idx, p2_act_idx), 1, 1,
                                          fill=False, edgecolor='red', lw=2))
            
        if game.chosen_equilibrium:
            p1_cne_idx = list(pivot_dict).index(game.chosen_equilibrium['P1 actions'])
            p2_cne_idx = list(pivot_dict.index.values).index(game.chosen_equilibrium['P2 actions'])
            axs[idxl].add_patch(Rectangle((p1_cne_idx, p2_cne_idx), 1, 1, fill=False,
                                          hatch='///', edgecolor='red', lw=0.1))
        if game.optimal_solution:
            try:
                p1_opt_idx = list(pivot_dict).index(game.optimal_solution['P1 actions'])
                p2_opt_idx = list(pivot_dict.index.values).index(game.optimal_solution['P2 actions'])
                axs[idxl].add_patch(Rectangle((p1_opt_idx, p2_opt_idx), 1, 1, fill=False,
                                              hatch='xxx', edgecolor='green', lw=0.1))
            except:
                print('Optimal solution is not among the actions:')
                print(game.optimal_solution)
        else:
            print('Optimal solution has not been calculated')
    plt.savefig('NE_sol_2D_'+suffix+'.png', dpi=dpi, bbox_inches='tight')
