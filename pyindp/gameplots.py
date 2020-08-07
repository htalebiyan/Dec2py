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
    This function plot the payoff functions of a normal game for one time step
    with nash equilibria and optimal solution marked on it (currently for 2-player games)

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
        sns.heatmap(pivot_dict, annot=False, linewidths=.2, cmap="Blues_r", ax=axs[idxl])
        axs[idxl].set_title('Player %d\'s payoffs, $R_c=$%d'%(l,game.v_r[l]))
        for _, val in game.solution.sol.items():
            for act in val['solution combination']:
                p1_act_idx = list(pivot_dict).index(act[0])
                p2_act_idx = list(pivot_dict.index.values).index(act[1])
                edgeclr = 'red'
                line_width = 2
                zor = 2
                if len(val['solution combination']) > 1:
                    edgeclr = 'gold'
                    line_width = 5
                    zor = 1
                axs[idxl].add_patch(Rectangle((p1_act_idx, p2_act_idx), 1, 1,
                                              fill=False, edgecolor=edgeclr,
                                              lw=line_width, zorder=zor))
            
        if game.chosen_equilibrium:
            mixed_profile = game.chosen_equilibrium['chosen mixed profile action']
            act = game.chosen_equilibrium['solution combination'][mixed_profile]
            p1_cne_idx = list(pivot_dict).index(act[0])
            p2_cne_idx = list(pivot_dict.index.values).index(act[1])
            edgeclr = 'red'
            if len(game.chosen_equilibrium['solution combination']) > 1:
                edgeclr = 'gold'
                axs[idxl].set_hatch_color = edgeclr
            axs[idxl].add_patch(Rectangle((p1_cne_idx, p2_cne_idx), 1, 1, fill=False,
                                          hatch='o', edgecolor=edgeclr, lw=0.01))
            
        if game.optimal_solution:
            try:
                p1_opt_idx = list(pivot_dict).index(game.optimal_solution['P1 actions'])
                p2_opt_idx = list(pivot_dict.index.values).index(game.optimal_solution['P2 actions'])
                axs[idxl].add_patch(Rectangle((p1_opt_idx, p2_opt_idx), 1, 1, fill=False,
                                              hatch='xxx', edgecolor='green', lw=0.01))
            except:
                print('Optimal solution is not among the actions:')
                print(game.optimal_solution)
        else:
            print('Optimal solution has not been calculated')
    plt.savefig('NE_sol_2D_'+suffix+'.png', dpi=dpi, bbox_inches='tight')
