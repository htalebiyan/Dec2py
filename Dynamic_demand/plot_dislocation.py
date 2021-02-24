import pickle
import seaborn as sns    
import pandas as pd
import matplotlib.pyplot as plt
sns.set(context='notebook', style='darkgrid', font_scale=1.2)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.close('all')
file_dir = 'C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/Node_arc_info/'
with open(file_dir+'seaside_pop_dislocation_demands_500yr.pkl', 'rb') as f:
    dynamic_params = pickle.load(f)

net_name = {1:'Water', 3:'Power'}

dpi=300
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(4000/dpi, 1500/dpi))
for idx, l in enumerate([1,3]):
    ax=axs[idx]
    demand_data = dynamic_params[l]
    demand_data['temp'] = demand_data['current pop']/(demand_data['total pop']+1e-8)
    demand_data = demand_data.apply(pd.to_numeric)
    # ax = sns.lineplot(x="time", y="temp", hue="node", data=demand_data, ax=ax, lw=1)
    flierprops = dict(markersize=3, linestyle='none')
    ax = sns.boxplot(x="time", y="temp", data=demand_data, ax=ax, linewidth=1,
                     flierprops=flierprops, palette=sns.cubehelix_palette(20))
    ax.set_title(net_name[l])
    ax.set(xlabel=r'time (weeks)', ylabel= '\% pre-event demand')
plt.savefig('dynamic_demand_plot.png', dpi=dpi, bbox_inches='tight')