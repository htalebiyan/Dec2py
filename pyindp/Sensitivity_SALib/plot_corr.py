import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(context='notebook', style='darkgrid', font_scale=1.2)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.close('all')

dpi = 300
data = pd.read_csv('corr.csv', header=0)
data.index = data['topology']+'-'+data['resource allocation']
data = data.rename(columns={'L': r'$L$', 'N': r'$N$', 'gamma': r'$\Upsilon$', 'Pi': r'$P_i$',
                   'Pd': r'$P_d$', 'Rc': r'$R_c$'})
for y in data.y.unique():
    data_fig = data[data['y']==y].drop(['y', 'topology','resource allocation'], axis=1)
    plt.figure(figsize=[1500/dpi,1000/dpi])
    ax = sns.heatmap(data_fig, annot=False, fmt="1.2f", vmin=-.75, vmax=.75,
                     cmap="RdYlGn")#.transpose()
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=0 )
    # plt.setp(ax.yaxis.get_majorticklabels(), rotation=0 )
    plt.savefig('corr_'+y+'.png', dpi=dpi, bbox_inches='tight')