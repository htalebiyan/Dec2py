import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import scikit_posthocs as sp

sns.set(style='darkgrid', font_scale=1)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.close('all')


def plot_radar(raw_data, row_titles, col_titles, suffix):
    # initialize the figure
    my_dpi = 300
    plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    # Initialise the spider plot
    ax = plt.subplot(1, 1, 1, polar=True)
    # Create a color palette:
    my_palette = ['darkorange', 'forestgreen', 'crimson', 'midnightblue', 'c', 'k']
    # correct labels
    labels = [x for x in row_titles]
    labels = [r'$R_c$' if x == 'Rc' else x for x in labels]
    labels = [r'$P_d$' if x == 'pd' else x for x in labels]
    labels = [r'$P_i$' if x == 'pi' else x for x in labels]
    # labels = [r'$\Upsilon$' if x == 'gamma' else x for x in labels]
    means = raw_data.mean(axis=1)
    # Loop to plot
    for row in range(len(raw_data.index)):
        title = row_titles[row]
        labels[row] = labels[row] + '(%1.2f)' % means[title]
        color = my_palette[row]
        # number of variable
        categories = list(raw_data)
        N = len(categories)

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], col_titles, color='k', y=-.2)
        # Draw ylabels
        ax.set_rlabel_position(10)
        plt.yticks([1, 2, 3, 4], ["1", "2", "3", "4"], color="k")
        plt.ylim(1, 5)
        ax.invert_yaxis()

        values = raw_data.iloc[row].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=0.4, linestyle='solid', zorder=6)
        ax.fill(angles, values, color=color, alpha=0.2, lw=0.01, zorder=6)
    # Add legend and save to file
    leg = plt.figlegend(ax.lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05),
                        ncol=2, fontsize='medium', frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)
    plt.savefig('SensRadar_' + suffix + '.png', dpi=my_dpi, bbox_inches="tight", bbox_extra_artists=(leg,))


def correct_legend_labels(labels):
    labels = ['N-INRG' if x == 'ng' else x for x in labels]
    labels = ['B-INRG-All cu' if x == 'bgCCCCUUUU' else x for x in labels]
    labels = ['B-INRG-All cu ex. Power' if x == 'bgCCNCUUUU' else x for x in labels]
    labels = ['B-INRG-All nu' if x == 'bgNNNNUUUU' else x for x in labels]
    labels = ['B-INRG-nn' if x == 'bgNNUU' else x for x in labels]
    labels = ['B-INRG-cn' if x == 'bgCNUU' else x for x in labels]
    labels = ['B-INRG-nc' if x == 'bgNCUU' else x for x in labels]
    labels = ['B-INRG-cc' if x == 'bgCCUU' else x for x in labels]
    labels = ['Rationality' if x == 'rationality' else x for x in labels]
    labels = ['br_level' if x == 'bounded rationality level' else x for x in labels]
    return labels


'''Plot results'''
[corr, sens_perf, delta_dict_perf, anova_perf] = pd.read_pickle('postprocess_dicts_sens_synth.pkl')

for idx, r in sens_perf.iterrows():
    sens_perf.loc[idx, "Decision/Res. Alloc."] = correct_legend_labels([r["decision"]])[0] + '\n' + r["auction"]
for idx, r in corr.iterrows():
    corr.loc[idx, "Decision/Res. Alloc."] = correct_legend_labels([r["decision_type"]])[0] + '-' + r["auction_type"]

'''Radar plot'''
# sens_perf_pivot = sens_perf.pivot_table(values='rank_corrected', index='config_param', columns='Decision/Res. Alloc.')
# sens_perf_pivot.reset_index()
# plot_radar(sens_perf_pivot, sens_perf_pivot.index.values, sens_perf_pivot.columns, suffix='performance')
# print(sens_perf_pivot.mean(axis=1))

'''Correlation plot'''
# sns.set(font_scale=1.2)
# dpi = 300
# corr['config param'] = corr['config_param'].replace({' No. Nodes': r'$N$', ' Interconnection Prob': r'$P_i$',
#                                                      ' Damage Prob': r'$P_d$', ' Resource Cap ': r'$R_c$'})
# for y in corr.y.unique():
#     corr_fig = corr[(corr['y'] == y) & (corr['pearson_corr'] != 'nan')]
#     corr_fig = corr_fig.pivot_table(values='pearson_corr', index='Decision/Res. Alloc.', columns='config param')
#     plt.figure(figsize=[1500 / dpi, 1000 / dpi])
#     ax = sns.heatmap(corr_fig, annot=False, fmt="1.2f", vmin=-1, vmax=1,
#                      cmap="RdYlGn")
#     plt.savefig('corr_' + y + '.png', dpi=dpi, bbox_inches='tight')

'''Line plot'''
# plt.figure()
# sns.lineplot(x="Decision/Res. Alloc.", y="rank", hue="config_param", style="config_param", lw=5, ms=12,
#              markers=True, dashes=False, data=sens_perf)
# plt.show()

'''anova bar plot'''
# plt.figure()
# c = 1
# for row in sens_perf.decision.unique():
#     for col in sens_perf.auction.unique():
#         plt.subplot(3, 4, c)
#         c += 1
#         data = sens_perf[(sens_perf['decision'] == row) & (sens_perf['auction'] == col)]
#         plt.bar(x=data['config_param'], height=data['delta'], yerr=data['delta_CI'], capsize=6)
#         plt.title(data["Decision/Res. Alloc."].unique())
#         xmin, xmax, ymin, ymax = plt.axis()
#         plt.ylim(0.05, ymax)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
#                     wspace=0.35)

''' posthoc plot'''
# plt.figure()
# c = 1
# for name, val in delta_dict_perf.items():
#     plt.subplot(3, 4, c)
#     c += 1
#     df = pd.melt(val, id_vars=[], value_vars=val.columns)
#     sns.violinplot(x="variable", y="value", data=df)
#     plt.title(name)
#     xmin, xmax, ymin, ymax = plt.axis()
#     plt.ylim(0.05, ymax)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
#                     wspace=0.35)
# plt.show()
# plt.figure()
# c = 1
# for name, val in anova_perf.items():
#     plt.subplot(3, 4, c)
#     plt.title(name)
#     c += 1
#     df = val['posthoc_matrix']
#     heatmap_args = {'linewidths': 0.1, 'linecolor': '0.5', 'clip_on': False,
#                     'square': True, 'cbar_ax_bbox': [0.01, 0.35, 0.04, 0.3]}
#     sp.sign_plot(df, **heatmap_args)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
#                     wspace=0.35)

''' Layer Topology '''
config_list_folder = 'C:/Users/ht20/Documents/Files/Generated_Network_Dataset_v4.1/GeneralNetworks/'
config_data = pd.read_csv(config_list_folder + 'List_of_Configurations.txt', header=0, sep="\t")
config_data = config_data.assign(topology='general')

results_folder = 'C:/Users/ht20/Documents/Files/Game_synthetic/v4.1/postprocess/'
dfs = pd.read_pickle(results_folder + 'postprocess_dicts_EDM10.pkl')
comp_df = pd.merge(dfs[4], config_data, left_on=['Magnitude'], right_on=['Config Number'])
comp_df['lambda_U'] = pd.to_numeric(comp_df['lambda_U'], errors='coerce')
for idx, row in comp_df.iterrows():
    params = comp_df.loc[idx, ' Topology Parameter'].split(',')
    if row['layer'] == 1:
        comp_df.loc[idx, 'Topo.'] = comp_df.loc[idx, ' Net Types'][2]
        comp_df.loc[idx, 'Topo. Param'] = float(params[0][1:])
    elif row['layer'] == 2:
        comp_df.loc[idx, 'Topo.'] = comp_df.loc[idx, ' Net Types'][-3]
        comp_df.loc[idx, 'Topo. Param'] = float(params[1][:-1])

layer_topo = comp_df[~pd.isnull(comp_df['Topo.']) & ~pd.isnull(comp_df['lambda_U'])]
layer_topo = layer_topo.replace({'r': 'Random', 's': 'Scale Free', 'g': 'Grid', 't': 'Tree', 'm': 'MPG'})
layer_topo = layer_topo.replace({'ng': 'N-INRG', 'bgNNUU': 'B-INRG-nn', 'bgCNUU': 'B-INRG-cn',
                                 'bgNCUU': 'B-INRG-nc', 'bgCCUU': 'B-INRG-cc'})
layer_topo = layer_topo.rename(columns={'lambda_U': 'lambda U', 'decision_type': 'decision type'})
fig_df = layer_topo[(layer_topo['lambda U'] > -20) & (layer_topo['rationality'] != 'unbounded')]
# & (layer_topo['auction_type'] == 'UNIFORM')

my_dpi = 300
plt.figure(figsize=(1600 / my_dpi, 1600 / my_dpi), dpi=my_dpi)
pal = sns.color_palette('Set1')
with pal:
    ax = sns.boxplot(x="decision type", y="lambda U", hue="Topo.", data=fig_df,
                     fliersize=.1, showfliers=True, linewidth=.5)
    ax.set_ylabel(r'$\lambda_U$ of a single layer')
    ax.set_xlabel('Decision Type')
    ax.set_ylim(-3, 0.5)
plt.savefig('topo_sens.png', dpi=my_dpi, bbox_inches="tight")

# with pal:
#     g = sns.lmplot(x=' Resource Cap ', y='lambda U', hue="decision type", col='Topo.', data=fig_df,
#                    lowess=True, scatter_kws={"s": 3}, sharex=False, height=6, aspect=1,
#                    scatter=True)  #, col='Topo.'
#     for ax in g.axes.flat:
#         ax.set_ylabel(r'$\lambda_U$')
#         ax.set_xlabel(r'$R_c$')
#     g.set(ylim=(-2.5, 0))
