import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import scikit_posthocs as sp

sns.set(style='darkgrid', font_scale=0.8)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.close('all')

def plot_radar(raw_data,row_titles,col_titles, suffix):
    # initialize the figure
    my_dpi=300
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    # Initialise the spider plot
    ax = plt.subplot(1,1,1, polar=True)
    # Create a color palette:
    my_palette = ['darkorange','forestgreen','crimson','midnightblue','c','k']
    #correct labels
    labels = [x for x in row_titles]
    labels = [r'$R_c$' if x == 'Rc' else x for x in labels]
    labels = [r'$P_d$' if x == 'pd' else x for x in labels]
    labels = [r'$P_i$' if x == 'pi' else x for x in labels]
    labels = [r'$\Upsilon$' if x == 'gamma' else x for x in labels]
    means = raw_data.mean(axis=1)
    # Loop to plot
    for row in range(len(raw_data.index)):
        title=row_titles[row]
        labels[row] = labels[row]+'(%1.2f)'%means[title]
        color=my_palette[row]
        # number of variable
        categories=list(raw_data)
        N = len(categories)
         
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
         
        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
         
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], col_titles, color='k', y=-.03)
        # Draw ylabels
        ax.set_rlabel_position(10)
        plt.yticks([1,2,3,4,5,6], ["1","2","3","4","5","6"], color="k")
        plt.ylim(1,7)
        ax.invert_yaxis()
         
        values=raw_data.iloc[row].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=0.4, linestyle='solid', zorder=6)
        ax.fill(angles, values, color=color, alpha=0.2, lw=0.01,zorder=6)
    # Add legend and save to file
    leg = plt.figlegend(ax.lines,labels,loc='lower center', bbox_to_anchor=(0.6, 1.05),
                        ncol=3, fontsize='medium',frameon=False)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)
    plt.savefig('SensRadar_'+suffix+'.png', dpi=my_dpi, bbox_inches="tight",bbox_extra_artists=(leg, ))
    
'''Plot results'''
[corr, sens_perf, sens_res, delta_dict_perf, delta_dict_res,anova_perf, anova_res] = pd.read_pickle('postprocess_dicts_sens_synth.pkl')

'''Radar plot'''
sens_perf["Topology/Resource_allocation"] = sens_perf["auction"]+'\n'+sens_perf["topology"]
sens_perf = sens_perf.pivot_table(values='rank_corrected', index='config_param', columns='Topology/Resource_allocation')
sens_perf.reset_index()
plot_radar(sens_perf,sens_perf.index.values,sens_perf.columns, suffix='performance')
print(sens_perf.mean(axis=1))

sens_res["Topology/Resource_allocation"] = sens_res["auction"]+'\n'+sens_res["topology"]
sens_res = sens_res.pivot_table(values='rank_corrected', index='config_param', columns='Topology/Resource_allocation')
sens_res.reset_index()
plot_radar(sens_res,sens_res.index.values,sens_res.columns, suffix='allocation')
print(sens_res.mean(axis=1))
'''Correlation plot'''
sns.set(font_scale=1.2)
dpi = 300
corr['Topology/Resource allocation']= corr['auction_type']+'-'+corr['topology']
corr['config param'] = corr['config_param'].replace({' No. Layers': r'$L$',
                                                    ' No. Nodes': r'$N$',
                                                    ' Topology Parameter': r'$\Upsilon$',
                                                    ' Interconnection Prob': r'$P_i$',
                                                    ' Damage Prob': r'$P_d$',
                                                    ' Resource Cap': r'$R_c$'})
for y in corr.y.unique():
    corr_fig = corr[(corr['y']==y)&(corr['pearson_corr']!='nan')]
    corr_fig = corr_fig.pivot_table(values='pearson_corr',
                                    index='Topology/Resource allocation',
                                    columns='config param')
    plt.figure(figsize=[1500/dpi,1000/dpi])
    ax = sns.heatmap(corr_fig, annot=False, fmt="1.2f", vmin=-1, vmax=1,
                      cmap="RdYlGn")
    plt.savefig('corr_'+y+'.png', dpi=dpi, bbox_inches='tight')
'''Line plot'''
# # plt.figure()
# # ax = sns.lineplot(x="Topology/Resource_allocation", y="rank",
# #                   hue="config_param", style="config_param", lw=5, ms=12,
# #                   markers=True, dashes=False, data=sens_perf)
# ax.invert_yaxis()
'''Bar plot'''
# plt.figure()
# c = 1
# for row in sens_perf.topology.unique():
#     for col in sens_perf.auction.unique():
#         plt.subplot(3, 4, c)
#         c += 1
#         data = sens_perf[(sens_perf['topology']==row)&(sens_perf['auction']==col)]
#         plt.bar(x=data['config_param'],height=data['delta'],yerr=data['delta_CI'], capsize=6)
#         plt.title(data["Topology/Resource_allocation"].unique())
#         xmin, xmax, ymin, ymax = plt.axis()
#         plt.ylim(0.05,ymax)
'''Violin plot'''
# plt.figure()
# c = 1
# for name, val in delta_dict_res.items():
#     plt.subplot(3, 4, c)
#     c += 1
#     df = pd.melt(val, id_vars=[], value_vars=val.columns)
#     sns.violinplot(x="variable", y="value", data=df)
#     plt.title(name)
#     xmin, xmax, ymin, ymax = plt.axis()
#     plt.ylim(0.05,ymax)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
#                     wspace=0.35)
# plt.figure()
# c = 1
# for name, val in anova_res.items():
#     plt.subplot(3, 4, c)
#     plt.title(name)
#     c += 1
#     df = val['posthoc_matrix']
#     heatmap_args = {'linewidths': 0.1, 'linecolor': '0.5', 'clip_on': False, 
#                     'square': True, 'cbar_ax_bbox': [0.01, 0.35, 0.04, 0.3]}
#     sp.sign_plot(df, **heatmap_args)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
#                     wspace=0.35)