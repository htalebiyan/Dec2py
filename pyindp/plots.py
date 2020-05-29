'''plots'''
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
sns.set(context='notebook', style='darkgrid')
# plt.rc('text', usetex=True)
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

def plot_performance_curves_shelby(df, x='t', y='cost', cost_type='Total',
                                   decision_type=None, judgment_type=None,
                                   auction_type=None, valuation_type=None,
                                   ci=None, normalize=False, deaggregate=False,
                                   plot_resilience=False):
    '''

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    x : TYPE, optional
        DESCRIPTION. The default is 't'.
    y : TYPE, optional
        DESCRIPTION. The default is 'cost'.
    cost_type : TYPE, optional
        DESCRIPTION. The default is 'Total'.
    decision_type : TYPE
        DESCRIPTION.
    auction_type : TYPE, optional
        DESCRIPTION. The default is None.
    valuation_type : TYPE, optional
        DESCRIPTION. The default is None.
    ci : TYPE, optional
        DESCRIPTION. The default is None.
    normalize : TYPE, optional
        DESCRIPTION. The default is False.
    deaggregate : TYPE, optional
        DESCRIPTION. The default is False.
    plot_resilience : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    #: Make lists
    no_resources = df.no_resources.unique().tolist()
    if not decision_type:
        decision_type = df.decision_type.unique().tolist()
    if not judgment_type:
        judgment_type = df.judgment_type.unique().tolist()
    if not auction_type:
        auction_type = df.auction_type.unique().tolist()
    if not valuation_type:
        valuation_type = df.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    T = len(df[x].unique().tolist())
    #sns.color_palette("husl", len(auction_type)+1)
    row_plot = [valuation_type, 'valuation_type']
    col_plot = [no_resources, 'no_resources']
    hue_type = [auction_type, 'auction_type']
    style_type = 'decision_type'
    # Initialize plot properties
    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True, sharey=True,
                            figsize=(4000/dpi, 2500/dpi))
    colors = ['#154352', '#007268', '#5d9c51', '#dbb539', 'k']
    pal = sns.color_palette(colors[:len(hue_type[0])])
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            cost_data = df[(df.cost_type == cost_type)&
                           (df.decision_type.isin(decision_type))&
                           (df[col_plot[1]] == val_c)&
                           ((df[row_plot[1]] == val_r)|(df[row_plot[1]] == 'nan'))]
            sns.lineplot(x=x, y=y, hue=hue_type[1], style=style_type, markers=True, ci=ci,
                         ax=ax, palette=pal, data=cost_data[cost_data.layer == 'nan'],
                         **{'markersize':5})
            if deaggregate:
                cost_lyr = cost_data[cost_data.layer != 'nan']
                cost_lyr_pivot = cost_lyr.pivot_table(values='cost',
                    index=cost_lyr.columns.drop(['layer', 'cost', 'normalized_cost']).tolist(),
                    columns='layer')
                cost_lyr_pivot.reset_index(inplace=True)
                layers = cost_lyr['layer'].unique().tolist()
                for l in layers:
                    if l != 1:
                        cost_lyr_pivot[l] += cost_lyr_pivot[l-1]
                layers.sort(reverse=True)
                for l in layers:
                    pal_adj = sns.color_palette([[xx*l/len(layers) for xx in x] for x in pal])
                    sns.barplot(x=x, y=l, hue=hue_type[1], ax=ax, linewidth=0.5, ci=None,
                                data=cost_lyr_pivot, hue_order=hue_type[0],
                                palette=pal_adj, **{'alpha':0.35})
            ax.set(xlabel=r'time step $t$', ylabel=cost_type+' Cost')
            ax.get_legend().set_visible(False)
            ax.xaxis.set_ticks(np.arange(0, T, 1.0))#ax.get_xlim()
            if plot_resilience:
                resilience_data = df[(df.cost_type == 'Under Supply Perc')&
                                     (df.decision_type.isin(decision_type))&
                                     (df[col_plot[1]] == val_c)&
                                     ((df[row_plot[1]] == val_r)|(df[row_plot[1]] == 'nan'))]
                divider = make_axes_locatable(ax)
                ax_2 = divider.append_axes("bottom", size="100%", pad=0.12, sharex=ax)
                sns.lineplot(x=x, y=y, hue=hue_type[1], style=style_type, markers=True, ci=ci,
                             ax=ax_2, legend='full', palette=pal,
                             data=resilience_data[resilience_data.layer == 'nan'])
                if deaggregate:
                    cost_lyr = resilience_data[resilience_data.layer != 'nan']
                    cost_lyr_pivot = cost_lyr.pivot_table(values='cost',
                        index=cost_lyr.columns.drop(['layer', 'cost', 'normalized_cost']).tolist(),
                        columns='layer')
                    cost_lyr_pivot.reset_index(inplace=True)
                    layers = cost_lyr['layer'].unique().tolist()
                    for l in layers:
                        if l != 1:
                            cost_lyr_pivot[l] += cost_lyr_pivot[l-1]
                    layers.sort(reverse=True)
                    for l in layers:
                        pal_adj = sns.color_palette([[xx*l/len(layers) for xx in x] for x in pal])
                        sns.barplot(x=x, y=l, hue=hue_type[1], ax=ax_2, linewidth=0.5,
                                    ci=None, data=cost_lyr_pivot, hue_order=hue_type[0],
                                    palette=pal_adj, **{'alpha':0.35})
                ax_2.set(ylabel='% Unmet Demand')
                ax_2.get_legend().set_visible(False)
                if idx_c != 0.0:
                    ax_2.set_ylabel('')
                    ax_2.set_yticklabels([])
                ax.xaxis.set_ticks([])
                ax_2.xaxis.set_ticks(np.arange(0, T, 1.0))#ax.get_xlim()
    # Rebuild legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [x for x in handles if isinstance(x, mplt.lines.Line2D)]
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.35)
    # Add overll x- and y-axis titles
    _, axs_c, axs_r = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'Total resources=%d'%(col_plot[0][idx]))
    for idx, ax in enumerate(axs_r):
        ax.annotate('Valuation: '+row_plot[0][idx], xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 4, 0), xycoords=ax.yaxis.label,
                    textcoords='offset points', ha='right', va='center', rotation=90)
    plt.savefig('Performance_curves.png', dpi=dpi)

def plot_performance_curves_synthetic(df, x='t', y='cost', cost_type='Total', ci=None):
    '''
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    x : TYPE, optional
        DESCRIPTION. The default is 't'.
    y : TYPE, optional
        DESCRIPTION. The default is 'cost'.
    cost_type : TYPE, optional
        DESCRIPTION. The default is 'Total'.
    ci : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    no_resources = df.no_resources.unique().tolist()
    auction_type = df.auction_type.unique().tolist()
    auction_type.remove('')
    valuation_type = df.valuation_type.unique().tolist()
    valuation_type.remove('')
    T = len(df[x].unique().tolist())
    hor_grid = [0]
    ver_grid = valuation_type
    fig, axs = plt.subplots(len(ver_grid), len(hor_grid), sharex=True, sharey=True,
                            tight_layout=False)
    for idx_c, _ in enumerate(hor_grid):
        for idx_r, vt in enumerate(ver_grid):
            ax, _, _ = find_ax(axs, hor_grid, ver_grid, idx_r, idx_c)
            selected_data = df[(df['cost_type'] == cost_type)&
                               ((df['valuation_type'] == vt)|\
                                (df['valuation_type'] == ''))]
            selected_data2 = df[(df['cost_type'] == 'Under Supply Perc')&
                                ((df['valuation_type'] == vt)|\
                                 (df['valuation_type'] == ''))]
            with sns.xkcd_palette(['black', "windows blue", 'red', "green"]):
                ax = sns.lineplot(x=x, y=y, hue="auction_type", style='decision_type',
                                  markers=False, ci=ci, ax=ax, legend='full',
                                  data=selected_data)
                ax_2 = ax.twinx()
                ax_2 = sns.lineplot(x=x, y=y, hue="auction_type", style='decision_type',
                                    markers=False, ci=ci, ax=ax_2, legend='full',
                                    data=selected_data2)
                ax.set(xlabel=r'time step $t$', ylabel=cost_type+' Cost')
                if cost_type == 'Under Supply Perc':
                    ax.set(xlabel=r'time step $t$', ylabel='Unmet Demand Ratio')
                ax.get_legend().set_visible(False)
                ax.xaxis.set_ticks(np.arange(0, T+1, 1.0))   #ax.get_xlim()
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5)
    if len(ver_grid) == 1 and len(hor_grid) == 1:
        axs_c = [axs]
        axs_r = [axs]
    elif len(hor_grid) == 1:
        axs_r = [axs[0]]
        axs_c = axs
    elif len(no_resources) == 1:
        axs_r = axs
        axs_c = [axs[0]]
    else:
        axs_c = axs[0, :]
        axs_r = axs[:, 0]
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'%d'%(hor_grid[idx]))
    for idx, ax in enumerate(axs_r):
        ax.annotate('Valuation = '+ver_grid[idx], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points', ha='right',
                    va='center', rotation=90)
    plt.savefig('Performance_curves_'+cost_type+'.pdf', dpi=600)

def plot_relative_performance_shelby(lambda_df, cost_type='Total', lambda_type='U'):
    '''
    Parameters
    ----------
    lambda_df : TYPE
        DESCRIPTION.
    cost_type : TYPE, optional
        DESCRIPTION. The default is 'Total'.
    lambda_type : TYPE, optional
        DESCRIPTION. The default is 'U'.

    Returns
    -------
    None.

    '''
    #: Make lists
    # no_resources = lambda_df.no_resources.unique().tolist()
    # decision_type = lambda_df.decision_type.unique().tolist()
    judgment_type = lambda_df.judgment_type.unique().tolist()
    auction_type = lambda_df.auction_type.unique().tolist()
    if 'nan' in auction_type:
        auction_type.remove('nan')
    valuation_type = lambda_df.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    row_plot = [valuation_type, 'valuation_type']
    col_plot = [auction_type, 'auction_type']
    hue_type = [judgment_type, 'judgment_type']
    # Initialize plot properties
    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                            sharey=True, figsize=(4000/dpi, 2500/dpi))
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            selected_data = lambda_df[(lambda_df.cost_type == cost_type)&
                                      (lambda_df['lambda_'+lambda_type] != 'nan')&
                                      ((lambda_df[col_plot[1]] == val_c)|\
                                       (lambda_df[col_plot[1]] == 'nan'))&\
                                      ((lambda_df[row_plot[1]] == val_r)|\
                                       (lambda_df[row_plot[1]] == 'nan'))]
            with sns.color_palette("RdYlGn", 8):  #sns.color_palette("YlOrRd", 7)
                sns.barplot(x='no_resources', y='lambda_'+lambda_type,
                            hue=hue_type[1], data=selected_data, linewidth=0.5,
                            edgecolor=[.25, .25, .25], capsize=.05,
                            errcolor=[.25, .25, .25], errwidth=1, ax=ax)
                ax.get_legend().set_visible(False)
                ax.grid(which='major', axis='y', color=[.75, .75, .75], linewidth=.75)
                ax.set_xlabel(r'\# resources')
                if idx_r != len(valuation_type)-1:
                    ax.set_xlabel('')
                ax.set_ylabel(r'E[$\lambda_{%s}$], Valuation: %s'%(lambda_type, row_plot[0][idx_r]))
                if idx_c != 0:
                    ax.set_ylabel('')
                ax.xaxis.set_label_position('bottom')
                ax.set_facecolor('w')
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='upper right', frameon=True, framealpha=0.5, ncol=1)
    _, axs_c, _ = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'Res. Alloc.: %s'%(auction_type[idx]))
    plt.savefig('Relative_perforamnce.png', dpi=dpi)

def plot_relative_performance_synthetic(lambda_df, cost_type='Total', lambda_type='U'):
    '''
    Parameters
    ----------
    lambda_df : TYPE
        DESCRIPTION.
    cost_type : TYPE, optional
        DESCRIPTION. The default is 'Total'.
    lambda_type : TYPE, optional
        DESCRIPTION. The default is 'U'.

    Returns
    -------
    None.

    '''
#    auction_type = lambda_df.auction_type.unique().tolist()
#    auction_type.remove('')
    valuation_type = lambda_df.valuation_type.unique().tolist()
    valuation_type.remove('')
    fig, axs = plt.subplots(len(valuation_type), 1, sharex=True, sharey='row', tight_layout=False)
    for idxvt, vt in enumerate(valuation_type):
        ax = axs[idxvt]
        selected_data = lambda_df[(lambda_df['cost_type'] == cost_type)&\
                                  (lambda_df['lambda_'+lambda_type] != 'nan')&\
                                  ((lambda_df['valuation_type'] == vt)|\
                                   (lambda_df['valuation_type'] == ''))]
        with sns.color_palette("RdYlGn", 8):  #sns.color_palette("YlOrRd", 7)
            sns.barplot(x='auction_type', y='lambda_'+lambda_type, hue="decision_type",
                        data=selected_data, linewidth=0.5, edgecolor=[.25, .25, .25],
                        capsize=.05, errcolor=[.25, .25, .25], errwidth=1, ax=ax)
            ax.get_legend().set_visible(False)
            ax.grid(which='major', axis='y', color=[.75, .75, .75], linewidth=.75)
            ax.set_xlabel(r'Auction Type')
            if idxvt != len(valuation_type)-1:
                ax.set_xlabel('')
            ax.set_ylabel(r'E[$\lambda_{%s}$], Valuation Type = %s'%(lambda_type, vt))
            ax.xaxis.set_label_position('top')
            ax.set_facecolor('w')
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='upper right', frameon=True, framealpha=0.5, ncol=1)
    plt.savefig('Relative_perforamnce.pdf', dpi=600)

def plot_auction_allocation_shelby(df_res, ci=None):
    '''
    Parameters
    ----------
    df_res : TYPE
        DESCRIPTION.
    ci : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    #: Make lists
    no_resources = df_res.no_resources.unique().tolist()
    layer = df_res.layer.unique().tolist()
    decision_type = df_res.decision_type.unique().tolist()
    # judgment_type = df_res.judgment_type.unique().tolist()
    auction_type = df_res.auction_type.unique().tolist()
    if 'nan' in auction_type:
        auction_type.remove('nan')
    valuation_type = df_res.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    figs = [valuation_type, 'valuation_type']
    row_plot = [layer, 'layer']
    col_plot = [no_resources, 'no_resources']
    hue_type = [auction_type, 'auction_type']
    style_type = [decision_type, 'decision_type']
    # Initialize plot properties
    dpi = 300
    for idxat, idx_f in enumerate(figs[0]):
        fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                                sharey=True, figsize=(4000/dpi, 2500/dpi))
        for idx_c, val_c in enumerate(col_plot[0]):
            for idx_r, val_r in enumerate(row_plot[0]):
                ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
                data = df_res[(df_res['layer'] == val_r)&(df_res['no_resources'] == val_c)&
                              ((df_res[figs[1]] == idx_f)|(df_res[figs[1]] == 'nan'))]
                with sns.xkcd_palette(['black', "windows blue", 'red', "green"]):
                    sns.lineplot(x='t', y='resource', hue=hue_type[1], style=style_type[1],
                                 markers=True, ci=ci, ax=ax, legend='full', data=data)
                    ax.get_legend().set_visible(False)
                    ax.set(xlabel=r'time step $t$', ylabel='No. resources')
                    ax.xaxis.set_ticks(np.arange(1, 11, 1.0))   #ax.get_xlim()
#                    ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 1.0), minor=True)
                    ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 1.0))
                    ax.grid(b=True, which='major', color='w', linewidth=1.0)
#                    ax.grid(b=True, which='minor', color='w', linewidth=0.5)
        handles, labels = ax.get_legend_handles_labels()
        labels = correct_legend_labels(labels)
        fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5, labelspacing=0.2)
        _, axs_c, axs_r = find_ax(axs, row_plot[0], col_plot[0])
        fig.suptitle('Valuation Type: '+valuation_type[idxat])
        for idx, ax in enumerate(axs_c):
            ax.set_title(r'Resource Cap: %d'%(col_plot[0][idx]))
        for idx, ax in enumerate(axs_r):
            ax.annotate('Layer '+str(row_plot[0][idx]), xy=(0.1, 0.5),
                        xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', ha='right', va='center', rotation=90)
        plt.savefig('Allocations_'+idx_f+'.png', dpi=600)

def plot_auction_allocation_synthetic(df_res, resource_type='resource', ci=None):
    '''
    Parameters
    ----------
    df_res : TYPE
        DESCRIPTION.
    resource_type : TYPE, optional
        DESCRIPTION. The default is 'resource'.
    ci : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    layer = df_res.layer.unique().tolist()
    auction_type = df_res.auction_type.unique().tolist()
    if '' in auction_type:
        auction_type.remove('')
    valuation_type = df_res.valuation_type.unique().tolist()
    if '' in valuation_type:
        valuation_type.remove('')
    T = len(df_res.t.unique().tolist())
    hor_grid = auction_type
    ver_grid = layer
    new_window = valuation_type
    for idxat, at in enumerate(new_window):
        fig, axs = plt.subplots(len(ver_grid), len(hor_grid), sharex=True, sharey=True,
                                tight_layout=False)
        for idxnr, nr in enumerate(hor_grid):
            for idxvt, vt in enumerate(ver_grid):
                if len(ver_grid) == 1 and len(hor_grid) == 1:
                    ax = axs
                elif len(ver_grid) == 1:
                    ax = axs[idxnr]
                elif len(hor_grid) == 1:
                    ax = axs[idxvt]
                else:
                    ax = axs[idxvt, idxnr]
                data = df_res[(df_res['layer'] == vt)&\
                              ((df_res['auction_type'] == nr)|(df_res['auction_type'] == ''))&\
                              ((df_res['valuation_type'] == at)|(df_res['valuation_type'] == ''))]
                with sns.xkcd_palette(['black', "windows blue", 'red', "green"]):
                    ax = sns.lineplot(x='t', y=resource_type, hue="decision_type",
                                      style='decision_type', markers=True, ci=ci,
                                      ax=ax, legend='full', data=data)
                    ax.get_legend().set_visible(False)
                    ax.set(xlabel=r'time step $t$', ylabel=resource_type)
                    if resource_type == "normalized_resource":
                        ax.set(ylabel=r'$\% R_c$')
                    ax.xaxis.set_ticks(np.arange(1, T+1, 1.0))   #ax.get_xlim()
                    ax.grid(b=True, which='major', color='w', linewidth=1.0)
        handles, labels = ax.get_legend_handles_labels()
        labels = correct_legend_labels(labels)
        fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5, labelspacing=0.2)
        _, axs_c, axs_r = find_ax(axs, ver_grid, hor_grid)
        fig.suptitle('Valuation Type: '+valuation_type[idxat])
        for idx, ax in enumerate(axs_c):
            ax.set_title(r'Auction Type: %s'%(hor_grid[idx]))
        for idx, ax in enumerate(axs_r):
            ax.annotate('Layer '+str(int(ver_grid[idx])), xy=(0.1, 0.5),
                        xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', ha='right', va='center', rotation=90)
        plt.savefig('Allocations_'+at+'.pdf', dpi=600)

def plot_relative_allocation_shelby(gap_res, distance_type='gap'):
    '''
    Parameters
    ----------
    gap_res : TYPE
        DESCRIPTION.
    distance_type : TYPE, optional
        DESCRIPTION. The default is 'gap'.

    Returns
    -------
    None.

    '''
    #: Make lists
    # no_resources = gap_res.no_resources.unique().tolist()
    layer = gap_res.layer.unique().tolist()
    # decision_type = gap_res.decision_type.unique().tolist()
    judgment_type = gap_res.judgment_type.unique().tolist()
    auction_type = gap_res.auction_type.unique().tolist()
    valuation_type = gap_res.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    row_plot = [valuation_type, 'valuation_type']
    col_plot = [auction_type, 'auction_type']
    hue_type = [judgment_type, 'judgment_type']
    # clrs = [['azure', 'light blue'], ['gold', 'khaki'], ['strawberry', 'salmon pink'],
    #         ['green', 'light green']] #['purple', 'orchid'
    clrs = [['#003f5c', '#006999'], ['#7a5195', '#00a1ae'], ['#ef5675', '#30cf6f'],
            ['#ffa600', '#ffe203']] #['purple', 'orchid']
    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                            sharey=True, figsize=(4000/dpi, 2500/dpi))
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            data_ftp = gap_res[(gap_res[col_plot[1]] == val_c)&\
                               ((gap_res[row_plot[1]] == val_r)|\
                               (gap_res[row_plot[1]] == 'nan'))]
            data_ftp_pivot = data_ftp.pivot_table(values=distance_type,
                index=data_ftp.columns.drop(['layer', 'gap', 'norm_gap']).tolist(),
                columns='layer')
            data_ftp_pivot.reset_index(inplace=True)
            lyrs = data_ftp['layer'].unique().tolist()
            for l in lyrs:
                if l != 1:
                    data_ftp_pivot[l] += data_ftp_pivot[l-1]
            lyrs.sort(reverse=True)
            for l in lyrs:
                with sns.color_palette(clrs[int(l)-1]): #pals[int(l)-1]:
                    sns.barplot(x='no_resources', y=l, hue=hue_type[1], ax=ax,
                                data=data_ftp_pivot, linewidth=0.5,
                                edgecolor=[.25, .25, .25], capsize=.05,
                                errcolor=[.25, .25, .25], errwidth=.75)
            ax.get_legend().set_visible(False)
            ax.grid(which='major', axis='y', color=[.75, .75, .75], linewidth=.75)
            ax.set_xlabel(r'\# resources')
            if idx_r != len(row_plot[0])-1:
                ax.set_xlabel('')
            ax.set_ylabel(r'$E[\omega^k]$, Valuation: %s' % (row_plot[0][idx_r]))
            if idx_c != 0:
                ax.set_ylabel('')
            ax.xaxis.set_label_position('bottom')
            ax.set_facecolor('w')
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    for idx, lab in enumerate(labels):
        layer_num = len(layer) - idx//(len(judgment_type))
        labels[idx] = lab[:7] + '. (Layer ' + str(layer_num) + ')'
    lgd = fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.95),
                     frameon=True, framealpha=0.5, ncol=4)#, fontsize='small'
    _, axs_c, _ = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'Auction Type: %s'%(col_plot[0][idx]))
    plt.savefig('Allocation_Gap.png', bbox_extra_artists=(lgd, ), dpi=dpi)

def plot_relative_allocation_synthetic(df_res, distance_type='distance_to_optimal'):
    '''
    Parameters
    ----------
    df_res : TYPE
        DESCRIPTION.
    distance_type : TYPE, optional
        DESCRIPTION. The default is 'distance_to_optimal'.

    Returns
    -------
    None.

    '''
#    no_resources = df_res.no_resources.unique().tolist()
    layer = df_res.layer.unique().tolist()
    decision_type = df_res.decision_type.unique().tolist()
#    decision_type=['judgeCall_OPTIMISTIC']
#    auction_type = df_res.auction_type.unique().tolist()
    valuation_type = df_res.valuation_type.unique().tolist()
    if '' in valuation_type:
        valuation_type.remove('')
    hor_grid = [0]
    ver_grid = valuation_type
    # pals = [sns.color_palette("Blues", 3), sns.color_palette("Reds", 3),
    #         sns.color_palette("Greens", 3), sns.cubehelix_palette(3)]
    clrs = [['strawberry', 'salmon pink'], ['azure', 'light blue'],
            ['green', 'light green'], ['bluish purple', 'orchid']]
    fig, axs = plt.subplots(len(ver_grid), len(hor_grid), sharex=True, sharey=True,
                            tight_layout=False, figsize=(8, 6))
    for idxat, _ in enumerate(hor_grid):
        for idxvt, vt in enumerate(ver_grid):
            ax, _, _ = find_ax(axs, hor_grid, ver_grid, idxat, idxvt)
            data_ftp = df_res[(df_res['valuation_type'] == vt)|(df_res['valuation_type'] == '')]
            for dt in decision_type:
                bottom = 0
                for P in layer:
                    data_ftp.loc[(data_ftp['layer'] == P)&(data_ftp['decision_type'] == dt)&\
                                 (data_ftp['valuation_type'] == vt), distance_type] += bottom
                    bottom = data_ftp[(data_ftp['layer'] == P)&(data_ftp['decision_type'] == dt)&\
                                      (data_ftp['valuation_type'] == vt)][distance_type].mean()
                bottom = 0
                for P in layer:
                    data_ftp.loc[(data_ftp['layer'] == P)&(data_ftp['decision_type'] == dt)&\
                                 (data_ftp['valuation_type'] == ''), distance_type] += bottom
                    bottom = data_ftp[(data_ftp['layer'] == P)&(data_ftp['decision_type'] == dt)&\
                                      (data_ftp['valuation_type'] == '')][distance_type].mean()
            for P in reversed(layer):
                with sns.xkcd_palette(clrs[int(P)-1]): #pals[int(P)-1]:
                    sns.barplot(x='auction_type', y=distance_type, hue="decision_type",
                                data=data_ftp[(data_ftp['layer'] == P)],
                                linewidth=0.5, edgecolor=[.25, .25, .25],
                                capsize=.05, errcolor=[.25, .25, .25], errwidth=.75, ax=ax)
            ax.get_legend().set_visible(False)
            ax.grid(which='major', axis='y', color=[.75, .75, .75], linewidth=.75)
            ax.set_xlabel(r'Auction Type')
            if idxvt != len(valuation_type)-1:
                ax.set_xlabel('')
            ax.set_ylabel(r'$E[\omega^k(r^k_d, r^k_c)]$, Valuation: %s' %(valuation_type[idxvt]))
            if idxat != 0:
                ax.set_ylabel('')
            ax.xaxis.set_label_position('bottom')
            ax.set_facecolor('w')
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    for idx, lab in enumerate(labels):
        layer_num = len(layer) - idx//(len(decision_type))
        labels[idx] = lab[:7] + '. (Layer ' + str(layer_num) + ')'
    lgd = fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.95),
                     frameon=True, framealpha=0.5, ncol=4)#, fontsize='small'
    plt.savefig('Allocation_Difference.pdf', bbox_extra_artists=(lgd, ), dpi=600)

def plot_run_time(df, ci=None):
    '''
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    ci : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    #: Make lists
    # no_resources = df.no_resources.unique().tolist()
    # layer = df.layer.unique().tolist()
    # decision_type = df.decision_type.unique().tolist()
    judgment_type = df.judgment_type.unique().tolist()
    auction_type = df.auction_type.unique().tolist()
    if 'nan' in auction_type:
        auction_type.remove('nan')
    valuation_type = df.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    T = len(df['t'].unique().tolist())
    value_vars = ['valuation_time', 'auction_time', 'decision_time']
    row_plot = [valuation_type, 'valuation_type']
    col_plot = [auction_type, 'auction_type']
    hue_type = [judgment_type, 'judgment_type']
    # style_type = [decision_type, 'decision_type']

    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                            sharey=True, figsize=(4000/dpi, 2500/dpi))
    pals = [sns.cubehelix_palette(4, rot=-0.4, reverse=True), sns.color_palette("Reds", 3),
            sns.cubehelix_palette(4, reverse=True)]
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            selected_data = df[((df[row_plot[1]] == val_r)|(df[row_plot[1]] == 'nan'))&\
                               ((df[col_plot[1]] == val_c)|(df[col_plot[1]] == 'nan'))]
            # selected_data.loc[:, 'valuation_time'] += selected_data.loc[:, 'auction_time']+\
            #     selected_data.loc[:, 'decision_time']
            # selected_data.loc[:, 'auction_time'] += selected_data.loc[:, 'decision_time']
            for tt in value_vars:
                with pals[int(value_vars.index(tt))]:
                    sns.lineplot(x='t', y=tt, hue=hue_type[1], markers=True, ci=ci, ax=ax,
                                 data=selected_data[selected_data['t'] > 0], **{'markersize':5})
                    # ax=sns.barplot(x='t', y=tt, hue=hue_type[1], data=selected_data,
                    #             linewidth=0.5, edgecolor=[.25, .25, .25], ax=ax,
                    #             capsize=.05, errcolor=[.25, .25, .25], errwidth=.75)
            ax.set(xlabel=r'time step $t$')
            if idx_r != len(row_plot[0])-1:
                ax.set_xlabel('')
            ax.set_ylabel('Run Time (sec), Val.: %s'% (row_plot[0][idx_r]))
            if idx_c != 0:
                ax.set_ylabel('')
            ax.get_legend().set_visible(False)
            ax.xaxis.set_ticks(np.arange(0, T, 1.0))   #ax.get_xlim()
    handles, labels = ax.get_legend_handles_labels()
    i = 0
    for x in labels:
        if x == 'judgment_type':
            labels[labels.index(x)] = value_vars[i]
            i += 1
    labels = correct_legend_labels(labels)
    # for idx, lab in enumerate(labels):
    #     labels[idx] = correct_legend_labels([value_vars[idx//3]])[0]+' ('+lab[:7]+')'
    fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5)
    _, axs_c, _ = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'Res. Alloc.: %s'%(col_plot[0][idx]))
    plt.savefig('run_time.png', dpi=dpi, bbox_inches='tight')

def plot_seperated_perform_curves(df, x='t', y='cost', cost_type='Total',
                                  ci=None, normalize=False):
    '''
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    x : TYPE, optional
        DESCRIPTION. The default is 't'.
    y : TYPE, optional
        DESCRIPTION. The default is 'cost'.
    cost_type : TYPE, optional
        DESCRIPTION. The default is 'Total'.
    ci : TYPE, optional
        DESCRIPTION. The default is None.
    normalize : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    res_caps = df.no_resources.unique().tolist()
    valuation_type = df.valuation_type.unique().tolist()
    auction_type = df.auction_type.unique().tolist()
    layers = df.layer.unique().tolist()
    if 'nan' in layers:
        layers.remove('nan')
    layer_names = {1:'Water', 2:'Gas', 3:'Power', 4:'Telecom.'}
    colors = ['#154352', '#dbb539', '#007268', '#5d9c51']
    pal = sns.color_palette(colors[:len(auction_type)-1]+['k'])
    dpi = 300
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, tight_layout=False,
                            figsize=(3000/dpi, 2500/dpi))
    cost_data = df[df.cost_type == cost_type]
    for idx, lyr in enumerate(layers):
        ax = axs[idx//2, idx%2]
        sns.lineplot(x=x, y=y, hue="auction_type", style='decision_type', markers=True,
                     ci=ci, ax=ax, palette=pal,
                     data=cost_data[cost_data.layer == lyr], **{'markersize':5})
        ax.set(xlabel=r'time step $t$', ylabel=cost_type+' Cost')
        ax.set_title(r'Layer: %s'%(layer_names[lyr]))
        ax.get_legend().set_visible(False)
        ax.xaxis.set_ticks(np.arange(0, 10, 1.0))   #ax.get_xlim()

    ax = fig.add_subplot(3, 2, 6)
    sns.lineplot(x=x, y=y, hue="auction_type", style='decision_type',
                 markers=True, ci=ci, palette=pal, ax=ax,
                 data=cost_data[cost_data.layer == 'nan'], **{'markersize':5})
    ax.set(xlabel=r'time step $t$', ylabel=cost_type+' Cost')
    ax.set_title(r'Overall')
    ax.get_legend().set_visible(False)
    ax.xaxis.set_ticks(np.arange(0, 10, 1.0))   #ax.get_xlim()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    head = 'Resource Cap: '+str(res_caps).strip('[]')+', Valuation: '+\
        str(valuation_type).strip('[]')
    fig.suptitle(head)
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='best', ncol=1, framealpha=0.5)
    plt.savefig('sep_perf.png', dpi=dpi, bbox_inches='tight')

def correct_legend_labels(labels):
    '''
    Parameters
    ----------
    labels : list
        DESCRIPTION.

    Returns
    -------
    labels : list
        DESCRIPTION.

    '''
    labels = ['iINDP' if x == 'sample_indp_12Node' else x for x in labels]
    labels = ['JC Optimistic' if x == 'sample_judgeCall_12Node_OPTIMISTIC' else x for x in labels]
    labels = ['JC Pessimistic' if x == 'sample_judgeCall_12Node_PESSIMISTIC' else x for x in labels]
    labels = ['Res. Alloc. Type' if x == 'auction_type' else x for x in labels]
    labels = ['Judge. Type' if x == 'judgment_type' else x for x in labels]
    labels = ['Decision Type' if x == 'decision_type' else x for x in labels]
    labels = ['Valuation Type' if x == 'valuation_type' else x for x in labels]
    labels = ['Auction Time' if x == 'auction_time' else x for x in labels]
    labels = ['Decision Time' if x == 'decision_time' else x for x in labels]
    labels = ['Valuation Time' if x == 'valuation_time' else x for x in labels]
    labels = ['iINDP' if x == 'indp' else x for x in labels]
    labels = ['iINDP' if x == 'nan' else x for x in labels]#!!!
    labels = ['td-INDP' if x == 'tdindp' else x for x in labels]
    labels = ['?JC Optimistic' if x == 'judgeCall_OPTIMISTIC' else x for x in labels]
    labels = ['?JC Pessimistic' if x == 'judgeCall_PESSIMISTIC' else x for x in labels]
    labels = ['Judge. Call' if x == 'jc' else x for x in labels]
    return labels

def find_ax(axs, row_plot, col_plot, idx_r=0, idx_c=0):
    '''
    Parameters
    ----------
    axs : TYPE
        DESCRIPTION.
    row_plot : TYPE
        DESCRIPTION.
    col_plot : TYPE
        DESCRIPTION.
    idx_r : TYPE
        DESCRIPTION.
    idx_c : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if len(row_plot) == 1 and len(col_plot) == 1:
        ax = axs
        axs_c = [axs]
        axs_r = [axs]
    elif len(row_plot) == 1:
        ax = axs[idx_c]
        axs_r = [axs[0]]
        axs_c = axs
    elif len(col_plot) == 1:
        ax = axs[idx_r]
        axs_r = axs
        axs_c = [axs[0]]
    else:
        ax = axs[idx_r, idx_c]
        axs_c = axs[0, :]
        axs_r = axs[:, 0]
    return ax, axs_c, axs_r
