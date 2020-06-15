'''plots'''
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
sns.set(context='notebook', style='darkgrid', font_scale=1.2)
# plt.rc('text', usetex=True)
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

def plot_performance_curves(df, x='t', y='cost', cost_type='Total',
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
    col_plot = [judgment_type, 'judgment_type']#no_resources, 'no_resources'
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
            with pal:
                sns.lineplot(x=x, y=y, hue=hue_type[1], style=style_type, markers=True, ci=ci,
                             ax=ax, data=cost_data[cost_data.layer == 'nan'],
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
                with pal:
                    sns.lineplot(x=x, y=y, hue=hue_type[1], style=style_type, markers=True,
                                 ci=ci, ax=ax_2, legend='full',
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
        ax.set_title(r'Total resources=%s'%(str(col_plot[0][idx])))
    for idx, ax in enumerate(axs_r):
        ax.annotate('Valuation: '+row_plot[0][idx], xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 4, 0), xycoords=ax.yaxis.label,
                    textcoords='offset points', ha='right', va='center', rotation=90)
    plt.savefig('Performance_curves.png', dpi=dpi)

def plot_relative_performance(lambda_df, cost_type='Total', lambda_type='U'):
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
    if 'nan' in judgment_type:
        judgment_type.remove('nan')
    no_resources = lambda_df.no_resources.unique().tolist()
    auction_type = lambda_df.auction_type.unique().tolist()
    if 'nan' in auction_type:
        auction_type.remove('nan')
    valuation_type = lambda_df.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    row_plot = [valuation_type, 'valuation_type']
    col_plot = [judgment_type , 'judgment_type']
    hue_type = [auction_type , 'auction_type']#[judgment_type, 'judgment_type']
    x = 'judgment_type' #no_resources
    # Initialize plot properties
    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                            sharey=True, figsize=(2100/dpi, 1600/dpi))
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            selected_data = lambda_df[(lambda_df.cost_type == cost_type)&
                                      (lambda_df['lambda_'+lambda_type] != 'nan')&
                                      ((lambda_df[col_plot[1]] == val_c)|\
                                       (lambda_df[col_plot[1]] == 'nan'))&\
                                      ((lambda_df[row_plot[1]] == val_r)|\
                                       (lambda_df[row_plot[1]] == 'nan'))]
            with sns.color_palette("Reds", 4): #sns.color_palette("RdYlGn", 8)
                sns.barplot(x=x, y='lambda_'+lambda_type,
                            hue=hue_type[1], data=selected_data, linewidth=0.5,
                            edgecolor=[.25, .25, .25], capsize=.05,
                            errcolor=[.25, .25, .25], errwidth=1, ax=ax) 
                ax.get_legend().set_visible(False)
                ax.set_xlabel(r'$R_c$')
                if idx_r != len(valuation_type)-1:
                    ax.set_xlabel('')
                ax.set_ylabel(r'E[$\lambda_{%s}$], Valuation: %s'%(lambda_type, row_plot[0][idx_r]))
                if idx_c != 0:
                    ax.set_ylabel('')
                ax.xaxis.set_label_position('bottom')
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor = (0.9,0.1),
               frameon=True, framealpha=.75, ncol=1)
    _, axs_c, _ = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'Judgment Type: %s'%(col_plot[0][idx]))
    plt.savefig('Relative_perforamnce.png', dpi=dpi)

def plot_auction_allocation(df_res, ci=None):
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
    judgment_type = df_res.judgment_type.unique().tolist()
    if 'nan' in judgment_type:
        judgment_type.remove('nan')
    auction_type = df_res.auction_type.unique().tolist()
    if 'nan' in auction_type:
        auction_type.remove('nan')
    valuation_type = df_res.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    figs = [judgment_type, 'judgment_type']#valuation_type, 'valuation_type'
    row_plot = [layer, 'layer']
    col_plot = [judgment_type, 'judgment_type']#no_resources, 'no_resources'
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
                data = df_res[(df_res[row_plot[1]] == val_r)&(df_res[col_plot[1]] == val_c)&
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
        fig.suptitle('Judgment Type: '+figs[0][idxat])
        for idx, ax in enumerate(axs_c):
            ax.set_title(r'Resource Cap: %s'%str(col_plot[0][idx]))
        for idx, ax in enumerate(axs_r):
            ax.annotate('Layer '+str(row_plot[0][idx]), xy=(0.1, 0.5),
                        xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', ha='right', va='center', rotation=90)
        plt.savefig('Allocations_'+idx_f+'.png', dpi=600)

def plot_relative_allocation(gap_res, distance_type='gap'):
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
    if 'nan' in judgment_type:
        judgment_type.remove('nan')
    auction_type = gap_res.auction_type.unique().tolist()
    valuation_type = gap_res.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    row_plot = [judgment_type, 'judgment_type']#valuation_type, 'valuation_type'
    col_plot = [auction_type, 'auction_type']
    hue_type = [layer, 'layer']
    clrs=['#5153ca', '#e4ad5d', '#c20809', '#5fb948']
    x = 'judgment_type' #no_resources
    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                            sharey=True, figsize=(3500/dpi, 1000/dpi))
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            data_ftp = gap_res[(gap_res[col_plot[1]] == val_c)&\
                               ((gap_res[row_plot[1]] == val_r)|\
                               (gap_res[row_plot[1]] == 'nan'))]
            with sns.color_palette(clrs):
                sns.barplot(x=x, y=distance_type, hue=hue_type[1], ax=ax,
                            data=data_ftp, linewidth=0.5,
                            edgecolor=[.25, .25, .25], capsize=.05,
                            errcolor=[.25, .25, .25], errwidth=.75) #no_resources
            ax.get_legend().set_visible(False)
            ax.set_xlabel(r'$R_c$')
            if idx_r != len(row_plot[0])-1:
                ax.set_xlabel('')
            ax.set_ylabel(r'$E[\omega^k]$, Valuation: %s' % (row_plot[0][idx_r]))
            if idx_c != 0:
                ax.set_ylabel('')
            ax.xaxis.set_label_position('bottom')
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    for idx, lab in enumerate(labels):
        layer_label = {1:'Water', 2:'Gas', 3:'Power', 4:'Telecomm.'} #!!! only for shelby
        labels[idx] = layer_label[idx+1]
    lgd = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.85, 0.97),
                     frameon=True, framealpha=0.5, ncol=1, fontsize='x-small')
    _, axs_c, _ = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'Res. Alloc. Type: %s'%(col_plot[0][idx]))
    plt.savefig('Allocation_Gap.png', dpi=dpi, bbox_inches='tight',
                bbox_extra_artists=(lgd, ))#

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
    no_resources = df.no_resources.unique().tolist()
    # layer = df.layer.unique().tolist()
    # decision_type = df.decision_type.unique().tolist()
    judgment_type = df.judgment_type.unique().tolist()
    if 'nan' in judgment_type:
        judgment_type.remove('nan')
    auction_type = df.auction_type.unique().tolist()
    if 'nan' in auction_type:
        auction_type.remove('nan')
    valuation_type = df.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    T = len(df['t'].unique().tolist())
    value_vars = ['total_time','valuation_time', 'auction_time']#'decision_time'
    row_plot = [valuation_type, 'valuation_type']
    col_plot = [judgment_type, 'judgment_type'] #no_resources, 'no_resources'
    hue_type = [auction_type, 'auction_type']

    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                            sharey=True, figsize=(4000/dpi, 1500/dpi))
    clrs=['#000000', '#e9937c', '#a6292d']
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            selected_data = df[((df[row_plot[1]] == val_r)|(df[row_plot[1]] == 'nan'))&\
                               ((df[col_plot[1]] == val_c)|(df[col_plot[1]] == 'nan'))]
            selected_data['total_time'] = selected_data.loc[:, 'valuation_time']+\
                selected_data.loc[:, 'auction_time']+\
                selected_data.loc[:, 'decision_time']
            id_vars = [x for x in list(selected_data.columns) if x not in value_vars]
            df_melt = pd.melt(selected_data, id_vars=id_vars, value_vars=value_vars)
            # Removing non-informative lines
            df_melt = df_melt[~((df_melt['decision_type']=='indp')&(df_melt['variable']=='valuation_time'))]
            df_melt = df_melt[~((df_melt['decision_type']=='indp')&(df_melt['variable']=='decision_time'))]
            df_melt = df_melt[~((df_melt['decision_type']=='indp')&(df_melt['variable']=='auction_time'))]
            df_melt = df_melt[~((df_melt['auction_type']=='UNIFORM')&(df_melt['variable']=='valuation_time'))]
            df_melt = df_melt[~((df_melt['auction_type']=='UNIFORM')&(df_melt['variable']=='decision_time'))]
            df_melt = df_melt[~((df_melt['auction_type']=='UNIFORM')&(df_melt['variable']=='auction_time'))]
            df_melt = df_melt[~((df_melt['auction_type']=='MCA')&(df_melt['variable']=='decision_time'))]
            df_melt = df_melt[~((df_melt['auction_type']=='MDA')&(df_melt['variable']=='decision_time'))]
            df_melt = df_melt[~((df_melt['auction_type']=='MAA')&(df_melt['variable']=='decision_time'))]

            with sns.color_palette(clrs):
                sns.lineplot(x='t', y="value", hue=hue_type[1], style='variable',
                             markers=True, ci=ci, ax=ax, style_order=value_vars,
                             data=df_melt[df_melt['t'] > 0], **{'markersize':6})
            ax.set(xlabel=r'Time Step $t$')
            if idx_r != len(row_plot[0])-1:
                ax.set_xlabel('')
            ax.set_ylabel('Mean Time (sec), Val.: %s'% (row_plot[0][idx_r]))
            if idx_c != 0:
                ax.set_ylabel('')
            ax.get_legend().set_visible(False)
            ax.xaxis.set_ticks(np.arange(0, T, 2.0))   #ax.get_xlim()
    handles, labels = ax.get_legend_handles_labels()
    i = 0
    for x in labels:
        if x == 'judgment_type':
            labels[labels.index(x)] = value_vars[i]
            i += 1
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95,0.88),
               ncol=1, framealpha=0.5, fontsize='x-small')
    _, axs_c, _ = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'$R_c=$ %s'%(col_plot[0][idx]))
    plt.savefig('run_time.png', dpi=dpi)

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
    layer_names = {1:'Water', 2:'Gas', 3:'Power', 4:'Telecom.'}#!!! just for shelby
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
    labels = ['Total Time' if x == 'total_time' else x for x in labels]
    labels = ['Decision Time' if x == 'decision_time' else x for x in labels]
    labels = ['Valuation Time' if x == 'valuation_time' else x for x in labels]
    labels = ['Time Type' if x == 'variable' else x for x in labels]
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

# Color repository
# clrs = [['azure', 'light blue'], ['gold', 'khaki'], ['strawberry', 'salmon pink'],
#         ['green', 'light green']] #['purple', 'orchid'
# clrs = [['strawberry','salmon pink'],['azure','light blue'],['green','light green'],['bluish purple','orchid']]
# clrs = [['#003f5c', '#006999'], ['#7a5195', '#00a1ae'], ['#ef5675', '#30cf6f'],
#         ['#ffa600', '#ffe203']] #['purple', 'orchid']
    # pals = [sns.cubehelix_palette(4, rot=-0.4, reverse=True), sns.color_palette("Reds_r", 10),
    #         sns.cubehelix_palette(4, reverse=True)]