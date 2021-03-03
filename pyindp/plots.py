'''plots'''
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
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
    return_period = df.return_period.unique().tolist()
    if not decision_type:
        decision_type = df.decision_type.unique().tolist()
    if not judgment_type:
        judgment_type = df.judgment_type.unique().tolist()
    # if 'nan' in judgment_type:
    #     judgment_type.remove('nan')
    if not auction_type:
        auction_type = df.auction_type.unique().tolist()
    # if 'nan' in auction_type:
    #     auction_type.remove('nan')
    if not valuation_type:
        valuation_type = df.valuation_type.unique().tolist()
    # if 'nan' in valuation_type:
    #     valuation_type.remove('nan')
    T = len(df[x].unique().tolist())
    #sns.color_palette("husl", len(auction_type)+1)
    row_plot = [judgment_type, 'judgment_type'] # valuation_type
    col_plot = [no_resources, 'no_resources'] # no_resources, judgment_type
    hue_type = [return_period, 'return_period'] #auction_type
    style_type = 'return_period'  #decision_type
    # Initialize plot properties
    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True, sharey=True,
                            figsize=(6000/dpi, 1500/dpi))
    colors = ['#154352', '#007268', '#5d9c51', '#dbb539', 'k']
    # colors = ['r', 'b', 'k']
    # pal = sns.color_palette(colors[:len(hue_type[0])])
    pal = sns.diverging_palette(250, 1000, l=65, center="dark")
    x_ticks = np.arange(0, T+1, 2.0)
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            cost_data = df[(df.cost_type == cost_type)&
                           (df.decision_type.isin(decision_type))&
                           (df[col_plot[1]] == val_c)&
                           ((df[row_plot[1]] == val_r)|(df[row_plot[1]] == 'nan'))]
            with pal:
                sns.lineplot(x=x, y=y, hue=hue_type[1], style=style_type, markers=True,
                             ci=ci, ax=ax, data=cost_data[cost_data.layer == 'nan'],
                             **{'markersize':5})
            if deaggregate:
                cost_lyr = cost_data[cost_data.layer != 'nan']
                cost_lyr_pivot = cost_lyr.pivot_table(values='cost',
                    index=cost_lyr.columns.drop(['layer', 'cost', 'normalized_cost']).tolist(),
                    columns='layer')
                cost_lyr_pivot.reset_index(inplace=True)
                layers = cost_lyr['layer'].unique().tolist()
                for idxl, l in enumerate(layers):
                    if idxl != 0:
                        cost_lyr_pivot[l] += cost_lyr_pivot[layers[idxl-1]]
                layers.sort(reverse=True)
                for l in layers:
                    pal_adj = sns.color_palette([[xx*l/len(layers) for xx in x] for x in pal])
                    sns.barplot(x=x, y=l, hue=hue_type[1], ax=ax, linewidth=0.5, ci=None,
                                data=cost_lyr_pivot, hue_order=hue_type[0],
                                palette=pal_adj, **{'alpha':0.35})
                    ax.xaxis.grid(True)
            
            ax.get_legend().set_visible(False)
            ax.xaxis.set_ticks(x_ticks)
            ax.set(xlabel=r'time step $t$', ylabel= cost_type+' Cost')#r'\% Unmet Demand')
            # ax.yaxis.set_ticks(np.arange(-.4, 0.05, 0.05))

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
                    for idxl, l in enumerate(layers):
                        if idxl != 0:
                            cost_lyr_pivot[l] += cost_lyr_pivot[layers[idxl-1]]
                    layers.sort(reverse=True)
                    for l in layers:
                        pal_adj = sns.color_palette([[xx*l/len(layers) for xx in x] for x in pal])
                        sns.barplot(x=x, y=l, hue=hue_type[1], ax=ax_2, linewidth=0.5,
                                    ci=None, data=cost_lyr_pivot, hue_order=hue_type[0],
                                    palette=pal_adj, **{'alpha':0.35})
                    ax_2.xaxis.grid(True)
                ax_2.set(xlabel=r'time step $t$ (weeks)', ylabel=r'\% Unmet Demand')
                ax_2.get_legend().set_visible(False)
                if idx_c != 0.0:
                    ax_2.set_ylabel('')
                    ax_2.set_yticklabels([])
                ax.xaxis.set_ticks([])
                ax_2.xaxis.set_ticks(x_ticks)#ax.get_xlim()
                ax_2.yaxis.set_ticks(np.arange(-.8, 0.01, 0.2))
    # Rebuild legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [x for x in handles if isinstance(x, mplt.lines.Line2D)]
    labels = correct_labels(labels)
    fig.legend(handles, labels, loc='lower right', ncol=1, framealpha=0.35,
               bbox_to_anchor=(.9, 0.11)) 
    # Add overll x- and y-axis titles
    _, axs_c, axs_r = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        label = correct_labels([str(col_plot[0][idx])])[0]
        ax.set_title(r'$R_c:$ %s'%(label))
    for idx, ax in enumerate(axs_r):
        ax.annotate(str(row_plot[0][idx]), xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 4, 0), xycoords=ax.yaxis.label,
                    textcoords='offset points', ha='right', va='center', rotation=90)
    plt.savefig('Performance_curves.png', dpi=dpi, bbox_inches='tight')

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
    no_resources = lambda_df.no_resources.unique().tolist()
    decision_type = lambda_df.decision_type.unique().tolist()
    if 'indp_sample_12Node' in decision_type:
        decision_type.remove('indp_sample_12Node')
    if 'indp' in decision_type:
        decision_type.remove('indp')
    judgment_type = lambda_df.judgment_type.unique().tolist()
    if 'nan' in judgment_type:
        judgment_type.remove('nan')
    auction_type = lambda_df.auction_type.unique().tolist()
    if 'nan' in auction_type:
        auction_type.remove('nan')
    valuation_type = lambda_df.valuation_type.unique().tolist()
    # if 'nan' in valuation_type:
    #     valuation_type.remove('nan')
    row_plot = [valuation_type, 'valuation_type'] #valuation_type
    col_plot = [decision_type , 'decision_type']
    hue_type = [auction_type , 'auction_type']
    x = 'no_resources' 
    # Initialize plot properties
    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                            sharey=True, figsize=(2000/dpi, 1200/dpi))
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
                ax.set_ylabel(r'E[$\lambda_{%s}$], $%s$'%(lambda_type, row_plot[0][idx_r]))
                ax.set_ylabel(r'E[$\lambda_{%s}$]'%lambda_type)
                if idx_c != 0:
                    ax.set_ylabel('')
                ax.xaxis.set_label_position('bottom')
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_labels(labels)
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor = (0.9,0.15),
                frameon=True, framealpha=.75, ncol=1)
    _, axs_c, _ = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        corrected_label = correct_labels([col_plot[0][idx]])[0]
        ax.set_title(r'Decision Type: %s'%(corrected_label))
    plt.savefig('Relative_perforamnce.png', dpi=dpi, bbox_inches='tight')

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
    T = len(df_res.t.unique().tolist())
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
    col_plot = [no_resources, 'no_resources']#no_resources, judgment_type
    hue_type = [auction_type, 'auction_type'] # decision_type
    style_type = [decision_type, 'decision_type'] # auction_type
    # Initialize plot properties
    dpi = 300
    for idxat, idx_f in enumerate(figs[0]):
        fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                                sharey=True, figsize=(3000/dpi, 2500/dpi))
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        for idx_c, val_c in enumerate(col_plot[0]):
            for idx_r, val_r in enumerate(row_plot[0]):
                ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
                data = df_res[(df_res[row_plot[1]] == val_r)&(df_res[col_plot[1]] == val_c)&
                              ((df_res[figs[1]] == idx_f)|(df_res[figs[1]] == 'nan'))]
                # with sns.xkcd_palette(['red', "windows blue", 'black', "green"]): #!!!
                with sns.color_palette(['k', 'r', 'b']):
                    sns.lineplot(x='t', y='resource', hue=hue_type[1], style=style_type[1],
                                 markers=True, ci=ci, ax=ax, legend='full', data=data)
                    ax.get_legend().set_visible(False)
                    ax.set(xlabel=r'time step $t$', ylabel='No. resources')
                    ax.xaxis.set_ticks(np.arange(1, T+1, 1.0))   #ax.get_xlim()
#                    ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 1.0), minor=True)
                    ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 1.0))
                    ax.grid(b=True, which='major', color='w', linewidth=1.0)
#                    ax.grid(b=True, which='minor', color='w', linewidth=0.5)
        handles, labels = ax.get_legend_handles_labels()
        labels = correct_labels(labels)
        fig.legend(handles, labels, loc='center', ncol=4, framealpha=0.5,
                   labelspacing=0.2, bbox_to_anchor=(.5, .95)) #!!!
        _, axs_c, axs_r = find_ax(axs, row_plot[0], col_plot[0])
        # fig.suptitle('Judgment Type: '+figs[0][idxat]) #!!!
        for idx, ax in enumerate(axs_c):
            ax.set_title(r'Resource Cap: %s'%str(col_plot[0][idx]))
        for idx, ax in enumerate(axs_r):
            names = ['Water', 'Gas', 'Power', 'Telecom.'] #!!!
            ax.annotate(names[idx], xy=(0.1, 0.5),
                        xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', ha='right', va='center', rotation=90)
                        #!!!'Layer '+str(row_plot[0][idx])
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
    decision_type = gap_res.decision_type.unique().tolist()
    if 'indp_sample_12Node' in decision_type:
        decision_type.remove('indp_sample_12Node')
    if 'indp' in decision_type:
        decision_type.remove('indp')
    if 'tdindp' in decision_type:
        decision_type.remove('tdindp')
    judgment_type = gap_res.judgment_type.unique().tolist()
    if 'nan' in judgment_type:
        judgment_type.remove('nan')
    auction_type = gap_res.auction_type.unique().tolist()
    if 'nan' in auction_type:
        auction_type.remove('nan')
    valuation_type = gap_res.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    row_plot = [decision_type, 'decision_type']#valuation_type, judgment_type
    col_plot = [auction_type, 'auction_type']
    hue_type = [layer, 'layer']
    clrs=['#5153ca', '#e4ad5d', '#c20809', '#5fb948']
    x = 'no_resources' #no_resources judgment_type
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
                            errcolor=[.25, .25, .25], errwidth=.75)
            ax.get_legend().set_visible(False)
            ax.set_xlabel(r'$R_c$')
            if idx_r != len(row_plot[0])-1:
                ax.set_xlabel('')
            corrected_label = correct_labels([row_plot[0][idx_r]])[0]
            ax.set_ylabel(r'$E[\omega^k]$, %s' % (corrected_label))
            if idx_c != 0:
                ax.set_ylabel('')
            ax.xaxis.set_label_position('bottom')
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_labels(labels)
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
    decision_type = df.decision_type.unique().tolist()
    if 'indp_sample_12Node' in decision_type:
        decision_type.remove('indp_sample_12Node')
    if 'indp' in decision_type:
        decision_type.remove('indp')
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
    row_plot = [decision_type, 'decision_type'] #valuation_type
    col_plot = [no_resources, 'no_resources'] #no_resources, judgment_type
    hue_type = [auction_type, 'auction_type']

    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True,
                            sharey=True, figsize=(4000/dpi, 1500/dpi))
    clrs=['#000000', '#e9937c', '#a6292d']
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            selected_data = df[((df[row_plot[1]] == val_r)|(df[row_plot[1]] == 'indp'))&\
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
            ax.set_ylabel('Mean Time (sec), %s'% (row_plot[0][idx_r]))
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
    labels = correct_labels(labels)
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
    T = df.t.unique().tolist()
    if 'nan' in layers:
        layers.remove('nan')
    layer_names = {1:'Water', 2:'Gas', 3:'Power', 4:'Telecom.'}#!!! Current convention
    # colors = ['#154352', '#dbb539', '#007268', '#5d9c51']
    # pal = sns.color_palette(colors[:len(auction_type)-1]+['k'])
    pal = sns.color_palette(['r', 'b'])
    dpi = 300
    fig, axs = plt.subplots(len(layers)//2+1, 2, sharex=True, sharey='row',
                            tight_layout=False, figsize=(4000/dpi, 3000/dpi))
    cost_data = df[df.cost_type == cost_type]
    for idx, lyr in enumerate(layers):
        ax = axs[idx//2, idx%2]
        with pal:
            sns.lineplot(x=x, y=y, hue="decision_type", style='auction_type', markers=True,
                         ci=ci, ax=ax, data=cost_data[cost_data.layer == lyr], **{'markersize':5})
            ax.set(xlabel=r'time step $t$', ylabel=cost_type+' Cost')
            ax.set_title(r'Layer: %s'%(layer_names[lyr]))
            ax.get_legend().set_visible(False)
            ax.xaxis.set_ticks(np.arange(0, len(T), 2.0))   #ax.get_xlim()

    ax = fig.add_subplot(len(layers)//2+1, 2, (len(layers)//2+1)*2)
    with pal:
        sns.lineplot(x=x, y=y, hue="decision_type", style='auction_type',
                      markers=True, ci=ci, ax=ax,
                      data=cost_data[cost_data.layer == 'nan'], **{'markersize':5})
        ax.set(xlabel=r'time step $t$', ylabel=cost_type+' Cost')
        ax.set_title(r'Overall')
        ax.get_legend().set_visible(False)
        ax.xaxis.set_ticks(np.arange(0, len(T), 2.0))   #ax.get_xlim()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    head = 'Resource Cap: '+str(res_caps).strip('[]')+', Valuation: '+\
        str(valuation_type).strip('[]')
    fig.suptitle(head)
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_labels(labels)
    fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5)
    plt.savefig('sep_perf.png', dpi=dpi, bbox_inches='tight')

def plot_ne_sol_2player(game, suffix='', plot_dir=''):
    '''
    This function plot the payoff functions of a normal game for one time step
    with nash equilibria and optimal solution marked on it (currently for 2-player games)

    Parameters
    ----------
    game : NormalGame object
        The game object for which the payoff matrix and solutions should be plotted.
    plot_dir: str
        The directory where the plott should be saved. The default is ''.
    suffix: str
        Suffix that should be added to the plot file name. The default is ''.
        
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
        pivot_dict = payoff_dict.pivot(index='P'+str(game.players[1])+' actions',
                                       columns='P'+str(game.players[0])+' actions',
                                       values='P'+str(l)+' payoff')
        mask = pivot_dict.copy()
        for col in mask:
            for i, row in mask.iterrows():
                if mask.at[i,col] == -1e100:
                    mask.loc[i,col] = True
                else:
                    mask.loc[i,col] = False
        sns.heatmap(pivot_dict, annot=False, linewidths=.2, cmap="Blues_r", mask=mask, ax=axs[idxl])
        axs[idxl].set_facecolor("black")
        axs[idxl].set_title('Player %d\'s payoffs, $v_%d=$%d'%(l,l,game.v_r[l]))
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
                p1_opt_act = tuple(sorted(game.optimal_solution['P'+str(game.players[0])+' actions'],
                                          key=lambda x: str(x[0])))
                p2_opt_act = tuple(sorted(game.optimal_solution['P'+str(game.players[1])+' actions'],
                                          key=lambda x: str(x[0])))
                p1_opt_idx = list(pivot_dict).index(p1_opt_act)
                p2_opt_idx = list(pivot_dict.index.values).index(p2_opt_act)
                axs[idxl].add_patch(Rectangle((p1_opt_idx, p2_opt_idx), 1, 1, fill=False,
                                              hatch='xxx', edgecolor='green', lw=0.01))
            except:
                print('Optimal solution is not among the actions:')
                print(game.optimal_solution)
        else:
            print('Optimal solution has not been calculated')
    plt.savefig(plot_dir+'/NE_sol_2D_'+suffix+'.png', dpi=dpi, bbox_inches='tight')
    
def plot_ne_analysis(df, x='t', ci=None):
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
    decision_type = df.decision_type.unique().tolist()
    judgment_type = df.judgment_type.unique().tolist()
    if 'nan' in judgment_type:
        judgment_type.remove('nan')
    auction_type = df.auction_type.unique().tolist()
    # if 'nan' in auction_type:
    #     auction_type.remove('nan')
    valuation_type = df.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    T = len(df[x].unique().tolist())
    row_plot=['payoff_similarity', 'action_similarity', 'payoff_ratio', 'no_ne']
    col_plot = [no_resources, 'no_resources'] # no_resources, judgment_type
    hue_type = [auction_type, 'auction_type'] #auction_type
    style_type = 'auction_type'  #decision_type
    # Initialize plot properties
    dpi = 300
    fig, axs = plt.subplots(len(row_plot), len(col_plot[0]), sharex=True, sharey='row',
                            figsize=(4000/dpi, 3000/dpi))
    colors = ['#154352', '#007268', '#5d9c51', '#dbb539', 'k']
    pal = sns.color_palette(colors[:len(hue_type[0])])
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot):
            ax, _, _ = find_ax(axs, row_plot, col_plot[0], idx_r, idx_c)
            ne_data = df[(df.decision_type.isin(decision_type))&
                         (df[col_plot[1]] == val_c)]
            with pal:
                sns.lineplot(x=x, y=val_r, hue=hue_type[1], style=style_type, markers=True,
                             ci=ci, ax=ax, data=ne_data, **{'markersize':5})
            ax.set(xlabel=r'time step $t$', ylabel= correct_labels([val_r])[0])
            ax.get_legend().set_visible(False)
            ax.xaxis.set_ticks(np.arange(1, T+1, 1.0))#ax.get_xlim()
    # Rebuild legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [x for x in handles if isinstance(x, mplt.lines.Line2D)]
    labels = correct_labels(labels)
    fig.legend(handles, labels, loc='center right', ncol=1, framealpha=0.35,
               bbox_to_anchor=(.84, 0.69))
    # Add overll x- and y-axis titles
    _, axs_c, axs_r = find_ax(axs, row_plot, col_plot[0])
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'$R_c=$%s'%(str(col_plot[0][idx])))
    plt.savefig('ne_analysis_.png', dpi=dpi, bbox_inches='tight')

def plot_ne_cooperation(df, x='t', ci=None):
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
    decision_type = df.decision_type.unique().tolist()
    judgment_type = df.judgment_type.unique().tolist()
    if 'nan' in judgment_type:
        judgment_type.remove('nan')
    auction_type = df.auction_type.unique().tolist()
    if 'nan' in auction_type:
        auction_type.remove('nan')
    valuation_type = df.valuation_type.unique().tolist()
    if 'nan' in valuation_type:
        valuation_type.remove('nan')
    T = len(df[x].unique().tolist())
    row_plot = [auction_type, 'auction_type']
    col_plot = [no_resources, 'no_resources'] # no_resources, judgment_type
    # hue_type = [auction_type, 'auction_type'] #auction_type
    # style_type = 'auction_type'  #decision_type
    # value_vars = ['cooperative', 'partially_cooperative', 'OA', 'NA', 'NA_possible',
    #               'opt_cooperative', 'opt_partially_cooperative', 'opt_OA',
    #               'opt_NA', 'opt_NA_possible']
    # value_vars = ['cooperative', 'partially_cooperative',
    #               'opt_cooperative', 'opt_partially_cooperative']
    value_vars = ['OA', 'NA', 'NA_possible', 'opt_OA',
                  'opt_NA', 'opt_NA_possible']
    id_vars = [x for x in df.columns if x not in value_vars]
    # Initialize plot properties
    dpi = 300
    fig, axs = plt.subplots(len(row_plot[0]), len(col_plot[0]), sharex=True, sharey='row',
                            figsize=(4000/dpi, 3000/dpi))
    # colors = ['#154352', '#007268', '#5d9c51', '#dbb539', 'k']
    # pal = sns.color_palette(colors)
    pal = sns.color_palette()
    for idx_c, val_c in enumerate(col_plot[0]):
        for idx_r, val_r in enumerate(row_plot[0]):
            ax, _, _ = find_ax(axs, row_plot[0], col_plot[0], idx_r, idx_c)
            ne_data = df[(df.decision_type.isin(decision_type))&
                         (df[col_plot[1]] == val_c)&(df[row_plot[1]] == val_r)]
            ne_data = pd.melt(ne_data, id_vars=id_vars, value_vars=value_vars,
                              var_name='Cooperation Status')
            ne_data['Method'] = np.where((ne_data['Cooperation Status'] == 'cooperative')|\
                                        (ne_data['Cooperation Status'] == 'partially_cooperative')|\
                                        (ne_data['Cooperation Status'] == 'OA')|\
                                        (ne_data['Cooperation Status'] == 'NA')|\
                                        (ne_data['Cooperation Status'] == 'NA_possible'),
                                        'INRSG', 'INDP')
            ne_data = ne_data.replace(['opt_cooperative', 'opt_partially_cooperative',
                                       'opt_OA', 'opt_NA', 'opt_NA_possible'],
                                      ['cooperative', 'partially_cooperative', 'OA',
                                       'NA', 'NA_possible'])
            with pal:
                sns.lineplot(x=x, y='value', hue='Cooperation Status', style='Method',
                             markers=True, ci=ci, ax=ax, data=ne_data, **{'markersize':5})
            ax.set(xlabel=r'time step $t$', ylabel= row_plot[0][idx_r])
            ax.get_legend().set_visible(False)
            ax.xaxis.set_ticks(np.arange(1, T+1, 1.0))#ax.get_xlim()
    # Rebuild legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [x for x in handles if isinstance(x, mplt.lines.Line2D)]
    labels = correct_labels(labels)
    fig.legend(handles, labels, loc='center right', ncol=1, framealpha=0.35,
               bbox_to_anchor=(.84, 0.69))
    # Add overll x- and y-axis titles
    _, axs_c, axs_r = find_ax(axs, row_plot[0], col_plot[0])
    for idx, ax in enumerate(axs_c):
        ax.set_title(r'$R_c=$%s'%(str(col_plot[0][idx])))
    plt.savefig('ne_cooperation.png', dpi=dpi, bbox_inches='tight')

def correct_labels(labels):
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
    labels = ['iINDP' if x == 'indp_sample_12Node' else x for x in labels]
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
    labels = ['Dynamic Param. iINDP' if x == 'dp_indp' else x for x in labels]
    labels = ['Optimal' if x == 'nan' else x for x in labels]#!!!
    labels = ['td-INDP' if x == 'tdindp' else x for x in labels]
    labels = ['Judge. Call' if x == 'jc' else x for x in labels]
    labels = ['Judge. Call' if x == 'jc_sample_12Node' else x for x in labels]
    labels = ['Normal Game' if x == 'ng' else x for x in labels]
    labels = ['Normal Game' if x == 'ng_sample_12Node' else x for x in labels]
    labels = ['Total Resources' if x == 'no_resources' else x for x in labels]
    labels = ['Total Resources' if x == 'no_resource' else x for x in labels]
    labels = ['Payoff Similarity' if x == 'payoff_similarity' else x for x in labels]
    labels = ['Action Similarity' if x == 'action_similarity' else x for x in labels]
    labels = ['Payoff Ratio' if x == 'payoff_ratio' else x for x in labels]
    labels = ['\# NE' if x == 'no_ne' else x for x in labels]
    labels = ['Cooperative' if x == 'cooperative' else x for x in labels]
    labels = ['Par. Cooperative' if x == 'partially_cooperative' else x for x in labels]
    labels = ['Non Cooperative (OA)' if x == 'OA' else x for x in labels]
    labels = ['Non Cooperative (NA)' if x == 'NA' else x for x in labels]
    labels = ['No More Actions (NA)' if x == 'NA_possible' else x for x in labels]
    labels = ['Return Period' if x == 'return_period' else x for x in labels]
    for idx, x in enumerate(labels):
        if 'b' in x and 't' in x:
            splited_x = x[1:].split('t')
            labels[idx] = 'b='+str(splited_x[0])+' c-t='+str(splited_x[1])
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