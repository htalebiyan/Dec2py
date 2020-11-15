import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from os import listdir
from os.path import isfile, join
sns.set(context='notebook',style='darkgrid')
# plt.rc('text', usetex=True)
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.close('all')

def plot_df(mag_num, sample_num, num_layers, res, pred_results, pred_error,
            rep_prec, costs_opt, cost_type = 'Total'):
    """Prepares the dataframes to plot results"""
    ### Read data ###
    subfolder_results = 'indp_results_L'+str(num_layers)+'_m'+str(mag_num)+'_v'+str(res)
    pred_error['magnitude'] = mag_num
    pred_error['sample'] = sample_num
    pred_error['Rc'] = res
    rep_prec['rep_prec'] = mag_num
    rep_prec['rep_prec'] = sample_num
    rep_prec['Rc'] = res
    cost_df = pd.DataFrame(columns=['magnitude', 'sample', 'Rc', 't',
                                    'pred sample', 'type', 'cost'])
    T = len(pred_results[0].results.keys())
    for t in range(T):
        opt_result = costs_opt[cost_type]
        temp_dict = {'magnitude':mag_num, 'sample':sample_num, 'Rc':res,
                     't':t, 'pred sample':-1, 'type':'optimal', 'cost':opt_result[t]}
        cost_df = cost_df.append(temp_dict, ignore_index=True)
        for pred_s in pred_results:
            if opt_result[0] != pred_results[pred_s].results[0]['costs'][cost_type]:
                sys.exit('Error: Unmatched intial cost')
            temp_dict = {'magnitude':mag_num, 'sample':sample_num, 'Rc':res,
                         't':t, 'pred sample':pred_s, 'type':'predicted',
                         'cost':pred_results[pred_s].results[t]['costs'][cost_type]}
            cost_df = cost_df.append(temp_dict, ignore_index=True)
    return cost_df, pred_error, rep_prec

def replace_labels(df, col):
    rename_dict={"Intercept": r'$\alpha$',
                 "w_n_t_1": r'$\beta_{1,i}$ (Neighbor nodes)',
                 "w_a_t_1": r'$\beta_{2,i}$ (Connected arcs)',
                 "w_d_t_1": r'$\beta_{3,i}$ (Dependee nodes)',
                 "w_c_t_1": r'$\beta_{4,i}$ (All nodes)',
                 "y_c_t_1": r'$\beta_{5,i}$ (All arcs)',
                 "Node": r'$\gamma_{1,i}$ (Node cost)',
                 "Arc": r'$\gamma_{2,i}$ (Arc cost)',
                 "Under_Supply": r'$\gamma_{3,i}$ (Demand deficit)',
                 "Flow": r'$\gamma_{4,i}$ (Flow cost)',
                 "Rc": r'$R_c$ (Resource)'}
    for i in df[col].unique():
        df=df.replace(i,rename_dict[i])
    return df

def plot_correlation(node_data,keys,exclusions):
    import math
    corr_matrices={}
    for key in keys:
        if key[0]=='w':
            corr_matrices[key]=node_data[key].drop([], axis=1).corr()
        # if key[0]=='y':
        #     corr_matrices[key]=arc_data[key].drop([], axis=1).corr()
    corr_mean= pd.DataFrame().reindex_like(corr_matrices[key])
    corr_std= pd.DataFrame().reindex_like(corr_matrices[key])
    for v in list(node_data[keys[0]].columns):
        for vv in list(node_data[keys[0]].columns):
            corr_vec = []
            for key in keys:
                if key[0]=='w':
                    if math.isnan(corr_matrices[key][v][vv]):
                        pass
                    else:
                        corr_vec.append(corr_matrices[key][v][vv])
            if len(corr_vec):
                mean = sum(corr_vec)/len(corr_vec)
                variance = sum([((x-mean)**2) for x in corr_vec])/len(corr_vec)
                corr_mean[v][vv]=mean
                corr_std[v][vv]=variance**0.5

    plot_df = corr_mean.drop(exclusions, axis=1).drop(exclusions, axis=0)
    #node_data[key] #corr_mean #corr_std
    # sns.pairplot(node_data[key].drop(columns=[x for x in exclusions if x not in ['y_t_1','index']]),
    #     kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.figure()
    sns.heatmap(plot_df,annot=True, center=0, cmap="vlag",linewidths=.75)
    sns.clustermap(plot_df, cmap="vlag")

def plot_cost(df):
    df = df.replace('predicted','Logistic Model Prediction')
    df = df.replace('data','Optimal Scenario')
    FIGURE_DF = df
    sns.lineplot(x="t", y='cost', style='Rc', hue='type', data=FIGURE_DF, markers=True, ci=95)
    plt.savefig('Total_cost_vs_time.png',dpi = 600, bbox_inches='tight')

def plot_results(pe_df, rp_df):
    f=plt.figure()
    figure_df = pe_df[(pe_df['real rep time']!=0)|(pe_df['pred rep time']!=0)]
    figure_df=figure_df.reset_index()
    p2=sns.boxplot(x="real rep time", y="prediction error",data=figure_df,
                   whis=1, linewidth=0.75, fliersize=3,palette=sns.cubehelix_palette(20))
    p2.text(0.15,0.82, "Positive error: rush\nNegative error: lag",ha="left", va="center",
            transform = f.transFigure,bbox=dict(boxstyle="round",fc=(.88,.88,.91), ec=(.8,.8,.8)),
             fontsize=11)
    plt.savefig('prediction_error_vs_time.png',dpi=600,bbox_inches='tight')

    f=plt.figure()
    p3=sns.distplot(figure_df['prediction error'], kde=False, rug=False,norm_hist=False,
                    bins=np.arange(-10,10,1)-0.5, color=(.41,.26,.34))
    p3.set_ylabel('Frequency')
    p3.set_xticks(np.arange(-10,10, 2.0)) #p3.axes.get_xlim()
    p3.text(0.15,0.82, "Positive error: rush\nNegative error: lag",ha="left", va="center",
            transform = f.transFigure,bbox=dict(boxstyle="round",fc=(.88,.88,.91), ec=(.8,.8,.8)),
             fontsize=11)
    plt.savefig('prediction_error_hist.png',dpi=600,bbox_inches='tight')
    ###############################################################################
    plt.figure()
    figure_df = pd.melt(rp_df, id_vars=['sample','t','Rc','pred sample'],
                        value_vars=['real rep prec','pred rep prec'])#[result_df['sample']==200]
    figure_df=figure_df.replace('pred rep prec','Logistic Model Prediction')
    figure_df=figure_df.replace('real rep prec','Optimal Scenario')
    figure_df=figure_df.rename(columns={'variable':'Result Type','value':'Repaired Percentage'})
    g=sns.lineplot(x="t", y="Repaired Percentage",style='Result Type',hue='Result Type',
                 data=figure_df,palette="muted",markers=True,ci=95)
    g.legend(loc=2)
    plt.savefig('repaired_element.png',dpi=600,bbox_inches='tight')
    ###############################################################################
'''R2'''
def plot_R2():
    t_suf = ''
    root = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc/'
    mypath=root+'parameters'+t_suf+'/R2.txt'
    plt.figure()
    cols=['node','layer','type','i','j','$R^2$']
    r2 = result_df=pd.read_csv(mypath,delimiter=' ',index_col=False, header=None,
                               nrows=334, names=cols)
    figure_data = r2[r2['type']=='Test']['$R^2$']
    ax = sns.distplot(figure_data, kde=False, rug=True, norm_hist=True, color='#4a266a')
    ax.axvline(figure_data.mean(), color='#aacfd0', linestyle='--', label='Mean')
    ax.axvline(figure_data.median(), color='#7f4a88', linestyle='-', label='Median')
    ax.legend()
    ax.set_ylabel('Probability')
    plt.savefig('R2_node.png',dpi=600,bbox_inches='tight')
    return figure_data
    ###############################################################################
'''Analyze the coefficients'''
def plot_coef():
    # t_suf = '20200415'
    # samples,costs,costs_local,initial_net = pickle.load(open('data'+t_suf+'/initial_data.pkl', "rb" ))
    # keys=samples.keys()

    t_suf = ''
    root = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc/'
    mypath=root+'parameters'+t_suf+'/'
    keys = [f[17:-4] for f in listdir(mypath) if isfile(join(mypath, f)) and f[:2]!='R2']
    plt.figure()
    node_params=pd.DataFrame(columns=['key','name','mean','sd','mc_error','hpd_2.5','hpd_97.5','n_eff','Rhat'])
    arc_params=pd.DataFrame(columns=['key','name','mean','sd','mc_error','hpd_2.5','hpd_97.5','n_eff','Rhat'])
    for key in keys:
        if key[0]=='w':
            filename= mypath+'/model_parameters_'+key+'.txt'
            paramters = result_df=pd.read_csv(filename,delimiter=' ',index_col=False)
            paramters=paramters.rename(columns={'Unnamed: 0': 'name'})
            paramters['key']=key
            node_params=pd.concat([node_params,paramters], axis=0, ignore_index=True)
        if key[0]=='y':
            filename= mypath+'/model_parameters_'+key+'.txt'
            try:
                paramters = result_df=pd.read_csv(filename,delimiter=' ',index_col=False)
                paramters=paramters.rename(columns={'Unnamed: 0': 'name'})
                paramters['key']=key
                arc_params=pd.concat([arc_params,paramters], axis=0, ignore_index=True)
            except:
                pass
    node_params['cov'] =abs(node_params['sd']/node_params['mean'])
    node_params_filtered=node_params[(node_params['mc_error']<0.01) & (abs(node_params['Rhat']-1)<0.005)]
    node_params_filtered=replace_labels(node_params_filtered, 'name')
    node_params_filtered=node_params_filtered.rename(columns={'name':'Predictors','mean':'Estimated Mean'})

    ax=sns.boxplot(y="Predictors", x="Estimated Mean",data=node_params_filtered, whis=1,
                    linewidth=0.75, fliersize=3, palette=sns.cubehelix_palette(20))
    ax.set_xlim((-25,25))
    ax.set_xticks(np.arange(-25,25, 5.0))
    plt.savefig('Node_mean_predictors.png',dpi=600,bbox_inches='tight')
    # # plt.figure()
    arc_params['cov'] =abs(arc_params['sd']/arc_params['mean'])
    # # ax=sns.boxplot(y="name", x="mean",data=arc_params, whis=1,fliersize=1,linewidth=1,
    # #                 palette=sns.cubehelix_palette(25))
    # # ax.set_xlim((-25,25))

    # # sns.pairplot(node_params_filtered.drop(columns=['hpd_2.5','hpd_97.5']),
    # #     kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    return node_params,arc_params
    ###############################################################################


# node_params,arc_params=plot_coef()
# plot_results()
# r2 = plot_R2()

# rmv_key=arc_params[(abs(arc_params['Rhat']-1)>.005)|(arc_params['mc_error']>0.05)]['key'].unique()
# keys=[x for x in train_data.keys() if x not in rmv_key]
# data=pd.DataFrame(index=keys)
# for key in keys:
#     temp=arc_params[arc_params['key']==key]
#     data.loc[key,'train_mean']=train_data[key]['y_t'].mean()
#     data.loc[key,'mean']=temp['mean'].mean()
#     data.loc[key,'Rhat']=temp['Rhat'].mean()-1
#     data.loc[key,'mc_error']=temp['mc_error'].mean()
#     data.loc[key,'sd']=temp['sd'].mean()
#     data.loc[key,'cov']=temp['cov'].mean()
#     data.loc[key,'ratio']=float(temp[(temp['name']=='Rc')]['mean'])#/max(abs(temp['mean'])))
# corr=data.corr()
# sns.heatmap(corr,annot=True, center=0, cmap="vlag",linewidths=.75)