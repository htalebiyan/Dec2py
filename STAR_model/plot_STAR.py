import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import cPickle as pickle
from os import listdir
from os.path import isfile, join

def add_labels(df, col):
    rename_dict={"Intercept": r'$\alpha$',
                 "w_n_t_1": r'$\beta_{1,i}$ (Neighbor nodes)',
                 "w_a_t_1": r'$\beta_{2,i}$ (Connected arcs)',
                 "w_d_t_1": r'$\beta_{3,i}$ (Dependee nodes)',
                 "y_n_t_1": r'$\beta_{3,i}$ (End nodes)',
                 "w_c_t_1": r'$\beta_{4,i}$ (All nodes)',
                 "y_c_t_1": r'$\beta_{5,i}$ (All arcs)',
                 "Node": r'$\gamma_{1,i}$ (Node cost)',
                 "Arc": r'$\gamma_{2,i}$ (Arc cost)',
                 "Under_Supply": r'$\gamma_{3,i}$ (Demand deficit)',
                 "Flow": r'$\gamma_{4,i}$ (Flow cost)',
                 "Rc": r'$R_c$ (Resource)'}
    df['Predictors']=df[col]
    for i in df[col].unique():
        df['Predictors']=df['Predictors'].replace(i,rename_dict[i])
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
    corr_cov= pd.DataFrame().reindex_like(corr_matrices[key])      
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
                corr_cov[v][vv]=0.0
                if mean!=0.0:
                    corr_cov[v][vv]=variance**0.5/mean
            
            
    plot_df = corr_mean.drop(exclusions, axis=1).drop(exclusions, axis=0)
    #node_data[key] #corr_mean #corr_cov
    # sns.pairplot(node_data[key].drop(columns=[x for x in exclusions if x not in ['y_t_1','index']]),
    #     kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    plt.figure()
    sns.heatmap(plot_df,annot=True, center=0, cmap="vlag",linewidths=.75)
    sns.clustermap(plot_df, cmap="vlag")

def plot_results():   
    sns.set(context='notebook',style='darkgrid')
    # plt.rc('text', usetex=True)
    # plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.close('all')
              
    t_suf = '20200429'    
    folder_name = 'results'+t_suf
    
    cols_results=['sample','Time Step','resource_cap','pred_sample','Result Type','Total Cost',
                  'run_time','performance']    
    tc_df=pd.read_csv(folder_name+'/results'+t_suf+'.txt',delimiter='\t',
                          names=cols_results,index_col=False) 
    
    cols_results=['sample','element','resource_cap','pred_sample','Optimal Repair Time',
                  'predicted_repair_time','Prediction Error']    
    pe_df=pd.read_csv(folder_name+'/pred_error'+t_suf+'.txt',delimiter='\t',
                          names=cols_results,index_col=False)
    
    cols_results=['sample','Time Step','resource_cap','pred_sample','real_repair_perc',
                  'predicted_repair_perc'] 
    rp_df=pd.read_csv(folder_name+'/rep_prec'+t_suf+'.txt',delimiter='\t',
                          names=cols_results,index_col=False)   
    
    ###############################################################################
    tc_df=tc_df.replace('predicted','Logistic Model Prediction')    
    tc_df=tc_df.replace('data','Optimal Scenario')               
    figure_df = tc_df#[result_df['sample']==200]
    sns.lineplot(x="Time Step", y="Total Cost",style='Result Type',hue='Result Type',
                 data=figure_df,markers=True,ci=99)
    plt.savefig('Total_cost_vs_time.png',dpi=600,bbox_inches='tight')
    
    ###############################################################################
    f=plt.figure()                   
    figure_df = pe_df#[result_df['sample']==200]
    for index, row in figure_df.iterrows():
        if row['element'][0]=='y':
            figure_df=figure_df.drop(index=index)
    figure_df=figure_df.reset_index()
    p2=sns.boxplot(x="Optimal Repair Time", y="Prediction Error",data=figure_df,
                   whis=1, linewidth=0.75, fliersize=3,palette=sns.cubehelix_palette(20))
    p2.text(0.15,0.82, "Positive error: rush\nNegative error: lag",ha="left", va="center",
            transform = f.transFigure,bbox=dict(boxstyle="round",fc=(.88,.88,.91), ec=(.8,.8,.8)),
             fontsize=11)
    plt.savefig('prediction_error_vs_time.png',dpi=600,bbox_inches='tight')
    
    f=plt.figure() 
    p3=sns.distplot(figure_df['Prediction Error'], kde=False, rug=False,norm_hist=False,
                    bins=np.arange(-10,10,1)-0.5, color=(.41,.26,.34))
    p3.set_ylabel('Frequency')
    p3.set_xticks(np.arange(-10,10, 2.0)) #p3.axes.get_xlim()
    p3.text(0.15,0.82, "Positive error: rush\nNegative error: lag",ha="left", va="center",
            transform = f.transFigure,bbox=dict(boxstyle="round",fc=(.88,.88,.91), ec=(.8,.8,.8)),
             fontsize=11)
    plt.savefig('prediction_error_hist.png',dpi=600,bbox_inches='tight')
    ###############################################################################
    plt.figure()                    
    figure_df = pd.melt(rp_df, id_vars=['sample','Time Step','resource_cap','pred_sample'],
                        value_vars=['real_repair_perc','predicted_repair_perc'])#[result_df['sample']==200]
    figure_df=figure_df.replace('predicted_repair_perc','Logistic Model Prediction')    
    figure_df=figure_df.replace('real_repair_perc','Optimal Scenario') 
    figure_df=figure_df.rename(columns={'variable':'Result Type','value':'Repaired Percentage'})
    g=sns.lineplot(x="Time Step", y="Repaired Percentage",style='Result Type',hue='Result Type',
                 data=figure_df,palette="muted",markers=True,ci=99)  
    g.legend(loc=2)
    plt.savefig('repaired_element.png',dpi=600,bbox_inches='tight')
    ###############################################################################
    
'''Analyze the coefficients''' 
def read_coef():
    t_suf = ''#'20200505'
    root='C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc/'
    mypath=root+'parameters'+t_suf+'/'
    keys = [f[17:-4] for f in listdir(mypath) if isfile(join(mypath, f)) and f[:2]!='R2']
    node_params=pd.DataFrame(columns=['key','name','mean','sd','mc_error','hpd_2.5','hpd_97.5','n_eff','Rhat'])
    arc_params=pd.DataFrame(columns=['key','name','mean','sd','mc_error','hpd_2.5','hpd_97.5','n_eff','Rhat'])
    for key in keys:
        if key[0]=='w':
            filename= mypath+'model_parameters_'+key+'.txt'
            paramters = result_df=pd.read_csv(filename,delimiter=' ',index_col=False) 
            paramters=paramters.rename(columns={'Unnamed: 0': 'name'})
            paramters['key']=key
            node_params=pd.concat([node_params,paramters], axis=0, ignore_index=True)
        if key[0]=='y':
            filename= mypath+'model_parameters_'+key+'.txt' 
            paramters = result_df=pd.read_csv(filename,delimiter=' ',index_col=False) 
            paramters=paramters.rename(columns={'Unnamed: 0': 'name'})
            paramters['key']=key
            arc_params=pd.concat([arc_params,paramters], axis=0, ignore_index=True)
            
    node_params['CoV'] =abs(node_params['sd']/node_params['mean'])
    node_params_filtered=node_params[(node_params['mc_error']<0.1) &
                                     (abs(node_params['Rhat']-1)<0.01) &
                                     (node_params['CoV']<1)]
    node_params_filtered=add_labels(node_params_filtered, 'name')
    node_params_filtered=node_params_filtered.rename(columns={'mean':'Estimated Mean'})
    arc_params['CoV'] =abs(arc_params['sd']/arc_params['mean'])
    arc_params_filtered=arc_params[(arc_params['mc_error']<0.1) &
                                   (abs(arc_params['Rhat']-1)<0.01) &
                                   (arc_params['CoV']<1)]
    arc_params_filtered=add_labels(arc_params_filtered, 'name')
    arc_params_filtered=arc_params_filtered.rename(columns={'mean':'Estimated Mean'})     
    return arc_params,arc_params_filtered,node_params,node_params_filtered

def plot_coef(arc_params_filtered,node_params_filtered):
    sns.set(context='notebook',style='darkgrid')
    # plt.rc('text', usetex=True)
    # plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.close('all')
           
    f, axs = plt.subplots(2, 1,sharex=True,gridspec_kw={'height_ratios': [11/20.0,9/20.0]})
    sns.boxplot(y="Predictors", x="Estimated Mean",data=node_params_filtered, whis=1, ax=axs[0],
                    linewidth=0.5, fliersize=.5, palette=sns.cubehelix_palette(20))
    sns.boxplot(y="Predictors", x="Estimated Mean",data=arc_params_filtered, whis=1, ax=axs[1],
                    linewidth=0.5, fliersize=.5, palette=sns.cubehelix_palette(20))
    axs[0].set_ylabel('Predictors - Nodes')
    axs[0].set_xlabel('')
    axs[1].set_ylabel('Predictors - Arcs')
    axs[1].set_xlim((-25,25))
    axs[1].set_xticks(np.arange(-25,25, 5.0))
    plt.savefig('mean_predictors.png',dpi=600,bbox_inches='tight')

    f, axs = plt.subplots(2, 1, sharex=True,gridspec_kw={'height_ratios': [11/20.0,9/20.0]})
    sns.boxplot(y="Predictors", x="CoV",data=node_params_filtered, whis=1, ax=axs[0],
                    linewidth=0.5, fliersize=.5, palette=sns.cubehelix_palette(20))
    sns.boxplot(y="Predictors", x="CoV",data=arc_params_filtered, whis=1, ax=axs[1],
                    linewidth=0.5, fliersize=.5, palette=sns.cubehelix_palette(20))
    axs[0].set_ylabel('Predictors - Nodes')
    axs[0].set_xlabel('')
    axs[1].set_ylabel('Predictors - Arcs')
    axs[1].set_xlim((-0,1))
    axs[1].set_xticks(np.arange(-0,1, 0.1))
    plt.savefig('cov_predictors.png',dpi=600,bbox_inches='tight')   

# arc_params,arc_params_filtered,node_params,node_params_filtered=read_coef()
# plot_coef(arc_params_filtered,node_params_filtered)
    
# sns.set(style="white", color_codes=True)
# g=sns.jointplot('Estimated Mean', 'CoV', data=arc_params_filtered,
#                   kind="kde", space=0, color="b",xlim=(-25,25),ylim=(-0,1))
# sns.set(style="white", color_codes=True)
# g=sns.jointplot('Estimated Mean', 'CoV', data=node_params_filtered,
#                   kind="kde", space=0, color="b",xlim=(-25,25),ylim=(-0,1))

# df = pd.merge(node_params, node_params_filtered, on=list(node_params.columns), 
              # how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
# for i in df['key'].unique():
#     print train_data[i]['w_t'].mean()
# print len(df['key'].unique())

# sns.pairplot(node_params_filtered.drop(columns=['hpd_2.5','hpd_97.5']),
#     kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})