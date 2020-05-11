import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import cPickle as pickle
from os import listdir
from os.path import isfile, join

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
def plot_coef():
    sns.set(context='notebook',style='darkgrid')
    # plt.rc('text', usetex=True)
    # plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.close('all')
    
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
    # node_params_filtered=node_params[(node_params['mc_error']<0.1) & (abs(node_params['Rhat']-1)<0.005)]
    # node_params_filtered=replace_labels(node_params_filtered, 'name')
    # node_params_filtered=node_params_filtered.rename(columns={'name':'Predictors','mean':'Estimated Mean'})
    
    # ax=sns.boxplot(y="Predictors", x="Estimated Mean",data=node_params_filtered, whis=1,
    #                linewidth=0.75, fliersize=3, palette=sns.cubehelix_palette(20))
    # ax.set_xlim((-25,25))
    # ax.set_xticks(np.arange(-25,25, 5.0))
    # plt.savefig('Node_mean_predictors.png',dpi=600,bbox_inches='tight')
    # # plt.figure()    
    arc_params['cov'] =abs(arc_params['sd']/arc_params['mean'])
    # # ax=sns.boxplot(y="name", x="mean",data=arc_params, whis=1,fliersize=1,linewidth=1,
    # #                 palette=sns.cubehelix_palette(25))
    # # ax.set_xlim((-25,25))
        
    # # sns.pairplot(node_params_filtered.drop(columns=['hpd_2.5','hpd_97.5']),
    # #     kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
    return node_params,arc_params

# node_params,arc_params=plot_coef()



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