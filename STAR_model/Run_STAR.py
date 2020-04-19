from STAR_utils import *
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import cPickle as pickle
from functools import partial
import indp 
import copy

if __name__ == "__main__": 
    t_suf = '20200415'
    samples,costs,costs_local,initial_net = pickle.load(open('data'+t_suf+'/initial_data.pkl', "rb" ))     
    # train_data,test_data = pickle.load(open('data'+t_suf+'/train_test_data.pkl', "rb" ))     
            
    plt.close('all')
    base_dir = "../data/Extended_Shelby_County/"
    damage_dir = "../data/random_disruption_shelby/"
    output_dir = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/'   
    failSce_param = {"type":"random","sample_range":range(50,550),"mags":range(0,1),
                    'filtered_List':None,'Base_dir':base_dir,'Damage_dir':damage_dir}
    v_r = [4]#,10,20,30,40,50,60,70,80,90,100]
    layers=[1,2,3,4]
 
    ''' Read all data '''
    # sample_all={}
    # costs_all={}
    # feature_all= {}
    # for res in v_r:
    #     params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'results/indp_results',
    #             "V":res,"ALGORITHM":"INDP"}
        
    #     samples,costs,costs_local,initial_net,_,_=importData(params,failSce_param,layers) 
    #     # sample_all[res]=samples
    #     # costs_all[res]=costs
    # save_initial_data(initial_net,samples,costs,costs_local)

    ''' Prepare training and testing datsets '''
    from os import listdir
    from os.path import isfile, join
    mypath='parameters20200419/'
    files = [f[17:-4] for f in listdir(mypath) if isfile(join(mypath, f))]
    keys= [x for x in samples.keys() if (x[0]=='w' and x not in files)] 
    # keys = ['w_(1, 4)']   #samples.keys()   #['y_(2, 2),(11, 2)','y_(11, 2),(2, 2)'] 
    
    # node_data,arc_data = prepare_data(samples,costs,costs_local,initial_net,keys)
    # train_data,test_data = train_test_split(node_data,arc_data,keys)
    # save_prepared_data(train_data,test_data)
    
    ''' train and test model'''
    exclusions=['w_t_1','w_h_t_1','y_t_1','time',
                'Total','Under_Supply_Perc','Over_Supply','Space_Prep',
                'Arc_layer','Node_layer','Under_Supply_layer','Flow_layer',
                'Total_layer','Under_Supply_Perc_layer','Over_Supply_layer',
                'Space_Prep_layer'] 
    # ##,'w_n_t_1','w_a_t_1', 'w_d_t_1','time','w_h_t_1','Total','Flow','Under_Supply_layer'
    # for key in keys:  
    #     print '\n'+key
    #     trace,model = train_model({key:train_data[key]},exclusions) 
    #     save_traces(trace)
    #     _,_ = test_model({key:train_data[key]},{key:test_data[key]},
    #                       trace,model,exclusions,plot=False)
    
    # plot_correlation(node_data,keys,exclusions)
    
    '''compare restoration plans'''  
    test_samples =range(250,450,10) #[50,100,150,200,250,300,350,400]
    no_prediction_samples=1
    t_suf = '20200419'
    model_folder='./traces'+t_suf
    param_folder ='./parameters'+t_suf
    for res in v_r:
        for s in test_samples:#failSce_param['sample_range']:
            print '\nSample '+`s`,
            for pred_s in range(0,no_prediction_samples):
                network_object = copy.deepcopy(initial_net[0])
                indp.add_random_failure_scenario(network_object,DAM_DIR=damage_dir,sample=s)
                compare_resotration(pred_s,samples,costs,costs_local,s,res,network_object,failSce_param,
                                    initial_net,layers,output_dir,model_folder,param_folder)    
                
    # t_suf = '20200419'    

    # cols_results=['sample','time','resource_cap','pred_sample','result_type','cost',
    #               'run_time','performance']    
    # folder_name = 'results'+t_suf
    # result_df=pd.read_csv(folder_name+'/results'+t_suf+'.txt',delimiter='\t',
    #                       names=cols_results,index_col=False)                   
    # figure_df = result_df#[result_df['sample']==200]
    # sns.lineplot(x="time", y="cost",style='result_type',
    #               data=figure_df[figure_df['result_type']=='predicted'])
    # sns.lineplot(x="time", y="cost",data=figure_df[figure_df['result_type']=='data'])

    # cols_results=['sample','element','resource_cap','pred_sample','real_repair_time',
    #               'predicted_repair_time','prediction_error']    
    # folder_name = 'results'+t_suf
    # result_df=pd.read_csv(folder_name+'/pred_error'+t_suf+'.txt',delimiter='\t',
    #                       names=cols_results,index_col=False)
    # plt.figure()                   
    # figure_df = result_df#[result_df['sample']==200]
    # for index, row in figure_df.iterrows():
    #     if row['element'][0]=='y':
    #         figure_df=figure_df.drop(index=index)
    # figure_df=figure_df.reset_index()
    # sns.boxplot(x="real_repair_time", y="prediction_error",data=figure_df,palette="muted", whis=1)
    # plt.ylim((-10,10))
    # plt.figure() 
    # sns.distplot(figure_df['prediction_error'], kde=False, rug=False)
    # plt.xlim((-15,15))

    # cols_results=['sample','time','resource_cap','pred_sample','real_repair_perc',
    #               'predicted_repair_perc'] 
    # plt.figure()    
    # folder_name = 'results'+t_suf
    # result_df=pd.read_csv(folder_name+'/rep_prec'+t_suf+'.txt',delimiter='\t',
    #                       names=cols_results,index_col=False)                   
    # figure_df = result_df#[result_df['sample']==200]
    # sns.lineplot(x="time", y="real_repair_perc",data=figure_df,palette="muted",ci=95)
    # sns.lineplot(x="time", y="predicted_repair_perc",data=figure_df,palette="muted",ci=95)  
    
    '''Analyze the coefficients''' 
    # t_suf = '20200419'
    # keys = samples.keys() 
    # node_params=pd.DataFrame(columns=['key','name','mean','sd','mc_error','hpd_2.5','hpd_97.5','n_eff','Rhat'])
    # arc_params=pd.DataFrame(columns=['key','name','mean','sd','mc_error','hpd_2.5','hpd_97.5','n_eff','Rhat'])
    # for key in keys:
    #     if key[0]=='w':
    #         filename= 'parameters'+t_suf+'/model_parameters_'+key+'.txt'
    #         paramters = result_df=pd.read_csv(filename,delimiter=' ',index_col=False) 
    #         paramters=paramters.rename(columns={'Unnamed: 0': 'name'})
    #         paramters['key']=key
    #         node_params=pd.concat([node_params,paramters], axis=0, ignore_index=True)
    #     if key[0]=='y':
    #         filename= 'parameters'+t_suf+'/model_parameters_'+key+'.txt'
    #         try: 
    #             paramters = result_df=pd.read_csv(filename,delimiter=' ',index_col=False) 
    #             paramters=paramters.rename(columns={'Unnamed: 0': 'name'})
    #             paramters['key']=key
    #             arc_params=pd.concat([arc_params,paramters], axis=0, ignore_index=True)
    #         except:
    #             pass
    # node_params['cov'] =abs(node_params['sd']/node_params['mean'])
    # node_params_filtered=node_params[(node_params['mc_error']<0.1) & (abs(node_params['Rhat']-1)<0.005)]
    # node_params_filtered=replace_labels(node_params_filtered, 'name')
    # ax=sns.boxplot(y="name", x="mean",data=node_params_filtered, whis=1,fliersize=1,linewidth=1,
    #                 palette=sns.cubehelix_palette(25))
    # ax.set_xlim((-15,15))
    
    # plt.figure()    
    # arc_params['cov'] =abs(arc_params['sd']/arc_params['mean'])
    # ax=sns.boxplot(y="name", x="mean",data=arc_params, whis=1,fliersize=1,linewidth=1,
    #                 palette=sns.cubehelix_palette(25))
    # ax.set_xlim((-25,25))
        
    # sns.pairplot(node_params_filtered.drop(columns=['hpd_2.5','hpd_97.5']),
    #     kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})