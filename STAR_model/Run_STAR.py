import STAR_utils
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import cPickle as pickle
from functools import partial
import indp 
import copy
if __name__ == "__main__":  
    plt.close('all')
    base_dir = "../data/Extended_Shelby_County/"
    damage_dir = "../data/random_disruption_shelby/"
    output_dir = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/'   
    failSce_param = {"type":"random","sample_range":range(50,500),"mags":range(0,1),
                    'filtered_List':None,'Base_dir':base_dir,'Damage_dir':damage_dir}
    v_r = [4]#,10,20,30,40,50,60,70,80,90,100]
    layers=[1,2,3,4]
    
    # ''' Read all data '''
    # sample_all={}
    # feature_all= {}
    # for res in v_r:
    #     params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'results/indp_results',
    #             "V":res,"ALGORITHM":"INDP"}
        
    #     samples,initial_net,_,_=STAR_utils.importData(params,failSce_param,layers) 
    #     sample_all[res]=samples
    # STAR_utils.save_initial_data(initial_net,samples)
    
    # t_suf = ''
    # samples,initial_net = pickle.load(open('data'+t_suf+'/initial_data.pkl', "rb" ))     
    
    ''' Prepare training and testing datsets '''
    keys=['w_(1, 1)','y_(2, 2),(11, 2)','y_(11, 2),(2, 2)']#,'w_(11, 2)','w_(22, 3)','w_(8, 4)'] #$samples.keys() #,'y_(2, 2),(11, 2)','y_(11, 2),(2, 2)'
    node_data,arc_data = STAR_utils.prepare_data(samples,initial_net,keys)
    train_data,test_data = STAR_utils.train_test_split(node_data,arc_data,keys)
    STAR_utils.save_prepared_data(train_data,test_data)
    
    # for key in keys:
    #     sns.pairplot(node_data[key].drop(columns=['sample','w_t_1']))
    
    # ''' train and test model'''        
    # trace,model = STAR_utils.train_model(train_data) 
    # STAR_utils.save_traces(trace)
    # ppc,ppc_test = STAR_utils.test_model(train_data,test_data,trace,model)

    '''compare restoration plans'''  
    # test_samples = [50,100,150,200,250,300,350,400,450]
    # no_prediction_samples=5
    # for res in v_r:
    #     for s in test_samples:#failSce_param['sample_range']:
    #         print '\nSample '+`s`,
    #         network_object = copy.deepcopy(initial_net[0])
    #         indp.add_random_failure_scenario(network_object,DAM_DIR=damage_dir,sample=s)
    #         for pred_s in range(no_prediction_samples):
    #             STAR_utils.compare_resotration(samples,s,res,network_object,
    #                       failSce_param,initial_net,layers,output_dir,pred_s)    
        
    # cols_results=['sample','time','resource_cap','pred_sample','result_type','cost',
    #               'run_time','performance']
    # t_suf = '20200323'
    # folder_name = 'results'+t_suf
    # result_df=pd.read_csv(folder_name+'/results'+t_suf+'.txt',delimiter='\t',
    #                       names=cols_results,index_col=False)                   
    # figure_df = result_df[result_df['sample']==300]
    # sns.lineplot(x="time", y="cost", hue='result_type',style='result_type',data=figure_df)
    # sns.lineplot(x="time", y="cost",data=figure_df[figure_df['result_type']=='data'])

    # '''Analyze the coefficients''' 
    # keys = samples.keys() 
    # node_params=pd.DataFrame(columns=['key','name','mean','sd','mc_error','hpd_2.5','hpd_97.5','n_eff','Rhat'])
    # arc_params=pd.DataFrame(columns=['key','name','mean','sd','mc_error','hpd_2.5','hpd_97.5','n_eff','Rhat'])
    # for key in keys:
    #     if key[0]=='w':
    #         filename= 'parameters/model_parameters_'+key+'.txt'
    #         paramters = result_df=pd.read_csv(filename,delimiter=' ',index_col=False) 
    #         paramters=paramters.rename(columns={'Unnamed: 0': 'name'})
    #         paramters['key']=key
    #         node_params=pd.concat([node_params,paramters], axis=0, ignore_index=True)
    #     if key[0]=='y':
    #         filename= 'parameters/model_parameters_'+key+'.txt'
    #         try: 
    #             paramters = result_df=pd.read_csv(filename,delimiter=' ',index_col=False) 
    #             paramters=paramters.rename(columns={'Unnamed: 0': 'name'})
    #             paramters['key']=key
    #             arc_params=pd.concat([arc_params,paramters], axis=0, ignore_index=True)
    #         except:
    #             pass
    # ax=sns.boxplot(x="name", y="mean",data=node_params,
    #                 palette="muted", whis=1.5)
    # ax.set_ylim((-10,10))
    
    # plt.figure()    
    # ax1=sns.boxplot(x="name", y="mean",data=arc_params,
    #                 palette="muted", whis=1.5)
    # ax1.set_ylim((-300,300))
        