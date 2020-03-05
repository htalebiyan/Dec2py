import STAR_utils
import matplotlib.pyplot as plt 
import networkx as nx

import indp 
import indputils
import numpy as np
import pandas as pd
import seaborn as sns
import flow 

if __name__ == "__main__":  
    plt.close('all')
    ''' Decide the failure scenario (Andres or Wu) and network dataset (shelby or synthetic)
    Help:
    For Andres scenario: sample range: failSce_param["sample_range"], magnitudes: failSce_param['mags']
    For Wu scenario: set range: failSce_param["sample_range"], sce range: failSce_param['mags']
    For Synthetic nets: sample range: failSce_param["sample_range"], configurations: failSce_param['mags']  
    '''
    # listFilteredSce = '../data/damagedElements_sliceQuantile_0.95.csv'
    # base_dir = "../../data/Extended_Shelby_County/"
    # damage_dir = "../../data/random_disruption_shelby/"
    # output_dir = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/'   
    # # failSce_param = {"type":"ANDRES","sample_range":range(1,1001),"mags":[6,7,8,9],
    # #                   'Base_dir':base_dir,'Damage_dir':damage_dir}
    # # failSce_param = {"type":"WU","sample_range":range(23,24),"mags":range(5,6),
    # #                 'filtered_List':listFilteredSce,'Base_dir':base_dir,'Damage_dir':damage_dir}
    # failSce_param = {"type":"random","sample_range":range(50,500),"mags":range(0,1),
    #                 'filtered_List':None,'Base_dir':base_dir,'Damage_dir':damage_dir}
    # v_r = [4]#,10,20,30,40,50,60,70,80,90,100]
    # layers=[1,2,3,4]
    # sample_all={}
    # network_objects_all={}
    # feature_all= {}
    # for res in v_r:
    #     params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'results/indp_results',
    #             "V":res,"ALGORITHM":"INDP"}
        
    #     samples,network_objects,initial_net,_,_=STAR_utils.importData(params,failSce_param,layers) 
    #     sample_all[res]=samples
    #     network_objects_all[res]=network_objects
    # train_data_all,train_data = STAR_utils.train_data(samples,res,initial_net)
    trace,model = STAR_utils.train_model(train_data_all,train_data) #samplesDiff, trainData, ppc = 
    ppc = STAR_utils.test_model(train_data_all,trace,model)
    
    # no_samples = samples[samples.keys()[0]].shape[1]
    # no_time_steps = samples[samples.keys()[0]].shape[0]
    # cols=['sample','time','resource_cap','data_cost','data_run_time','data_pref',
    #       'predicted_cost','predicted_run_time','predicted_pref']
    # result_df = pd.DataFrame(columns=cols)
    # no_time_steps=10
    # for res in v_r:
    #     for s in failSce_param['sample_range']:
    #         s_itr = s - failSce_param['sample_range'][0]
    #         for t in range(no_time_steps):
    #             '''compute total cost for the predictions'''
    #             # print 'Flow problem: time step '+`t`
    #             decision_vars={0:{}} #0 becasue iINDP
    #             for key,val in samples.iteritems():
    #                 decision_vars[0][key]=val[t,s_itr] #0 becasue iINDP
    #             flow_results=flow.flow_problem(network_objects[0,s],v_r=0,
    #                             layers=layers,controlled_layers=layers,
    #                           decision_vars=decision_vars,print_cmd=True, time_limit=None)
                
    #             ''' read cost from the actual data computed by INDP '''
    #             folder_name = output_dir+'results/indp_results_L'+`len(layers)`+'_m0_v'+`res`
    #             real_results=indputils.INDPResults()
    #             real_results=real_results.from_csv(folder_name,s,suffix="")
    #             row = np.array([s,t,res,
    #                             real_results[t]['costs']['Total'],
    #                             real_results[t]['run_time'],
    #                             real_results[t]['costs']['Under Supply Perc'],
    #                             flow_results[1][0]['costs']['Total'],
    #                             flow_results[1][0]['run_time'],
    #                             flow_results[1][0]['costs']['Under Supply Perc']])
    #             temp = pd.Series(row,index=cols)
    #             result_df=result_df.append(temp,ignore_index=True)
    #             ### Write models to file  
    #             indp.save_INDP_model_to_file(flow_results[0],'./models',t,l=0)
    # melted_result_df_cost=pd.melt(result_df, id_vars =['time','sample','resource_cap'], value_vars =['data_cost', 'predicted_cost'])
    # figure_df = melted_result_df_cost.convert_objects(convert_numeric=True)
    # sns.lineplot(x="time", y="value",hue="variable", style="variable",data=figure_df)