from STAR_utils import *
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import _pickle as pickle
from functools import partial
import indp 
import copy
from os import listdir
from os.path import isfile, join
# from plot_STAR import plot_correlation

if __name__ == "__main__": 
    # t_suf = '20200430'
    # samples_all,costs_all,costs_local_all,initial_net = pickle.load(open('data'+t_suf+'/initial_data.pkl', "rb" ))
    # train_data,test_data = pickle.load(open('data'+t_suf+'/train_test_data.pkl', "rb" ))

    plt.close('all')
    base_dir = "../data/Extended_Shelby_County/"
    damage_dir = "../data/random_disruption_shelby/"
    output_dir = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/'
    failSce_param = {"type":"random","sample_range":range(0,2),"mags":range(0,1),
                    'filtered_List':None,'Base_dir':base_dir,'Damage_dir':damage_dir}
    v_r = [3] #[1,2,3,4,5,6,8,10,12,15,18,20,30,40,50,60,70,80,90,100]
    layers=[1,2,3,4]
 
    ''' Read all data '''
    samples_all = {}
    costs_all = {}
    print('Importing data:')
    for res in v_r:
        params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'results/indp_results',
                "V":res,"ALGORITHM":"INDP"}
        
        samples,costs,initial_net,_,_=importData(params,failSce_param,layers) 
        samples_all[res]=samples
        costs_all[res]=costs
    save_initial_data(initial_net,samples_all,costs_all)

    ''' Prepare training and testing datsets '''    
    # keys = ['y_(8, 2),(15, 2)','y_(15, 2),(8, 2)']
    ##['w_(1, 4)']   #samples.keys()   #['y_(2, 2),(11, 2)','y_(11, 2),(2, 2)'] 
    
    # node_data_all={key:pd.DataFrame() for key in keys}
    # arc_data_all={key:pd.DataFrame() for key in keys}
    # for res in v_r:
    #     print('\nNumber of resources: '+`res`)
    #     node_data,arc_data = prepare_data(samples_all[res],costs_all[res],
    #                                       initial_net,res,keys)
    #     for key in keys:
    #         if  arc_data_all[key].empty: #node_data_all[key].empty or arc_data_all[key].empty:
    #             # node_data_all[key]=node_data[key]
    #             arc_data_all[key]=arc_data[key]
    #         else:
    #             # node_data_all[key]=node_data_all[key].append(node_data[key],ignore_index=True)
    #             arc_data_all[key]=arc_data_all[key].append(arc_data[key],ignore_index=True)
    
    # train_data,test_data = train_test_split(node_data_all,arc_data_all,keys)
    # save_prepared_data(train_data,test_data)
    
    ''' train and test model'''
    # exclusions=['w_t_1','y_t_1','w_h_t_1','time','Total','Under_Supply_Perc','Over_Supply','Space_Prep'] 
    # ##,'y_t_1','w_n_t_1','w_a_t_1', 'w_d_t_1','time','w_h_t_1','Total','Flow','Under_Supply_layer'
    
    # mypath='parameters20200501/'
    # files = [f[17:-4] for f in listdir(mypath) if isfile(join(mypath, f))]
    # keys= [x for x in train_data.keys() if (x not in files)] 
    
    # for key in keys:  
    #     print('\n'+key)
    #     trace,model = train_model({key:train_data[key]},exclusions) 
    #     save_traces(trace)
    #     _,_ = test_model({key:train_data[key]},{key:test_data[key]},
    #                       trace,model,exclusions,plot=False)
    
    # plot_correlation(node_data_all,keys,['w_t_1'])
    
    '''compare restoration plans'''  
    # test_samples =range(50,551,100) #[50,100,150,200,250,300,350,400]
    # v_r = [2,4,15,50,100]
    # no_prediction_samples=1
    # t_suf = '20200428'
    # model_folder='./traces'+t_suf
    # param_folder ='./parameters'+t_suf
    # for res in v_r:
    #     print '\nRc '+`res`,
    #     for s in test_samples:#failSce_param['sample_range']:
    #         print '\nSample '+`s`,
    #         real_results_dir=output_dir+'results/indp_results_L'+`len(layers)`+'_m'+`0`+'_v'+`res`
    #         s_idx = find_sce_index(s,res,'./data20200426/missing_scenarios.txt') # index of the scenario in the data
    #         if s_idx!=-1:   
    #             for pred_s in range(0,no_prediction_samples):
    #                 network_object = copy.deepcopy(initial_net[0])
    #                 indp.add_random_failure_scenario(network_object,DAM_DIR=damage_dir,sample=s)
    #                 compare_resotration(pred_s,samples_all[res],costs_all[res],s,s_idx,res,network_object,failSce_param,
    #                                     initial_net,layers,real_results_dir,model_folder,param_folder) 
    #         else:
    #             print('\nThe scenatio does not exist in the data.')
    