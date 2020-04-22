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
    # from os import listdir
    # from os.path import isfile, join
    # mypath='parameters20200419/'
    # files = [f[17:-4] for f in listdir(mypath) if isfile(join(mypath, f))]
    # keys= [x for x in samples.keys() if (x[0]=='w' and x not in files)] 
    keys = ['w_(1, 4)']   #samples.keys()   #['y_(2, 2),(11, 2)','y_(11, 2),(2, 2)'] 
    
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
    # test_samples =range(250,450,10) #[50,100,150,200,250,300,350,400]
    # no_prediction_samples=1
    # t_suf = '20200419'
    # model_folder='./traces'+t_suf
    # param_folder ='./parameters'+t_suf
    # for res in v_r:
    #     for s in test_samples:#failSce_param['sample_range']:
    #         print '\nSample '+`s`,
    #         for pred_s in range(0,no_prediction_samples):
    #             network_object = copy.deepcopy(initial_net[0])
    #             indp.add_random_failure_scenario(network_object,DAM_DIR=damage_dir,sample=s)
    #             compare_resotration(pred_s,samples,costs,costs_local,s,res,network_object,failSce_param,
    #                                 initial_net,layers,output_dir,model_folder,param_folder)    