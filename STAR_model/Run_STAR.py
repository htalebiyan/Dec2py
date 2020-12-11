import STAR_utils
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from functools import partial
import indp 
import copy
from os import listdir
from os.path import isfile, join
from plot_star import plot_correlation

if __name__ == "__main__": 
    plt.close('all')
    direct = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc_only_nodes_damaged/'
    rooy_folder = direct+'data'
    samples_all, costs_all, initial_net = pickle.load(open(rooy_folder+'/initial_data.pkl', "rb" ))
    train_data,test_data = pickle.load(open(rooy_folder+'/train_test_data.pkl', "rb" ))

    
    base_dir = "../data/Extended_Shelby_County/"
    damage_dir = "../data/random_disruption_shelby/"
    output_dir = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/results_only_nodes_damaged/'
    failSce_param = {"type":"random","sample_range":range(0,1000),"mags":range(0,1),
                    'filtered_List':None,'Base_dir':base_dir,'Damage_dir':damage_dir}
    v_r = [1,2,3,4,5,6,8,10,12,14,16,18,20,30,40,50,60,70,80,90,100]
    layers = [1,2,3,4]

    aaa =[]
    for k,v in train_data.items():
        if v['w_d_t_1'].mean()!=0:
            aaa.append(k)
            print(k, v['w_d_t_1'].mean())
    ''' Read all data '''
    # samples_all = {}
    # costs_all = {}
    # print('Importing data:')
    # for res in v_r:
    #     params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'indp_results',
    #             "V":res,"ALGORITHM":"INDP", 'L':layers}
    #     samples, costs, initial_net, _, _ = STAR_utils.importData(params, failSce_param)
    #     samples_all[res] = samples
    #     costs_all[res] = costs
    # STAR_utils.save_initial_data(initial_net, samples_all, costs_all)

    ''' Prepare training and testing datsets '''    
    # keys = [x for x in samples_all[1].keys() if x[0] == 'w']
    # #['w_(1, 4)']   #samples.keys()   #['y_(65, 3),(21, 3)','y_(21, 3),(65, 3)'] 
    
    # node_data_all={key:pd.DataFrame() for key in keys if key[0] == 'w'}
    # arc_data_all={key:pd.DataFrame() for key in keys if key[0] == 'y'}
    # for res in v_r:
    #     print('\nNumber of resources: ', res)
    #     node_data, arc_data = STAR_utils.prepare_data(samples_all[res], costs_all[res],
    #                                                   initial_net, res, keys)
    #     for key in keys:
    #         if key[0] == 'y':
    #             if arc_data_all[key].empty:
    #                 arc_data_all[key]=arc_data[key]
    #             else:
    #                 arc_data_all[key]=arc_data_all[key].append(arc_data[key],ignore_index=True)
    #         if key[0] == 'w':
    #             if node_data_all[key].empty:
    #                 node_data_all[key]=node_data[key]
    #             else:
    #                 node_data_all[key]=node_data_all[key].append(node_data[key],ignore_index=True)
    # train_data,test_data = STAR_utils.train_test_split(node_data_all, arc_data_all,keys)
    # STAR_utils.save_prepared_data(train_data,test_data)

    ''' train and test model'''
    # exclusions=['w_t_1','y_t_1','w_a_t_1','y_c_t_1','Arc','Under_Supply_Perc',
    #             'Under_Supply','Over_Supply','Space_Prep','time'] 
    # ## Shared inputs included: ,'Flow','Node','Rc','w_c_t_1', 'w_h_t_1','Total'
    # ## Node model inputs inlcuded: 'w_n_t_1', 'w_d_t_1', 
    # ## Arc model inputs inlcuded: ??? 
    
    # # plot_correlation(node_data_all,keys,['w_t_1','w_a_t_1','y_c_t_1','Arc'])
        
    # mypath='parameters/'
    # files = [f[17:-4] for f in listdir(mypath) if isfile(join(mypath, f))]
    # keys= [x for x in train_data.keys() if (x not in files)] 
    # for key in keys:
    #     print('\n'+key)
    #     trace,model = STAR_utils.train_model({key:train_data[key]},exclusions) 
    #     STAR_utils.save_traces(trace)
    #     _,_ = STAR_utils.test_model({key:train_data[key]},{key:test_data[key]},
    #                       trace,model,exclusions,plot=False)
    

