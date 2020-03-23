import STAR_utils
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import cPickle as pickle
from functools import partial
import multiprocessing

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
    # network_objects_all={}
    # feature_all= {}
    # for res in v_r:
    #     params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'results/indp_results',
    #             "V":res,"ALGORITHM":"INDP"}
        
    #     samples,network_objects,initial_net,_,_=STAR_utils.importData(params,failSce_param,layers) 
    #     sample_all[res]=samples
    #     network_objects_all[res]=network_objects
    # STAR_utils.save_initial_data(initial_net,samples,network_objects)
        
    # ''' Prepare training and testing datsets '''
    # keys=['w_(1, 2)','y_(2, 2),(11, 2)','y_(11, 2),(2, 2)'] #$samples.keys() 
    # node_data,arc_data = STAR_utils.prepare_data(samples,res,initial_net,keys)
    # train_data,test_data = STAR_utils.train_test_split(node_data,arc_data,keys)
    # STAR_utils.save_prepared_data(train_data,test_data)
    
    # ''' train and test model'''        
    # trace,model = STAR_utils.train_model(train_data) 
    # STAR_utils.save_traces(trace)
    # ppc,ppc_test = STAR_utils.test_model(train_data,test_data,trace,model)

    t_suf = '20200322'
    samples,network_objects,initial_net = pickle.load(open('data'+t_suf+'/initial_data.pkl', "rb" ))
    no_prediction_samples=5
    for res in v_r:
        for s in failSce_param['sample_range']:
            print '\nSample '+`s`,
            pool = multiprocessing.Pool()
            pred_s_iterable = range(no_prediction_samples) 
            func = partial(STAR_utils.compare_resotration,samples,s,res,network_objects,
                          failSce_param,initial_net,layers,output_dir)
            pool.map(func, pred_s_iterable)
            pool.close()
            pool.join()               
    # cols_results=['sample','time','resource_cap','pred_sample','result_type','cost',
    #               'run_time','performance']
    # result_df = pd.DataFrame(columns=cols_results)                
    # figure_df = result_df.convert_objects(convert_numeric=True)
    # sns.lineplot(x="time", y="cost",hue="pred_sample", style="result_type",data=figure_df[figure_df['result_type']=='predicted'])
    # sns.lineplot(x="time", y="cost",data=figure_df[figure_df['result_type']=='data'])
