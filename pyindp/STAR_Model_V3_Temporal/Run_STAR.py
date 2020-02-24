import STAR_utils
import matplotlib.pyplot as plt 
import networkx as nx 

if __name__ == "__main__":  
    plt.close('all')
    ''' Decide the failure scenario (Andres or Wu) and network dataset (shelby or synthetic)
    Help:
    For Andres scenario: sample range: failSce_param["sample_range"], magnitudes: failSce_param['mags']
    For Wu scenario: set range: failSce_param["sample_range"], sce range: failSce_param['mags']
    For Synthetic nets: sample range: failSce_param["sample_range"], configurations: failSce_param['mags']  
    '''
    listFilteredSce = '../data/damagedElements_sliceQuantile_0.95.csv'
    base_dir = "../../data/Extended_Shelby_County/"
    damage_dir = "../../data/random_disruption_shelby/"
    output_dir = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/'   
    
    # failSce = read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/",magnitude=8)
    # failSce_param = {"type":"ANDRES","sample_range":range(1,1001),"mags":[6,7,8,9],
    #                  'Base_dir':base_dir,'Damage_dir':damage_dir}
    # failSce_param = {"type":"WU","sample_range":range(23,24),"mags":range(5,6),
    #                 'filtered_List':listFilteredSce,'Base_dir':base_dir,'Damage_dir':damage_dir}
    # failSce_param = {"type":"random","sample_range":range(50,500),"mags":range(0,1),
    #                 'filtered_List':None,'Base_dir':base_dir,'Damage_dir':damage_dir}
    # v_r = [4]#,10,20,30,40,50,60,70,80,90,100]
    # sample_all={}
    # feature_all= {}
    # for res in v_r:
    #     layers=[1,2,3,4]
    #     params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'results/indp_results',
    #             "V":res,"ALGORITHM":"INDP"}
        
    #     samples,_,initial_net,_,_=STAR_utils.importData(params,failSce_param,layers) 
    #     sample_all[res]=samples
    # train_data_all,train_data = STAR_utils.train_data(samples,res,initial_net)
    # trace,model = STAR_utils.train_model(train_data_all,train_data) #samplesDiff, trainData, ppc = 
    ppc = STAR_utils.test_model(train_data_all,trace,model)