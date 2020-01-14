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
    listFilteredSce = '../../data/damagedElements_sliceQuantile_0.95.csv'
    base_dir = "../../" #'C:/Users/ht20/Documents/Files/Generated_Network_Dataset_v3.1/'
    output_dir = 'C:/Users/ht20/Documents/Files/Auction_Extended_Shelby_County_Data/' #'C:/Users/ht20/Documents/Files/Auction_synthetic_networks_v3.1/'

#    failSce = read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/",magnitude=8)
#    failSce_param = {"type":"ANDRES","sample_range":range(1,1001),"mags":[6,7,8,9]}
    failSce_param = {"type":"WU","sample_range":range(0,50),"mags":range(0,96),
                     'filtered_List':listFilteredSce,'Base_dir':base_dir}
#    failSce_param = {"type":"synthetic","sample_range":range(0,5),"mags":range(0,100),
#                     'filtered_List':None,'topology':'Grid','Base_dir':base_dir}
    v = 3
    layers=[1,2,3,4]
    params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'results/indp_results',
            "V":v,"ALGORITHM":"INDP"}
    
    samples,network_objects,initial_net=STAR_utils.input_matrices=STAR_utils.importData(params,failSce_param,layers)
    STAR_utils.train_model(samples)
    
   