"""Runs models in parallel"""
import predict_restoration
def run_parallel(sample):
    """Runs models in parallel"""
    mags = range(0, 1)
    t_suf = ''
    model_dir = 'C:/Users/ht20/Documents/Files/STAR_models/Shelby_final_all_Rc'
    fail_sce_param = {"type":"random", "sample":None, "mag":None, 'filtered_List':None,
                      'Base_dir':"../data/Extended_Shelby_County/",
                      'Damage_dir':"../data/random_disruption_shelby/"}
    pred_dict = {'num_pred':5, 'model_dir':model_dir+'/traces'+t_suf,
                 'param_folder':model_dir+'/parameters'+t_suf,
                 'output_dir':'./results'}
    params = {"NUM_ITERATIONS":10, "V":5, "ALGORITHM":"INDP", 'L':[1, 2, 3, 4]}
    for mag in mags:
        fail_sce_param['sample'] = sample
        fail_sce_param['mag'] = mag
        predict_restoration.predict_resotration(pred_dict, fail_sce_param, params)
        