""" Runs INDP, td-INDP, and Judgment Call """
import sys
import os
import multiprocessing
import pandas as pd
import indp
import dindputils
import gametree
import gameutils

def batch_run(params, fail_sce_param, player_ordering=[3, 1]):
    # Set root directories
    base_dir = fail_sce_param['BASE_DIR']
    damage_dir = fail_sce_param['DAMAGE_DIR']
    topology = None
    infrastructure_data = None
    ext_interdependency = None
    if fail_sce_param['TYPE'] == 'Andres':
        infrastructure_data = 'shelby_old'
        ext_interdependency = "../data/INDP_4-12-2016"
    elif fail_sce_param['TYPE'] == 'WU':
        infrastructure_data = 'shelby_extended'
        if fail_sce_param['FILTER_SCE'] is not None:
            list_high_dam = pd.read_csv(fail_sce_param['FILTER_SCE'])
    elif fail_sce_param['TYPE'] == 'random':
        infrastructure_data = 'shelby_extended'
    elif fail_sce_param['TYPE'] == 'synthetic':
        topology = fail_sce_param['TOPO']

    print('----Running for resources: '+str(params['V']))
    for m in fail_sce_param['MAGS']:
        for i in fail_sce_param['SAMPLE_RANGE']:
            try:
                list_high_dam
                if len(list_high_dam.loc[(list_high_dam.set == i)&\
                                         (list_high_dam.sce == m)].index) == 0:
                    continue
            except NameError:
                pass

            print('---Running Magnitude '+str(m)+' sample '+str(i)+'...')
            print("Initializing network...")
            if infrastructure_data:
                params["N"], _, _ = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            sim_number=0, magnitude=6, sample=0, v=params["V"],
                            infrastructure_data=infrastructure_data)
            else:
                params["N"], params["V"], params['L'] = indp.initialize_network(BASE_DIR=base_dir,
                            external_interdependency_dir=ext_interdependency,
                            magnitude=m, sample=i, infrastructure_data=infrastructure_data,
                            topology=topology)
            params["SIM_NUMBER"] = i
            params["MAGNITUDE"] = m
            # Check if the results exist
            output_dir_full = ''
            if params["ALGORITHM"] in ["INDP"]:
                output_dir_full = params["OUTPUT_DIR"]+'_L'+str(len(params["L"]))+'_m'+\
                    str(params["MAGNITUDE"])+"_v"+str(params["V"])+'/agents/actions_'+str(i)+'_L1_.csv'
            if os.path.exists(output_dir_full):
                print('results are already there\n')
                continue

            if fail_sce_param['TYPE'] == 'WU':
                indp.add_Wu_failure_scenario(params["N"], DAM_DIR=damage_dir,
                                             noSet=i, noSce=m)
            elif fail_sce_param['TYPE'] == 'ANDRES':
                indp.add_failure_scenario(params["N"], DAM_DIR=damage_dir,
                                          magnitude=m, v=params["V"], sim_number=i)
            elif fail_sce_param['TYPE'] == 'random':
                indp.add_random_failure_scenario(params["N"], DAM_DIR=damage_dir,
                                                 sample=i)
            elif fail_sce_param['TYPE'] == 'synthetic':
                indp.add_synthetic_failure_scenario(params["N"], DAM_DIR=base_dir,
                                                    topology=topology, config=m, sample=i)
            
            dynamic_params = None
            if params['DYNAMIC_PARAMS']:
                return_type = 'step_function'
                net_names = {'water':1,'gas':2,'power':3,'telecom':4}
                dynamic_params = {}
                for key, val in net_names.items():
                    filename = params['DYNAMIC_PARAMS']+'dynamic_demand_'+return_type+'_'+key+'.pkl'
                    with open(filename, 'rb') as f:
                        dd_df = pickle.load(f)
                    dynamic_params[val] = dd_df[(dd_df['sce']==m)&(dd_df['set']==i)]

            if params["ALGORITHM"] == "INDP":
                indp.run_indp(params, validate=False, T=params["T"], layers=params['L'],
                              controlled_layers=params['L'], saveModel=False, print_cmd_line=False,
                              dynamic_params=dynamic_params, co_location=False)
            elif params["ALGORITHM"] == "INFO_SHARE":
                indp.run_info_share(params, layers=params['L'], T=params["T"])
            elif params["ALGORITHM"] == "INRG":
                indp.run_inrg(params, layers=params['L'], player_ordering=player_ordering)
            elif params["ALGORITHM"] == "BACKWARDS_INDUCTION":
                gametree.run_backwards_induction(params["N"], i, players=params['L'],
                                                 player_ordering=player_ordering,
                                                 T=params["T"], outdir=params["OUTPUT_DIR"])
            elif params["ALGORITHM"] == "JC":
                dindputils.run_judgment_call(params, save_jc_model=False, print_cmd=False)
            elif params["ALGORITHM"] in ["NORMALGAME", "BAYESGAME"]:
                gameutils.run_game(params, save_results=True, print_cmd=False,
                                   save_model=False, plot2D=False)

def run_method(fail_sce_param, v_r, layers, method, judgment_type=None,
               res_alloc_type=None, valuation_type=None, output_dir='..', misc =None):
    for v in v_r:
        if method == 'INDP':
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":output_dir+'indp_results',
                      "V":v, "T":1, 'L':layers, "ALGORITHM":"INDP"}
        elif method == 'JC':
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":output_dir+'jc_results',
                      "V":v, "T":1, 'L':layers, "ALGORITHM":"JC",
                      "JUDGMENT_TYPE":judgment_type, "RES_ALLOC_TYPE":res_alloc_type,
                      "VALUATION_TYPE":valuation_type}
            if 'STM' in valuation_type:
                params['STM_MODEL_DICT'] = misc['STM_MODEL']
        elif method in ['NORMALGAME', 'BAYESGAME']:
            if method == "NORMALGAME":
                out_dir = output_dir+'ng_results'
            elif method == "BAYESGAME":
                out_dir = output_dir+'bg'+''.join(misc['SIGNALS'].values())+\
                    ''.join(misc['BELIEFS'].values())+'_results'
            params = {"NUM_ITERATIONS":10, "OUTPUT_DIR":out_dir,
                      "V":v, "T":1, "L":layers, "ALGORITHM":method,
                      'EQUIBALG':'enumerate_pure', "JUDGMENT_TYPE":judgment_type,
                      "RES_ALLOC_TYPE":res_alloc_type, "VALUATION_TYPE":valuation_type}
            if misc:
                params['PAYOFF_DIR'] = misc['PAYOFF_DIR']
                if params['PAYOFF_DIR']:
                    params['PAYOFF_DIR'] += 'ng_results'
            if method == 'BAYESGAME':
                params["SIGNALS"] = misc['SIGNALS']
                params["BELIEFS"] = misc['BELIEFS']
        else:
            sys.exit('Wrong method name: '+method)

        params['DYNAMIC_PARAMS'] = misc['DYNAMIC_PARAMS']
        if misc['DYNAMIC_PARAMS']:
            prefix = params['OUTPUT_DIR'].split('/')[-1]
            params['OUTPUT_DIR'] = params['OUTPUT_DIR'].replace(prefix,'dp_'+prefix)

        batch_run(params, fail_sce_param)

def run_parallel(i):
    '''
    Runs methods in parallel

    Parameters
    ----------
    i : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    filter_sce = None
    base_dir = '/scratch/ht20/Generated_Network_Dataset_v4/'
    damage_dir = '/scratch/ht20/Generated_Network_Dataset_v4/'
    output_dir = '/scratch/ht20/results/'
    dynamic_params_dir = None
    payoff_dir = OUTPUT_DIR+'General/results/'

    sample_no = 0
    mag_no = i

    fail_sce_param = {"TYPE":"synthetic","SAMPLE_RANGE":range(sample_no,sample_no+1),
                     "MAGS":range(mag_no,mag_no+1),'FILTER_SCE':filter_sce, 'TOPO':'General',
                     'BASE_DIR':base_dir,'DAMAGE_DIR':damage_dir}
    rc = [0]
    layers = [1, 2]
    judge_type = ["OPTIMISTIC"] #OPTIMISTIC #'DET-DEMAND' #PESSIMISTIC
    res_alloc_type = ["OPTIMAL"] #"MDA", "MAA", "MCA", 'UNIFORM' "OPTIMAL"
    val_type = ['DTC'] #'DTC'

    misc = {'PAYOFF_DIR':payoff_dir, 'DYNAMIC_PARAMS':dynamic_params_dir, 'REDUCED_ACTIONS':True} 

    run_method(fail_sce_param, rc, layers, method='INDP', output_dir=output_dir,
               misc = {'DYNAMIC_PARAMS':dynamic_params_dir})
    # run_method(fail_sce_param, rc, layers, method='TD_INDP', output_dir=output_dir, 
			   # misc = {'DYNAMIC_PARAMS':dynamic_params_dir})
    run_method(fail_sce_param, rc, layers, method='NORMALGAME', judgment_type=judge_type,
               res_alloc_type=res_alloc_type, valuation_type=val_type, output_dir=output_dir,
               misc = misc)
	for sig in [{1:'C', 2:'C'}, {1:'C', 2:'N'}, {1:'N', 2:'C'}, {1:'N', 2:'N'}]:#{x:'N' for x in layers}
		misc["SIGNALS"] = s
		misc["BELIEFS"] = {1:'U', 2:'U'}
		run_method(fail_sce_param, rc, layers, method='BAYESGAME', judgment_type=judge_type,
				   res_alloc_type=res_alloc_type, valuation_type=val_type, output_dir=output_dir,
				   misc = misc)

if __name__ == "__main__":
    NUM_CORES = multiprocessing.cpu_count()
    print('number of cores:'+str(NUM_CORES)+'\n')
    POOL = multiprocessing.Pool(NUM_CORES)
    RESULTS = POOL.map(run_parallel, range(100))
