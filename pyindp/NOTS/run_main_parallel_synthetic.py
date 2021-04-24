""" Runs INDP, td-INDP, and Judgment Call """
import os
import multiprocessing
import pandas as pd
import indp
import dindputils
import runutils
import gameutils


def run_parallel(i):
    """
    Runs methods in parallel

    Parameters
    ----------
    i : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    filter_sce = None
    base_dir = '/scratch/ht20/Generated_Network_Dataset_v4.1/'
    damage_dir = '/scratch/ht20/Generated_Network_Dataset_v4.1/'
    output_dir = '/scratch/ht20/results/'
    dynamic_params_dir = None
    payoff_dir = None

    sample_no = 0
    mag_no = i

    fail_sce_param = {"TYPE": "synthetic", "SAMPLE_RANGE": range(sample_no, sample_no + 1),
                      "MAGS": range(mag_no, mag_no + 1), 'FILTER_SCE': filter_sce, 'TOPO': 'General',
                      'BASE_DIR': base_dir, 'DAMAGE_DIR': damage_dir}
    rc = [0]
    layers = [1, 2]
    judge_type = ["OPTIMISTIC"]  # OPTIMISTIC #'DET-DEMAND' #PESSIMISTIC
    res_alloc_type = ["OPTIMAL"]  # "MDA", "MAA", "MCA", 'UNIFORM' "OPTIMAL"
    val_type = ['DTC']  # 'DTC'

    misc = {'PAYOFF_DIR': payoff_dir, 'DYNAMIC_PARAMS': dynamic_params_dir, 'REDUCED_ACTIONS': None}

    runutils.run_method(fail_sce_param, rc, layers, method='INDP', output_dir=output_dir,
                        misc={'DYNAMIC_PARAMS': dynamic_params_dir})
    # runutils.run_method(fail_sce_param, rc, layers, method='TD_INDP', output_dir=output_dir, 
    # misc = {'DYNAMIC_PARAMS':dynamic_params_dir})

    runutils.run_method(fail_sce_param, rc, layers, method='NORMALGAME', judgment_type=judge_type,
                        res_alloc_type=res_alloc_type, valuation_type=val_type, output_dir=output_dir,
                        misc=misc)
    # for sig in [{1:'C', 2:'C'}, {1:'C', 2:'N'}, {1:'N', 2:'C'}, {1:'N', 2:'N'}]:#{x:'N' for x in layers}
    # misc["SIGNALS"] = s
    # misc["BELIEFS"] = {1:'U', 2:'U'}
    # runutils.run_method(fail_sce_param, rc, layers, method='BAYESGAME', judgment_type=judge_type,
    # res_alloc_type=res_alloc_type, valuation_type=val_type, output_dir=output_dir,
    # misc = misc)


if __name__ == "__main__":
    NUM_CORES = multiprocessing.cpu_count()
    print('number of cores:' + str(NUM_CORES) + '\n')
    POOL = multiprocessing.Pool(NUM_CORES)
    RESULTS = POOL.map(run_parallel, range(100))
