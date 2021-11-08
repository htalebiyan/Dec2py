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
    base_dir = '/scratch/ht20/Extended_Shelby_County/'
    damage_dir = '/scratch/ht20/Wu_Damage_scenarios/'
    output_dir = '/scratch/ht20/results/'
    dynamic_params_dir = None

    sample_no = i
    mag_no = 1000

    fail_sce_param = {"TYPE": "from_csv", "L2_RANGE": range(sample_no, sample_no + 1),
                      "L1_RANGE": range(mag_no, mag_no + 1), 'FILTER_SCE': None,
                      'BASE_DIR': base_dir, 'DAMAGE_DIR': damage_dir}
	root_dir = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/"
	dynamic_params_dir = {'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED': 'seaside', 'OUT_DIR': OUTPUT_DIR, 
						'POP_DISLOC_DATA': root_dir + 'Seaside_notebook/output/1000yr/',
						'MAPPING': {'POWER': root_dir + 'Power/bldgs2elec_Seaside.csv',
									'WATER': root_dir + 'Water/bldgs2wter_Seaside.csv'}}
	T = 10
    rc = [{'budget': {t: 2.4e5 for t in range(T)}, 'time': {t: 70 for t in range(T)}}]
	rc[0]['budget'][0] = 4.4e5
	rc[0]['time'][0] = 700
    layers = [1, 3]

    misc = {'PAYOFF_DIR': payoff_dir, 'DYNAMIC_PARAMS': dynamic_params_dir,
            'REDUCED_ACTIONS': 'EDM'}

    runutils_v2.run_method(fail_sce_param, rc, T, layers, method='INMRP', output_dir=output_dir,
                        misc={'DYNAMIC_PARAMS': dynamic_params_dir, 'EXTRA_COMMODITY': EXTRA_COMMODITY,
                             'TIME_RESOURCE': True})

if __name__ == "__main__":
    NUM_CORES = multiprocessing.cpu_count()
    print('number of cores:' + str(NUM_CORES) + '\n')
    POOL = multiprocessing.Pool(NUM_CORES)
    RESULTS = POOL.map(run_parallel, range(30))
