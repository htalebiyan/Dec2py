""" Runs INDP, td-INDP, and Judgment Call """
import multiprocessing
import runutils_v2


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
    sample_no = i
    mag_no = 1000

    root_dir = "C:/Users/ht20/Documents/GitHub/NIST_testbeds/Seaside/"
    base_dir = root_dir + 'Node_arc_info_v2/'
    damage_dir = root_dir + 'Damage_scenarios/cumulative_'+str(mag_no)+'yr_initial_damage/'
    output_dir = root_dir + 'results/'

    fail_sce_param = {"TYPE": "from_csv", "L2_RANGE": range(sample_no, sample_no + 1), 'BASE_DIR': base_dir,
                      'DAMAGE_DIR': damage_dir,  "L1_RANGE": range(mag_no, mag_no + 1), 'FILTER_SCE': None}
    dynamic_params_dir = {'TYPE': 'incore', 'RETURN': 'step_function', 'TESTBED': 'seaside', 'OUT_DIR': output_dir,
                          'POP_DISLOC_DATA': root_dir + 'Seaside_notebook/output/'+str(mag_no)+'yr/',
                          'MAPPING': {'POWER': root_dir + 'Power/bldgs2elec_Seaside.csv',
                                      'WATER': root_dir + 'Water/bldgs2wter_Seaside.csv'}}
    T = 10
    rc = [{'budget': {t: 2.4e5 for t in range(T)}, 'time': {t: 70 for t in range(T)}}]
    rc[0]['budget'][0] = 4.4e5
    rc[0]['time'][0] = 700
    layers = [3]
    runutils_v2.run_method(fail_sce_param, rc, T, layers, method='INMRP', output_dir=output_dir,
                           misc={'DYNAMIC_PARAMS': dynamic_params_dir, 'EXTRA_COMMODITY': None, 'TIME_RESOURCE': True})


if __name__ == "__main__":
    NUM_CORES = multiprocessing.cpu_count()
    print('number of cores:' + str(NUM_CORES) + '\n')
    POOL = multiprocessing.Pool(NUM_CORES)
    RESULTS = POOL.map(run_parallel, range(2))
