import pickle
import indp
root = '/home/hesam/Desktop/Files/Game_Shelby_County/results/ng_results_L4_m92_v12_OPTIMISTIC_OPTIMAL/'
with open(root+'objs_30.pkl', 'rb') as f:
    obj = pickle.load(f)

BASE_DIR = "../data/Extended_Shelby_County/"
DAMAGE_DIR = "../data/Wu_Damage_scenarios/"
obj.net, _, _ = indp.initialize_network(BASE_DIR=BASE_DIR,
            external_interdependency_dir=None,
            sim_number=0, magnitude=6, sample=0, v=12,
            shelby_data='shelby_extended')
indp.add_Wu_failure_scenario(obj.net, DAM_DIR=DAMAGE_DIR,
                             noSet=30, noSce=92)
N_hat_prime= [n for n in obj.net.G.nodes(data=True) if n[1]['data']['inf_data'].repaired==0.0]
for t in range(obj.time_steps):
    obj.objs[t+1].find_optimal_solution()
    print(obj.objs[t+1].optimal_solution['total cost'])
    indp.apply_recovery(obj.net, obj.results, t+1)

obj.save_object_to_file()