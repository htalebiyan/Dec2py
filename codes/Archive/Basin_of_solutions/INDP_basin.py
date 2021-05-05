"""  """
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import indp
import indpalt
import itertools
from textwrap import wrap
sns.set(context='notebook', style='darkgrid', font_scale=1.2)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

def convert_action(actions):
    conv_act = []
    for ac in actions:
        if '/' in ac:
            act = ac.split('/')
            u = act[0].split('.')
            v = act[1].split('.')
            conv_act.append(((int(u[0]),int(u[1])),(int(v[0]),int(v[1]))))
        else:
            n = ac.split('.')
            conv_act.append((int(n[0]),int(n[1])))
    return conv_act

''' Set up the problem '''
#: The address to the basic (topology, parameters, etc.) information of the network.
BASE_DIR = "../../data/Extended_Shelby_County/"

#: The address to damge scenario data.
DAMAGE_DIR = "../../data/Wu_Damage_scenarios/"

#: The address to where output are stored.
OUTPUT_DIR = '../../results/'

RC = 3 #, 4, 6, 8, 12]
LAYERS = [1, 2, 3, 4]
params = {"NUM_ITERATIONS":1, "OUTPUT_DIR":OUTPUT_DIR+'indp_results',
          "V":RC, "T":1, 'L':LAYERS, "ALGORITHM":"INDP"}
m = 85
i = 2

print('---Running Magnitude '+str(m)+' sample '+str(i)+'...')
print("Initializing network...")
params["N"], _, _ = indp.initialize_network(BASE_DIR=BASE_DIR,
            external_interdependency_dir="",
            sim_number=0, magnitude=6, sample=0, v=params["V"],
            shelby_data='shelby_extended')
indp.add_Wu_failure_scenario(params["N"], DAM_DIR=DAMAGE_DIR,
                              noSet=i, noSce=m)

''' Final Total Cost for all possible actions '''
# actions = []
# for n,d in params["N"].G.nodes(data=True):
#     if d['data']['inf_data'].functionality==0.0:
#         actions.append(n)
# for u,v,a in params["N"].G.edges(data=True):
#     if a['data']['inf_data'].functionality==0.0 and (v,u) not in actions:
#         actions.append((u,v))
# actions_super_set = []
# for v in range(RC+1):
#     actions_super_set.extend(list(itertools.combinations(actions, v)))        
# # compute payoffs for each possible combinations of actions
# results = pd.DataFrame(columns=['actions', 'Total Cost', 'id'])
# for idx, ac in enumerate(actions_super_set):
#     print('INDP for', ac, idx/len(actions_super_set)*100)
#     decision_vars = {0:{}} #0 becasue iINDP
#     for n,d in params["N"].G.nodes(data=True):
#         if d['data']['inf_data'].functionality!=0.0:
#             decision_vars[0]['w_'+str(n)] = 1.0
#         elif n in ac:
#             decision_vars[0]['w_'+str(n)] = 1.0
#         else:
#             decision_vars[0]['w_'+str(n)] = 0.0
#     for u,v,a in params["N"].G.edges(data=True):
#         if a['data']['inf_data'].functionality!=0.0:
#             decision_vars[0]['y_'+str(u)+","+str(v)] = 1.0
#         elif (u,v) in ac or (v,u) in ac:
#             decision_vars[0]['y_'+str(u)+","+str(v)] = 1.0
#         else:
#             decision_vars[0]['y_'+str(u)+","+str(v)] = 0.0
#     flow_results = indpalt.flow_problem(params["N"], v_r=0, layers=LAYERS,
#                                         controlled_layers=LAYERS,
#                                         decision_vars=decision_vars,
#                                         print_cmd=True, time_limit=None)
#     results = results.append({'actions': ac, 'id': idx,
#                               'Total Cost': flow_results[1].results[0]['costs']['Total']},
#                               ignore_index=True)
''' Find optimal solution '''
# indp_results = indp.indp(params["N"], v_r=RC, layers=LAYERS, controlled_layers=LAYERS,
#                           print_cmd=True, time_limit=None)
# optimal_tc = indp_results[1].results[0]['costs']['Total']
# optimal_act = indp_results[1].results[0]['actions']    
# optimal_act_conv = convert_action(optimal_act)
# min_tc_action = results[results["Total Cost"]==results["Total Cost"].min()]['actions'].iloc[0]
# for act in min_tc_action:
#     if act not in optimal_act_conv:
#         print(act, 'not in optimal solution')
#     elif len(act)==2:
#         duplicate = (act[1],act[0])
#         list_act = list(optimal_act_conv)
#         list_act.remove(duplicate)
#         optimal_act_conv = tuple(list_act)
# if optimal_act_conv!=min_tc_action:
#     print('Min actuin is not the same as the optimal solution')
    
''' Run Jc '''
# root = 'C:/Users/ht20/Documents/Files/Shelby_data_paper/Restoration_results_90_perc/'
# jc_tc = {}
# for jc in ['OPTIMISTIC']:
#     for rst in ['UNIFORM', 'MCA', 'OPTIMAL']:
#         if rst not in ["MDA", "MAA", "MCA"]:
#             output_dir_full = root+'jc_results'+'_L'+str(len(params["L"]))+'_m'+\
#                             str(m)+"_v"+str(RC)+'_'+jc+'_'+rst+'/'
#             jc_tc['JC/'+jc+'/'+rst] = indp.INDPResults().from_csv(output_dir_full, sample_num=i, suffix='real')
#         else:
#             for vt in ['DTC']:
#                 output_dir_full = root+'jc_results'+'_L'+str(len(params["L"]))+'_m'+\
#                                 str(m)+"_v"+str(RC)+'_'+jc+'_AUCTION_'+rst+'_'+vt   
#                 jc_tc['JC/'+jc+'/'+rst+'/'+vt] = indp.INDPResults().from_csv(output_dir_full, sample_num=i, suffix='real')
# coor_jc = {}
# for key, val in jc_tc.items():  
#         tc = val.results[1]['costs']['Total']
#         coor_jc[key] = (results[abs(results['Total Cost']-tc)<1e6]['actions'].iloc[0],tc)       

# with open('results.pkl', 'wb') as f:
#     pickle.dump([results, optimal_tc, optimal_act_conv, jc_tc, coor_jc], f)        
''' Plot '''
with open('results.pkl', 'rb') as f:
    [results, optimal_tc, optimal_act_conv, jc_tc, coor_jc] = pickle.load(f)

figure_df = results.sort_values("Total Cost", ascending=False)
figure_df["actions"] = figure_df["actions"].astype(str)
kwargs = {'edgecolor':"k", 'linewidth':0.1, 'linestyle':'-'}
g = sns.scatterplot(data=figure_df, x="actions", y="Total Cost",
                    s=15, marker="o",**kwargs)
offset = 30
bbox = dict(boxstyle="square", fc="0.9", ec=".1")
arrowprops = dict(arrowstyle="->", color=".1", 
                  connectionstyle="angle,angleA=0,angleB=-60,rad=10")
g.annotate('INDP', xy=(str(optimal_act_conv), optimal_tc),
            xytext=(.5*offset, -.5*offset), textcoords='offset points',
            bbox=bbox, arrowprops=arrowprops)
for key, val in coor_jc.items():
    adjust_offset = 1
    if key =='JC/OPTIMISTIC/MCA/DTC':
        adjust_offset = 1.5
    g.annotate(key, xy=(str(val[0]), val[1]),
             xytext=(-8*offset, offset*adjust_offset), textcoords='offset points',
             bbox=bbox, arrowprops=arrowprops) 
g.set(xticklabels=[], xticks=g.get_xticks()[0::50])
plt.savefig('basin.png', dpi=300)

