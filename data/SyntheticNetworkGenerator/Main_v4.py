'''
This file contains the functions to generate and store a database of
interdependenct random networks
Hesam Talebiyan - Last updated: 03/04/2018
'''

import Network_Data_Generator
import numpy as np
import os
import random
import math
import os
import sys
import shutil
import networkx as nx

def topo_param(net_type, no_nodes):
    if net_type == 'grid':
        grid_size_x = np.random.randint(low=3, high=10)
        grid_size_y = np.random.randint(low=3, high=10)
        while grid_size_y*grid_size_x<=no_nodes:
            grid_size_y += 1
        while grid_size_y*grid_size_x>no_nodes:
            grid_size_y -= 1
        no_nodes = int(round(grid_size_x*grid_size_y))
        topo_param = grid_size_x
        mean_num_arcs = 2*no_nodes-grid_size_x-no_nodes//grid_size_x
        assert isinstance(mean_num_arcs, int), 'number of arcs is not an integer'
        assert mean_num_arcs>0, 'number of arcs is <= 0'
    if net_type == 'random':
        # Existence Probability of each arc in the random network
        # The lower bound corresponds to the lower bound of the supercritical Regime
        # and np.log(noNodes)/noNodes in the upper bound corresponds to the
        # lower bound of the connected regime
        prob_LB = 1.0/no_nodes
        prob_UB = (np.log(no_nodes)/no_nodes+1)*0.5
        arc_prob = np.random.uniform(low=prob_LB, high=prob_UB)
        topo_param = arc_prob
        mean_num_arcs = 0.5*no_nodes*(no_nodes-1)*arc_prob
    if net_type == 'scalefree':
        # Exponent of the powerlaw of node degee distribution
        # whose bounds correspond to Ultra-Small World regime.
        expLB = 2.001
        expUB = 2.999
        exp = np.random.uniform(low=expLB, high=expUB)
        topo_param = exp
        mean_num_arcs = no_nodes*(no_nodes**(1/(exp-1)))*0.5
    if net_type == 'tree':
        # The ratio of diameter to number of arcs=n-1
        temp = []
        for i in range(100):
            G = nx.generators.trees.random_tree(no_nodes)
            temp.append(nx.algorithms.distance_measures.diameter(G))
        diam = random.choice(temp)
        topo_param = diam/(no_nodes-1)
        mean_num_arcs = no_nodes-1
    if net_type == 'mpg':
        topo_param = 0
        mean_num_arcs = 3*no_nodes-6
    return topo_param, no_nodes, mean_num_arcs

# Input values
no_samples = 5 # Number of sample sets of network
no_config = 100 # Number of configurations
noZones = 4 # noZones by noZones tile of zones
paramError = 0.1
rootfolder = '/home/hesam/Desktop/Files/Generated_Network_Dataset_v4.1/' # Root folder where the database is
#'C:\\Users\\ht20\Documents\\Files\Generated_Network_Dataset_v3.1\\'
rootfolder += 'GeneralNetworks/' #'GridNetworks/' # choose relevant dataset folder
prefix = 'GEN'
if not os.path.exists(rootfolder):
    os.makedirs(rootfolder)

# The text file which stores the information of each configurations
fileNameList = rootfolder+'List_of_Configurations.txt'
# Saving network data in the text files
fList = open(fileNameList,"a+")
header = 'Config Number\t No. Layers\t No. Nodes\t Topology Parameter\t Interconnection Prob'+\
    '\t Damage Prob\t Resource Cap \t Net Types\n'
fList.write(header)
fList.close()

net_type ={}
no_nodes_dict = {}
int_prob_dict={}
topo_param_dict = {}
dam_prob_dict = {}
mean_dam_nodes = {}
mean_dam_arcs = {}
cnfg = 0
while cnfg<no_config:
    write_config = True
    # Number of layers
    no_layers = np.random.randint(low=2, high=3)
    # MEAN number of nodes of the random network (same for both networks)
    no_nodes = np.random.randint(low=10, high=50)
    # MEAN Existence Probability of each interconnection (among all possible pairs)
    # in the interdependent random networks
    int_prob = np.random.uniform(low=0.001, high=0.05)
    # MEAN Probability of damage of each node or arc in the random networks
    # Bounds are chosen roughly based on INDP data for Shelby county associated with
    # M6 (0.05) - M9 (0.5) scenarios
    dam_prob = np.random.uniform(low=0.05, high=0.5)

    for k in range(1,no_layers+1):
        # Choose a network type randomly
        net_type[k] = random.choice(['grid','scalefree','random', 'tree', 'mpg'])
        no_nodes_dict[k] = int(round(no_nodes*(1+np.random.normal(0, paramError))))
        topo_param_dict[k], no_nodes_dict[k], mean_dam_arcs[k] = topo_param(net_type=net_type[k],
                                                                            no_nodes=no_nodes_dict[k])
        dam_prob_dict[k] = dam_prob*(1+np.random.normal(0, paramError))
        mean_dam_nodes[k] = no_nodes_dict[k]*dam_prob_dict[k]
        mean_dam_arcs[k] = mean_dam_arcs[k]*dam_prob_dict[k]
        for kt in range(1,no_layers+1):
            if k!=kt:
                int_prob_dict[(kt,k)]=int_prob*(1+np.random.normal(0, paramError))
    # Restoration Resource Cap for each network
    # based on the sum of mean number of damaged nodes and arcs
    max_res_cap = 0.33*(sum([x for x in mean_dam_nodes.values()])+sum([x for x in mean_dam_arcs.values()]))
    res_cap = np.random.randint(low=2, high=max(4,max_res_cap))
    fList.close()
    nodes={}
    arcs={}
    pos={}
    b = {}
    Mp={}
    Mm={}
    q={}
    damNodes = {}
    damArcs = {}
    u = {}
    c = {}
    fa = {}
    s = 0
    num_iterations = 0
    while s < no_samples:
        num_iterations += 1
        for k in range(1,no_layers+1):
            # Generating random networks
            if net_type[k] == 'grid':
                grid_x = topo_param_dict[k]
                grid_y = no_nodes_dict[k]//topo_param_dict[k]
                nodes[k], arcs[k], pos[k] = Network_Data_Generator.Grid_network(grid_x,
                                                                                grid_y, k)
            if net_type[k] == 'random':
                nodes[k], arcs[k], pos[k] = Network_Data_Generator.Random_network(topo_param_dict[k],
                                                                                  no_nodes_dict[k] ,k)
            if net_type[k] == 'scalefree':
                nodes[k], arcs[k], pos[k] = Network_Data_Generator.Scale_Free_network(topo_param_dict[k],
                                                                                      no_nodes_dict[k] ,k)
            if net_type[k] == 'tree':
                nodes[k], arcs[k], pos[k] = Network_Data_Generator.Tree_network(topo_param_dict[k],
                                                                                no_nodes_dict[k] ,k)
            if net_type[k] == 'mpg':
                nodes[k], arcs[k], pos[k] = Network_Data_Generator.MPG_network(no_nodes_dict[k] ,k)
            '''All bounds are are chosen based on INDP data for Shelby county'''
            # Supply/Demand values for each node in the networks
            b[k] = np.random.randint(low=0, high=700, size=no_nodes_dict[k])
            b[k] = (b[k] - np.mean(b[k])).astype('int') # Making some values negative corresponds to demand
            b[k][0] = b[k][0]-sum(b[k]) # balancing the demand and supply
            # Over- and unde- supply penalties for each node in the networks
            Mp[k] = np.random.randint(low=5e6, high=10e6, size=no_nodes_dict[k])
            Mm[k] = np.random.randint(low=5e5, high=10e5, size=no_nodes_dict[k])
            # reconstruction cost for each node in the networks
            q[k] = np.random.randint(low=5e3, high=15e3, size=no_nodes_dict[k])
            # Damaged nodes and arcs in each networks # fix in the text #!!!
            damNodes[k], damArcsIndex= Network_Data_Generator.random_damage_Data(no_nodes_dict[k],
                                                                                 len(arcs[k]),
                                                                                 dam_prob_dict[k])
            damArcs[k] = []
            for da in damArcsIndex:
                damArcs[k].append(arcs[k][da])
            # Capacity of each arc in the networks
            u[k] = np.random.randint(low=500, high=2000, size=len(arcs[k]))
            # Flow cost of each arc in the networks
            c[k] = np.random.randint(low=50, high=500, size=len(arcs[k]))
            # reconstruction cost of each arc in the networks
            fa[k] = np.random.randint(low=2500, high=7e4, size=len(arcs[k]))

        no_sel_pairs_dict={}
        no_relevant_action ={x:0 for x in range(1,no_layers+1)}
        selPairs = {}
        for k in range(1,no_layers+1):
            for kt in range(1,no_layers+1):
                if kt!=k:
                    selPairs[(kt,k)] = Network_Data_Generator.generate_interconnctions(nodes[kt],
                                                                                       nodes[k],
                                                                                       int_prob_dict[(kt,k)])
                    no_sel_pairs_dict[(kt,k)] = len(selPairs[(kt,k)])
                    temp_rel_act = []
                    for sp in selPairs[(kt,k)]:
                        if sp[0] in damNodes[kt]:
                            temp_rel_act.append(sp[0])
                    no_relevant_action[kt] = len(list(set(temp_rel_act)))
        total_actions ={x:0 for x in range(1,no_layers+1)}
        total_actions_full_resource = {}
        ### All relevant actions which do not use all resources
        total_actions_relevent_only ={x:0 for x in range(1,no_layers+1)}
        for k in range(1,no_layers+1):
            for v in range(res_cap//no_layers+1):
                # 1 inside the combination represents OA 
                total_actions[k] += math.comb(no_relevant_action[k]+1, v+1)
            # 1 respresents NA
            total_actions[k] += 1
        for k in range(1,no_layers+1):
            for v in range(min(res_cap//no_layers+1, no_relevant_action[k])-1):
                total_actions_relevent_only[k] += math.comb(no_relevant_action[k], v+1)
            total_actions_full_resource[k] = total_actions[k] - total_actions_relevent_only[k]
        size_payoff_matrix = 1
        for x in total_actions.values():
            size_payoff_matrix *= x
        size_payoff_matrix_full_resource = 1
        for x in total_actions_full_resource.values():
            size_payoff_matrix_full_resource *= x

        # Making target folder to save network data
        folder = rootfolder+prefix+'Config_%d/Sample_%d' % (cnfg,s)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if num_iterations>5*no_samples:
            write_config=False
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            break
        if size_payoff_matrix<50 or size_payoff_matrix>50000:
            continue
        else:
            s += 1

        # The text files which stores network data
        for k in range(1,no_layers+1):
            fileName = folder + '/N%d_Nodes.txt' %(k)
            f = open(fileName,"w")
            for j in nodes[k]:
                f.write('%d\t' % (j))
                f.write(''.join('%1.5f\t' % n for n in pos[k][j]))
                f.write('%d\t%d\t%d\t%d\t' % (b[k][j],Mp[k][j],Mm[k][j],q[k][j]))
                f.write('\n')
            f.close()

            fileName = folder + '/N%d_Arcs.txt' %(k)
            f = open(fileName,"w")
            for j in range(len(arcs[k])):
                f.write(''.join('%d\t' % n for n in arcs[k][j]))
                f.write('%d\t%d\t%d\t' % (u[k][j],c[k][j],fa[k][j]))
                f.write('\n')
            f.close()

            fileName = folder + '/N%d_Damaged_Nodes.txt' %(k)
            f = open(fileName,"w")
            for j in damNodes[k]:
                f.write('%d\t' % (j))
                f.write('\n')
            f.close()

            fileName = folder + '/N%d_Damaged_Arcs.txt' % (k)
            f = open(fileName,"w")
            for j in damArcs[k]:
                f.write(''.join('%d\t' % n for n in j))
                f.write('\n')
            f.close()

            for kt in range(1,no_layers+1):
                if kt!=k:
                    fileName = folder + '/Interdependent_Arcs_%d_%d.txt' % (kt,k)
                    f = open(fileName,"w")
                    for j in selPairs[(kt,k)]:
                        f.write('%d\t%d\tN%d\tN%d\n' % (j[0],j[1],kt,k))
                    f.close()
        #The text file which stores the general information of each configurations
        fileName = folder + '/Overview.txt'
        f = open(fileName,"w")
        f.write('Number of networks = %d\n' % (no_layers))
        text = 'Network types: '
        for k in range(1,no_layers+1):
            text += 'N%d = %s, ' %(k,net_type[k])
        f.write(text+'\n')
        text = 'Number of nodes: '
        for k in range(1,no_layers+1):
            text += 'N%d = %d,' %(k,no_nodes_dict[k])
        f.write(text+'\n')
        text = 'Topology parameter (Grid:X size, ScaleFree:Powerlaw exponent,'+\
            ' Random:Arc existance prob, Tree:diameter, MPG:0): '
        for k in range(1,no_layers+1):
            text += 'N%d = %1.2f,' %(k,topo_param_dict[k])
        f.write(text+'\n')
        text = 'Number of arcs: '
        for k in range(1,no_layers+1):
            text += 'N%d = %d,' %(k,arcs[k].shape[0])
        f.write(text+'\n')
        text = 'Existence probability of interdependent arcs: '
        for key,value in int_prob_dict.items():
            text += '%s = %1.4f,' %(key,value)
        f.write(text+'\n')
        text = 'Number of interdependent arcs: '
        for key,value in no_sel_pairs_dict.items():
            text += '%s = %d,' %(key,value)
        f.write(text+'\n')
        text = 'Damage probability of nodes: '
        for k in range(1,no_layers+1):
            text += 'N%d = %1.4f,' %(k,dam_prob_dict[k])
        f.write(text+'\n')
        text = 'Number of damaged nodes: '
        for k in range(1,no_layers+1):
            text += 'N%d = %d,' %(k,damNodes[k].shape[0])
        f.write(text+'\n')
        text = 'Number of damaged arcs: '
        for k in range(1,no_layers+1):
            text += 'N%d = %d,' %(k,len(damArcs[k]))
        f.write(text+'\n')
        f.write('Available resources for restoration in each time step = %d\n' % (res_cap))
        f.write('Size of the full game (approx.:uniform alloc.+1): %d\n'%size_payoff_matrix)
        f.write('Size of the reduced-sized game (Exhuasting resources, approx.:uniform alloc.+1): %d\n'%size_payoff_matrix_full_resource)
        f.write('Interconnections are chosen randomly\n')
        f.write('Damaged nodes and arcs are chosen randomly\n') # fix in the text #!!!
        f.close()
        # cost of preparation of geographical zones
        g = np.random.randint(low=2000, high=18e4, size=noZones*noZones)
        fileName = folder + '/Zone_prep.txt'
        f = open(fileName,"w")
        for j in g:
            f.write('%d\t' % j)
            f.write('\n')
        f.close()
        
    # Saving network data in the text files
    if write_config:
        fList = open(fileNameList,"a+")
        fList.write('%d\t%d\t%d\t%s\t%1.3f\t%1.2f\t%d\t%s\n' % (cnfg,no_layers,no_nodes,
                                                               str([x for x in topo_param_dict.values()]),
                                                               int_prob,dam_prob,res_cap,
                                                               [x[0] for x in net_type.values()]))
        fList.close()
        print('Configuration %d' % (cnfg))
        cnfg += 1

