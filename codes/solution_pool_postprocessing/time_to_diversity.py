import _pickle as pickle
import os
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore')

sp_file_address = "../../results/Solution_Pool/" \
                  "dp_inmrp_results_L1_m1000_vb737199t65_sample_2/solution_pool/solution_pool_t0_.pkl"
output_folder = './dp_inmrp_results_L1_m1000_vb737199t65_sample_2/'
with open(sp_file_address, 'rb') as f:
    [elements, solutions] = pickle.load(f)

solution_df = pd.DataFrame({'id': elements})
len_rep_elements = []
for idx, sol in solutions.items():
    solution_df[idx] = 0
    len_rep_elements.append(len(sol[0]['nodes']))
    for n in sol[0]['nodes']:
        solution_df.loc[solution_df['id'] == str(n), idx] = 1
distance_threshold = np.mean(len_rep_elements) / len(elements) / 8

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
results = {}
for num_sol in [50, 75, 100, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000]:
    # Choose a slice of solutions
    solution_copy = {}
    count = 0
    for key, val in solutions.items():
        if count < num_sol:
            solution_copy[key] = val
        count += 1

    num_of_sol = len(solution_copy)
    # assert num_of_sol == num_sol, 'number of solutions ERROR'
    A = np.zeros((num_of_sol, num_of_sol))
    W = np.zeros((num_of_sol, num_of_sol))
    count1 = 0
    for idx1 in solution_copy.keys():
        count2 = 0
        for idx2 in solution_copy.keys():
            if idx1 < idx2:
                dist = distance.hamming(solution_df[idx1], solution_df[idx2])
                W[count1, count2] = dist
                W[count2, count1] = dist
                if dist < distance_threshold:
                    A[count1, count2] = 1
                    A[count2, count1] = 1
            count2 += 1
        count1 += 1
    G = nx.from_numpy_matrix(A)
    inp_set = nx.algorithms.mis.maximal_independent_set(G, seed=1)
    results[num_sol] = [len(inp_set), len(inp_set) / num_of_sol]
    print(num_sol, 'Size of the independent set:', len(inp_set))
    print(num_sol, 'Size of the independent set/Number of solutions:', len(inp_set) / num_of_sol)

    # # Plot Weight Matrix
    # plt.figure(1)
    # plt.imshow(W, cmap='Reds')
    # cbar1 = plt.colorbar()
    # cbar1.ax.set_ylabel('Hamming Distance', rotation=90)
    # plt.xlabel('Solution ID')
    # plt.ylabel('Solution ID')
    # plt.savefig(output_folder + 'weight_matrix_' + str(num_sol) + '.png', dpi=300)
    # plt.show()
    # plt.close(1)
    # # # Plot Adjacency Matrix
    # # plt.figure(2)
    # # plt.imshow(A, cmap='Reds')
    # # cbar2 = plt.colorbar()
    # # plt.xlabel('Solution ID')
    # # plt.ylabel('Solution ID')
    # # plt.savefig('adjacency_matrix.png', dpi=300)
    # # plt.show()
    # # plt.close(2)
    # # Plot His of Number of repaired element across solutions
    # plt.figure(3)
    # plt.hist(len_rep_elements)
    # plt.xlabel('# repaired elements')
    # plt.ylabel('Frequency')
    # plt.savefig(output_folder + 'num_of_rep_elements_' + str(num_sol) + '.png', dpi=300)
    # plt.show()
    # plt.close(2)
    # # Plot Solution Net
    # plt.figure(4)
    # color_map = []
    # sizes = []
    # for node in G:
    #     if node < 1:
    #         color_map.append('red')
    #         sizes.append(20)
    #     else:
    #         color_map.append('blue')
    #         sizes.append(5)
    # pos = nx.spring_layout(G, seed=110)
    # nx.draw(G, pos=pos, node_color=color_map, with_labels=False, node_size=sizes)
    # plt.savefig(output_folder + 'solution_net_' + str(num_sol) + '.png', dpi=300)
    # plt.show()
    # plt.close(4)
with open(output_folder + 'results.pkl', 'wb') as f:
    pickle.dump(results, f)
