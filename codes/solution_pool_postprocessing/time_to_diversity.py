import _pickle as pickle
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt

sp_file_address = "../../results/inmrp_results_L1_m1000_vb737199t130/solution_pool/solution_pool_t0_.pkl"
with open(sp_file_address, 'rb') as f:
    [elements, solutions] = pickle.load(f)
# Choose a slice of solutions
solution_copy = {}
for key, val in solutions.items():
    if key < 75:
        solution_copy[key] = val

solution_df = pd.DataFrame({'id': elements})
len_rep_elements = []
for idx, sol in solution_copy.items():
    solution_df[idx] = 0
    len_rep_elements.append(len(sol[0]['nodes']))
    for n in sol[0]['nodes']:
        solution_df.loc[solution_df['id'] == str(n), idx] = 1

distance_threshold = np.mean(len_rep_elements)/len(elements)/8

num_of_sol = len(solution_copy)
A = np.zeros((num_of_sol, num_of_sol))
W = np.zeros((num_of_sol, num_of_sol))
for idx1 in solution_copy.keys():
    for idx2 in solution_copy.keys():
        dist = distance.hamming(solution_df[idx1], solution_df[idx2])
        W[idx1, idx2] = dist
        W[idx2, idx1] = dist
        if idx1 < idx2 and dist < distance_threshold:
            A[idx1, idx2] = 1
            A[idx2, idx1] = 1
G = nx.from_numpy_matrix(A)
inp_set = nx.algorithms.mis.maximal_independent_set(G)
print('Size of the independent set:', len(inp_set))
print('Size of the independent set/Number of solutions:', len(inp_set)/num_of_sol)

# Plot Weight Matrix
plt.figure(1)
plt.imshow(W, cmap='Reds')
cbar1 = plt.colorbar()
cbar1.ax.set_ylabel('Hamming Distance', rotation=90)
plt.xlabel('Solution ID')
plt.ylabel('Solution ID')
plt.savefig('weight_matrix.png', dpi=300)
plt.show()
plt.close(1)
# Plot Adjacency Matrix
plt.figure(2)
plt.imshow(A, cmap='Reds')
cbar2 = plt.colorbar()
plt.xlabel('Solution ID')
plt.ylabel('Solution ID')
plt.savefig('adjacency_matrix.png', dpi=300)
plt.show()
plt.close(2)
# Plot His of Number of repaired element across solutions
plt.figure(3)
plt.hist(len_rep_elements)
plt.xlabel('# repaired elements')
plt.ylabel('Frequency')
plt.savefig('num_of_rep_elements.png', dpi=300)
plt.show()
plt.close(2)
# Plot Solution Net
plt.figure(4)
color_map = []
sizes = []
for node in G:
    if node < 1:
        color_map.append('red')
        sizes.append(20)
    else:
        color_map.append('blue')
        sizes.append(5)
pos = nx.spring_layout(G, seed=110)
nx.draw(G, pos=pos, node_color=color_map, with_labels=False, node_size=sizes)
plt.savefig('solution_net.png', dpi=300)
plt.show()
plt.close(4)
