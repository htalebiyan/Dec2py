'''
This file contains the functions to generate, store, and plot interdependenct 
random and scale free networks
Hesam Talebiyan - Last updated: 03/08/2018 
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import collections
import math
from itertools import combinations, product

'''
This function yields all pairs of elements come from a number of input lists
'''
def pairs(*lists):
    for t in combinations(lists, 2):
        for pair in product(*t):
            yield pair

'''
This function returns list of nodes, arcs, and position of nodes 
for a randoms network with n nodes connected with probability p.
z denotes the third component of position vector of the nodes which
is necessary for plotting the network.
'''
def Random_network(p,n,z=0): 
    G = nx.erdos_renyi_graph(n, p)
    nodes = np.asanyarray(G.nodes())
    arcs = np.asanyarray(G.edges())
    pos = nx.spring_layout(G)
    posConv = {}
    for i in range(n):
        posConv[i] = (pos[i][0], pos[i][1], z)
#    Plot_Giant_Component(G,n,p)
    return nodes, arcs, posConv

'''
This function returns list of nodes, arcs, position of nodes, and
exponent of the powerlaw for a scale free network with n nodes connected 
with parameter gamma whose (0,0.5) range corresponds to ranf (2,3) range 
of the exponent of the power law which in turn corresponds to 
Ultra-Small World regime.
 Ref: Bollobas, C. Borgs, J. Chayes, and O. Riordan, Directed scale-free graphs,
 Proceedings of the 14th annual ACM-SIAM Symposium on Discrete Algorithms, 132-139, 2003.
plus these assumptions:
    delta_in=0
    delta_out=0
    alpha+beta+gamma = 1
    c1 = (alpha+beta)/(1+delta_in*(alpha+gamma))
    c2 = (gamma+beta)/(1+delta_out*(alpha+gamma))
    expIN = 1+1/c1
    expOUT = 1+1/c2
    and setting expIN=expOUT which yields
    gamma = alpha
    beta = 1-2*gamma
    
z denotes the third component of position vector of the nodes which
is necessary for plotting the network.
'''
def Scale_Free_network(exp,n,z=0):
    gamma = 1-1.0/(exp-1)
    beta = 1-2*gamma
    alpha = 1-beta-gamma
    delta_in=0
    delta_out=0
    G = nx.scale_free_graph(n,alpha,beta,gamma,delta_in,delta_out)
    G = nx.Graph(G)
    
    nodes = np.asanyarray(G.nodes())
    arcs = np.asanyarray(G.edges())
    pos = nx.spring_layout(G)
    posConv = {}
    for i in range(n):
        posConv[i] = (pos[i][0], pos[i][1], z)
#    Plot_Giant_Component(G,n)
#    Plot_Degree_histogram(G,loglogplot=1)
    return nodes, arcs, posConv

'''
This function returns list of nodes, arcs, position of nodes, and
exponent of the powerlaw for a scale free network with n nodes connected 
with parameter gamma whose (0,0.5) range corresponds to ranf (2,3) range 
of the exponent of the power law which in turn corresponds to 
Ultra-Small World regime.
 Ref: Bollobas, C. Borgs, J. Chayes, and O. Riordan, Directed scale-free graphs,
 Proceedings of the 14th annual ACM-SIAM Symposium on Discrete Algorithms, 132-139, 2003.
plus these assumptions:
    delta_in=0
    delta_out=0
    alpha+beta+gamma = 1
    c1 = (alpha+beta)/(1+delta_in*(alpha+gamma))
    c2 = (gamma+beta)/(1+delta_out*(alpha+gamma))
    expIN = 1+1/c1
    expOUT = 1+1/c2
    and setting expIN=expOUT which yields
    gamma = alpha
    beta = 1-2*gamma
    
z denotes the third component of position vector of the nodes which
is necessary for plotting the network.
'''
def Grid_network(gridsizex,gridsizey,z=0):
    G=nx.grid_2d_graph(gridsizex,gridsizey) 
    nodesPos = np.asarray(G.nodes())
    G = nx.convert_node_labels_to_integers(G)
    nodes = np.asarray(G.nodes())
    pos = {}
    mapping = {} 
    for i in range(gridsizex*gridsizey):
        mapping[(nodes[i])] = nodesPos[i,1]*gridsizex + nodesPos[i,0]   
    G=nx.relabel_nodes(G,mapping)    
    nodesID = nodes
    arcs = np.asarray(G.edges())
    pos = nx.spring_layout(G)
    posConv = {}
    for i in range(gridsizex*gridsizey):
        posConv[i] = (pos[i][0], pos[i][1], z)
#    nx.draw(G,pos,node_size=300)
#    nx.draw_networkx_labels(G,pos,dict(zip(nodes, nodes.T)),fontsize=12)
#    plt.show()
#    Plot_Giant_Component(G,gridsizex*gridsizey)
#    Plot_Degree_histogram(G,loglogplot=1)
    return nodes, arcs, posConv
    
'''
This function returns the list of interconnection between two networks.
Each interconnection (out of all possible pairs) has a probability pint 
to be there. Pn and Gn are lists of nodes in each network.
'''
def generate_interconnctions(Pn,Gn,pint):
    allPairs = []
    for pair in pairs(Pn, Gn):
        allPairs.append(pair)
    
    index = range(len(allPairs))   
    nn = len(allPairs)
    ni = int(round(pint*nn))
    selPairs = [allPairs[j] for j in np.random.choice(index,ni)]
    return selPairs

'''
This function plots the degree histogram of the network 
The input value is a networkx object G 
(Option of log-log plot of degree distribution vs. degree is available)
'''            
def Plot_Degree_histogram(G,loglogplot=0):
    degree_sequence = sorted([d for nn, d in G.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    
    fig, ax = plt.subplots()
    if loglogplot==1:
        cnt = [i/float(sum(cnt)) for i in cnt]
        plt.loglog(deg, cnt,'bo')
        plt.title("Degree Distribution Histogram")
        plt.ylabel("Degree probability")
        plt.xlabel("Degree")
    else:
        plt.bar(deg, cnt, width=0.80, color='b')
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)
    
    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
    pos = nx.spring_layout(G)
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
#    plt.savefig("Fig.png",dpi=200)
    plt.show()
    
'''
This function plots the giant component of the network G with n nodes connected
with probability p
'''  
def Plot_Giant_Component(G,n,p=0):
    
    layout = nx.spring_layout
    pos = layout(G)
    
    plt.title("p = %6.3f" % (p))
    nx.draw(G, pos,
            with_labels=False,
            node_size=10
           )
    # identify largest connected component
    Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
    G0 = Gcc[0]
    nx.draw_networkx_edges(G0, pos,
                           with_labels=False,
                           edge_color='r',
                           width=6.0
                          )
    # show other connected components
    for Gi in Gcc[1:]:
        if len(Gi) > 1:
            nx.draw_networkx_edges(Gi, pos,
                                   with_labels=False,
                                   edge_color='r',
                                   alpha=0.3,
                                   width=5.0
                                  )
    plt.show()
 
'''
This function returns the list of all damaged nodes
The list is randomly generated so that the nodes in each network are 
damaged with the same probability prob. 
nVec contains the numebr of nodes in each network
'''     
def random_damage_Data(noNode,noArc,prob):
    damNode = np.random.choice(noNode, int(prob*noNode), replace=False)
    damArc = np.random.choice(noArc, int(prob*noArc), replace=False)
    return damNode,damArc
    