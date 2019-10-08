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
THIS IS AN OLD VERSION WHICH IS MODIFIED LATER. LOOK AT THE MAIN CODES
This function converts the format of network data to fit the codes in 
CenteralizedINDP.py, DecenteralizedINDP.py, and InitialNetworkPerformance.py
In these two codes, the nodes of all networks are numbered together so if 
we have n nodes in power net and m nodes in gas net then the nodes are numbered 
from 0 to n+m-1. Accordingly, the numbring format of arcs (and everything else)
changes if to be used in above codes. Also, nodes' numbers should be saved
as string not integer.
 
++ Inputs ++
PNodes = A matrix that contains nodes' names, position coordinates, demands,
 over- and under-supply penalties of network 1 (power network)
GNodes = A matrix that contains nodes' names, position coordinates, demands, 
 over- and under-supply penalties of nodes of network 2 (gas network)
PArcs = A matrix that contains arcs' names, capacities, and flow costs
 of network 1 (power network)
GArcs = A matrix that contains arcs' names, capacities, and flow costs
 of network 2 (gas network)
IntArcs = set of interdependent arcs 
    (format [node number in net 1, node number in net 2])

++ Ouputs ++
nP = # of nodes in power net
nG = # of nodes in gas net
aP = # of arcs in power net
aG = # of arcs in gas net

convPos = List of position vectors of all nodes
convNodes = List of all nodes
convArcs = List of all arcs
convintdpnPairs = List of all interdependent arcs

convb = dictionary of all node demands. keys = (node number,network,commodity)
convMp = = dictionary of all node oversupply penalties.  keys = (node number,network,commodity)
convMm = = dictionary of all node undersupply penalties.  keys = (node number,network,commodity)
convq = = dictionary of all node reconstruction costs.  keys = (node number,network,commodity)

convu = dictionary of all arc capacities. keys = (node 1 number,node 2 number,network,commodity)
convc = dictionary of all arc flow costs. keys = (node 1 number,node 2 number,network,commodity)
convf = = dictionary of all arc reconstruction costs. keys = (node 1 number,node 2 number,network,commodity)
'''  
def conv_Data(PNodes,PArcs,GNodes,GArcs,IntArcs, g):
    nP = int(PNodes.shape[0])
    nG = int(GNodes.shape[0])
    aP = int(PArcs.shape[0])
    aG = int(GArcs.shape[0])

    convNodes = []
    for a in range(nP+nG):
        convNodes.append((str(a)))
        
    convPos = []
    for i in range(nP):
        convPos.append([PNodes[i,0],PNodes[i,1],PNodes[i,2]])
    for i in range(nG):
        convPos.append([GNodes[i,0],GNodes[i,1],GNodes[i,2]])
        
    convArcs = []
    for a in PArcs:
        convArcs.append((str(a[0]),str(a[1])))
        convArcs.append((str(a[1]),str(a[0])))
    for a in GArcs:
        convArcs.append((str(a[0]+nP),str(a[1]+nP)))
        convArcs.append((str(a[1]+nP),str(a[0]+nP)))

    convintdpnPairs = []
    if len(IntArcs.shape) == 1:
        if IntArcs.shape[0]==0:
            pass
        else:
            convintdpnPairs.append((str(IntArcs[0]),str(IntArcs[1]+nP)))            
    else:
        for a in IntArcs:
            convintdpnPairs.append((str(a[0]),str(a[1]+nP)))

    convb = {}
    for a in range(nP):
        convb[str(a),'P','power'] = PNodes[a][4]
    for a in range(nG):
        convb[str(a+nP),'G','gas'] = GNodes[a][4]
        
    convMp = {}
    for a in range(nP):
        convMp[str(a),'P','power'] = PNodes[a][5]
    for a in range(nG):
        convMp[str(a+nP),'G','gas'] = GNodes[a][5]
        
    convMm = {}
    for a in range(nP):
        convMm[str(a),'P','power'] = PNodes[a][6]
    for a in range(nG):
        convMm[str(a+nP),'G','gas'] = GNodes[a][6] 

    convq = {}
    for a in range(nP):
        convq[str(a),'P'] = PNodes[a][7]
    for a in range(nG):
        convq[str(a+nP),'G'] = GNodes[a][7]
               
    convu = {}
    for a in range(aP):
        convu[str(PArcs[a][0]),str(PArcs[a][1]),'P','power'] = PArcs[a][2]
        convu[str(PArcs[a][1]),str(PArcs[a][0]),'P','power'] = PArcs[a][2]
    for a in range(aG):
        convu[str(GArcs[a][0]+nP),str(GArcs[a][1]+nP),'G','gas'] = GArcs[a][2] 
        convu[str(GArcs[a][1]+nP),str(GArcs[a][0]+nP),'G','gas'] = GArcs[a][2]
              
    convc = {}
    for a in range(aP):
        convc[str(PArcs[a][0]),str(PArcs[a][1]),'P','power'] = PArcs[a][3]
        convc[str(PArcs[a][1]),str(PArcs[a][0]),'P','power'] = PArcs[a][3]
    for a in range(aG):
        convc[str(GArcs[a][0]+nP),str(GArcs[a][1]+nP),'G','gas'] = GArcs[a][3]
        convc[str(GArcs[a][1]+nP),str(GArcs[a][0]+nP),'G','gas'] = GArcs[a][3]

    convf = {}
    for a in range(aP):
        convf[str(PArcs[a][0]),str(PArcs[a][1]),'P'] = PArcs[a][4]
        convf[str(PArcs[a][1]),str(PArcs[a][0]),'P'] = PArcs[a][4]
    for a in range(aG):
        convf[str(GArcs[a][0]+nP),str(GArcs[a][1]+nP),'G'] = GArcs[a][4]
        convf[str(GArcs[a][1]+nP),str(GArcs[a][0]+nP),'G'] = GArcs[a][4]
    
    convg = {}    
    for a in range(g.shape[0]):
        convg[a] = g[a]
        
    noZones = int(g.shape[0]**(.5))
    bins = [-1.0]
    for i in range(noZones):
        bins.append(bins[i]+2.0/noZones+0.001)    
    zones = {i:[] for i in range(g.shape[0])}
             
    indsX = np.digitize(PNodes[:,1], bins)
    indsY = np.digitize(PNodes[:,2], bins)
    for i in range(nP):
        zones[(indsX[i]-1)*noZones+indsY[i]-1].append(str(int(i)))
    indsX = np.digitize(GNodes[:,1], bins)
    indsY = np.digitize(GNodes[:,2], bins)
    for i in range(nG):
        zones[(indsX[i]-1)*noZones+indsY[i]-1].append(str(int(i)+nP))
    return nP,nG,aP,aG,convPos,convNodes,convArcs,convintdpnPairs,convb,convMp,convMm,convq,convu,convc,convf,convg,zones

'''
This function returns the list of all damaged nodes
The list is randomly generated so that the nodes in each network are 
damaged with the same probability prob. 
nVec contains the numebr of nodes in each network
'''     
def random_damage_Data(noNode,prob):
    damNode = np.random.choice(noNode, int(prob*noNode), replace=False)
    return damNode
    
'''
This function converts the format of network damage data to fit the codes in 
CenteralizedINDP.py, DecenteralizedINDP.py, and InitialNetworkPerformance.py
In these two codes, the nodes of all networks are numbered together so if 
we have n nodes in power net and m nodes in gas net then the nodes are numbered 
from 0 to n+m-1. Accordingly, the numbring format of arcs (and everything else)
changes if to be used in above codes. Also, nodes' numbers should be saved
as string not integer.
 
++ Inputs ++
nP = # of nodes in power net
PDamNodes = A matrix that contains damaged nodes' names, and 
    position coordinates of network 1 (power network)
GDamNodes = A matrix that contains damaged nodes' names, and 
    position coordinates of network 2 (gas network)
PDamArcs = A matrix that contains damaged arcs' names of network 1 (power network)
GDamArcs = A matrix that contains damaged arcs' names of network 2 (gas network)


++ Ouputs ++
convDamNodes = List of all damaged nodes
convDamArcs = List of all damaged arcs

'''  
def conv_Damage_Data(nP,PDamNodes,PDamArcs,GDamNodes,GDamArcs):
    convDamNodes = []
    if PDamNodes.size: 
        if len(PDamNodes.shape) == 1:
            PDamNodes = np.reshape(PDamNodes, (1,PDamNodes.shape[0]))
        for a in PDamNodes:
            convDamNodes.append((str(a[0])))  
    if GDamNodes.size: 
        if len(GDamNodes.shape) == 1:
            GDamNodes = np.reshape(GDamNodes, (1,GDamNodes.shape[0]))
        for a in GDamNodes:
            convDamNodes.append((str(a[0]+nP)))  
            
    convDamArcs = []    
    if PDamArcs.size: 
        if len(PDamArcs.shape) == 1:
            PDamArcs = np.reshape(PDamArcs, (1,PDamArcs.shape[0]))
        for a in PDamArcs:
            convDamArcs.append((str(a[0]),str(a[1])))    
            convDamArcs.append((str(a[1]),str(a[0])))     
    if GDamArcs.size: 
        if len(GDamArcs.shape) == 1:
            GDamArcs = np.reshape(GDamArcs, (1,GDamArcs.shape[0]))
        for a in PDamArcs:
            convDamArcs.append((str(a[0]+nP),str(a[1]+nP)))    
            convDamArcs.append((str(a[1]+nP),str(a[0]+nP)))           
            
    return convDamNodes, convDamArcs