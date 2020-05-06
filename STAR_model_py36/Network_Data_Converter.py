'''
This function converts the format of network data to fit the codes in 
CenteralizedINDP.py, DecenteralizedINDP.py, and InitialNetworkPerformance.py

Hesam Talebiyan - Last updated: 03/08/2018 
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, product
import os
import collections
#from mayavi import mlab

'''
This function converts the format of network data to fit the codes in 
CenteralizedINDP.py, DecenteralizedINDP.py, and InitialNetworkPerformance.py
In these two codes, the nodes of all networks are numbered together so if 
we have n nodes in power net and m nodes in gas net then the nodes are numbered 
from 0 to n+m-1. Accordingly, the numbring format of arcs (and everything else)
changes if to be used in above codes. Also, nodes' numbers should be saved
as string not integer.
 
++ Inputs ++
Nodes = A dictionary of matrices that contains nodes' names, position
coordinates, demands, over- and under-supply penalties for all networks

Arcs = A dictionary of matricesthat contains arcs' names, capacities, 
and flow costs for all networks

IntArcs = A dictionary of matrices that contains all sets of interdependent arcs 
    (format [node number in net 1, node number in net 2])

++ Ouputs ++
nNode = # of nodes in all networks
nArcs = # of arcs in all networks

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
def conv_Data(Nodes,Arcs,IntArcs,g,nets,com,database):

    noNet = len(nets)
    nNode = []
    nArcs = []
    
    convNodes = []
    convPos = []
    convb = {}
    convMp = {}
    convMm = {}
    convq = {}
    
    convArcs = []
    convu = {}
    convc = {}
    convf = {}
    
    convintdpnPairs = {}
    
    startNodeNum = 0
    for k in range(noNet):
        nNode.append(int(Nodes[nets[k]].shape[0]))
        nArcs.append(int(Arcs[nets[k]].shape[0]))      

        for i in range(nNode[k]):
            convNodes.append(str(i+startNodeNum))
            convPos.append([Nodes[nets[k]][i,1],Nodes[nets[k]][i,2]
                                                    ,Nodes[nets[k]][i,3]])
            convb[str(i+startNodeNum),nets[k],com[nets[k]][0]] = Nodes[nets[k]][i][4]
            convMp[str(i+startNodeNum),nets[k],com[nets[k]][0]] = Nodes[nets[k]][i][5]
            convMm[str(i+startNodeNum),nets[k],com[nets[k]][0]] = Nodes[nets[k]][i][6]
            convq[str(i+startNodeNum),nets[k]] = Nodes[nets[k]][i][7]
                  
        for i in range(nArcs[k]):          
            convArcs.append((str(Arcs[nets[k]][i][0]+startNodeNum),
                             str(Arcs[nets[k]][i][1]+startNodeNum)))
            convArcs.append((str(Arcs[nets[k]][i][1]+startNodeNum),
                             str(Arcs[nets[k]][i][0]+startNodeNum)))

            convu[str(Arcs[nets[k]][i][0]+startNodeNum),
                  str(Arcs[nets[k]][i][1]+startNodeNum),
                  nets[k],com[nets[k]][0]] = Arcs[nets[k]][i][2]
            convu[str(Arcs[nets[k]][i][1]+startNodeNum),
                  str(Arcs[nets[k]][i][0]+startNodeNum),
                  nets[k],com[nets[k]][0]] = Arcs[nets[k]][i][2]                  

            convc[str(Arcs[nets[k]][i][0]+startNodeNum),
                  str(Arcs[nets[k]][i][1]+startNodeNum),
                  nets[k],com[nets[k]][0]] = Arcs[nets[k]][i][3]
            convc[str(Arcs[nets[k]][i][1]+startNodeNum),
                  str(Arcs[nets[k]][i][0]+startNodeNum),
                  nets[k],com[nets[k]][0]] = Arcs[nets[k]][i][3]

            convf[str(Arcs[nets[k]][i][0]+startNodeNum),
                  str(Arcs[nets[k]][i][1]+startNodeNum),
                  nets[k]] = Arcs[nets[k]][i][4]
            convf[str(Arcs[nets[k]][i][1]+startNodeNum),
                  str(Arcs[nets[k]][i][0]+startNodeNum),
                  nets[k]] = Arcs[nets[k]][i][4]
                  
        startNodeNum += nNode[k]

    for label,value in IntArcs.items():
        dependee = label[0]
        dependent = label[-1]
        sNNDependee = sum(nNode[0:nets.index(dependee)])
        sNNDependent = sum(nNode[0:nets.index(dependent)])
        convintdpnPairs[label] = []
        if len(IntArcs[label].shape) == 1:
            if IntArcs[label].shape[0]==0:
                pass
            else:
                convintdpnPairs[label].append((str(IntArcs[label][0]+sNNDependee),
                                                str(IntArcs[label][1]+sNNDependent)))            
        else:
            for a in IntArcs[label]:
                convintdpnPairs[label].append((str(a[0]+sNNDependee),
                                            str(a[1]+sNNDependent)))

   
    convg = {}    
    for a in range(g.shape[0]):
        convg[a] = g[a]
        
    zones = {i:[] for i in range(g.shape[0])}
    if database=="synthetic":
        # this part is just for the database type 'synthetic' with two layers
        gridSize = int(g.shape[0]**(.5))
        bins = [-1.0]
        for i in range(gridSize):
            bins.append(bins[i]+2.0/gridSize+0.001)    

                 
        indsX = np.digitize(Nodes['P'][:,1], bins)
        indsY = np.digitize(Nodes['P'][:,2], bins)
        for i in range(nNode[0]):
            zones[(indsX[i]-1)*gridSize+indsY[i]-1].append(str(int(i)))
        indsX = np.digitize(Nodes['G'][:,1], bins)
        indsY = np.digitize(Nodes['G'][:,2], bins)
        for i in range(nNode[1]):
            zones[(indsX[i]-1)*gridSize+indsY[i]-1].append(str(int(i)+nNode[0]))
    elif  database=="ShelbyCounty":
        pass
    else:
        print("You must be kidding me!!!")
        
    return nNode,nArcs,convPos,convNodes,convArcs,convintdpnPairs,convb,convMp,convMm,convq,convu,convc,convf,convg,zones

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
nNode = Number of nodes in each network
DamNodes = A dictionary of matrices that contains damaged nodes' names, and 
    position coordinates for all networks

DamArcs = A dictionary of matrices that contains damaged arcs' names 
    for all networks
    
nets = Label of each network

++ Ouputs ++
convDamNodes = List of all damaged nodes
convDamArcs = List of all damaged arcs

'''  
def conv_Damage_Data(nNode,DamNodes,DamArcs,nets):
    noNet = len(nets)
    
    convDamNodes = []
    convDamArcs = [] 
    startNodeNum = 0
    for k in range(noNet):
        if DamNodes[nets[k]].size:
#            if len(DamNodes[nets[k]].shape) == 1:
#                DamNodes[nets[k]] = np.reshape(DamNodes[nets[k]]
#                                        ,(1,DamNodes[nets[k]].shape[0]))
            for a in np.nditer(DamNodes[nets[k]]):
                convDamNodes.append((str(a+startNodeNum)))
                
        if DamArcs[nets[k]].size:
            if len(DamArcs[nets[k]].shape) == 1:
                DamArcs[nets[k]] = np.reshape(DamArcs[nets[k]]
                                        ,(1,DamArcs[nets[k]].shape[0]))
            for a in DamArcs[nets[k]]:
                convDamArcs.append((str(a[0]+startNodeNum),
                                    str(a[1]+startNodeNum)))    
                convDamArcs.append((str(a[1]+startNodeNum),
                                    str(a[0]+startNodeNum)))
        startNodeNum += nNode[k]
            
    return convDamNodes, convDamArcs

'''
This function computes and plot degree distribution of layers of
 Shelby County network.
'''
def degreeDist_Shelby():
    nets = ['W','G','P']
    folder = 'C:\\Users\\ht20\\Documents\\Files\\DataBase_tdINDP\\MyData\\INDPScenariosNewFormat\\Base_Data\\' 
    loglogplot = 0
    noNet = len(nets)
    Nodes = {}
    Arcs = {}
    IntArcs = {}
    DamNodes = {}
    DamArcs = {}
    # Reading network data from file 
    for k in range(1,noNet+1):
        Nodes[nets[k-1]] = np.loadtxt(folder+'N%d_Nodes.txt' % k)
        Arcs[nets[k-1]] = np.loadtxt(folder+'N%d_Arcs.txt' % k).astype('int')
        for ka in range(1,noNet+1):
            if os.path.isfile(folder+'Interdependent_Arcs_%d_%d.txt' % (k,ka)):
                IntArcs['%s_%s' %(nets[k-1],nets[ka-1])] = np.loadtxt(folder+
                        'Interdependent_Arcs_%d_%d.txt' % (k,ka), 
                        usecols = (0,1)).astype('int')
    #        DamNodes[nets[k-1]] = np.loadtxt(folderDamage+
    #                                'N%d_Damaged_Nodes.txt' % k).astype('int')
    #        DamArcs[nets[k-1]] = np.loadtxt(folderDamage+
    #                                'N%d_Damaged_Arcs.txt' % k).astype('int')
    #g = np.loadtxt(folder+'Zone_prep.txt').astype('int')
    
        G=nx.Graph()
        G.add_nodes_from(Nodes[nets[k-1]][:,0])
        G.add_edges_from(Arcs[nets[k-1]][:,0:2])
        
        degree_sequence = sorted([d for nn, d in G.degree()], reverse=True)  # degree sequence
        # print "Degree sequence", degree_sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        
        fig, ax = plt.subplots()
        if loglogplot==1:
            cnt = [i/float(sum(cnt)) for i in cnt]
            plt.loglog(deg, cnt,'bo')
            plt.title('Network %s' % nets[k-1])
            plt.ylabel("Degree probability")
            plt.xlabel("Degree")
        else:
            plt.bar(deg, cnt, width=0.80, color='b')
            plt.title('Network %s' % nets[k-1])
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