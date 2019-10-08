'''
This code evalute and control the quality of generated dataset of 
interdependent networks and plots which show the connectedness, 
average connectivity, and percentage of met demand in the overall 
interdependent network

Based on INDP implementation by Camilo Gomez, Andres Gonzalez
Hesam Talebiyan - Last updated: 03/04/2018 
'''  
import numpy as np
import seaborn as sns
from gurobipy import *
#import Plot
import Network_Data_Generator
import networkx as nx
import matplotlib.pyplot as plt
import os 

'''
This function solve INDP for two undamaged networks to find the total value of 
unmet demand in the undamaged networks
'''  
def initialPerformanceUndamagedNetwork(nP,nG,aP,aG,Pos,Nodes,Arcs,intArcs,bVec,MpVec,MmVec,uVec,cVec,doPlot=0):

    '''	INITIALIZATIONS
    '''
    # location of nodes for both networks
    coo = Pos
    # Sets
    noP = nP
    noG = nG
    nodes = Nodes	# All network nodes
    arcs = Arcs
    nets = ['P','G']
    commodities = {'power':'power', 'gas':'gas'}
    com = {'P':['power'], 'G':['gas']}
    dam_arcs = []
    dam_nodes = []
    # Subsets
    subnodes = {'P':[str(i) for i in range(noP)], 'G':[str(i) for i in range(noP,noP+noG)]} 	# Subset of nodes that belong to each network
    subarcs = {k:[] for k in nets} 									# Subset of arcs that belong to each network
    for k in nets:
    	for (i,j) in arcs:
    		if i in subnodes[k] and j in subnodes[k]:
    			subarcs[k].append((i,j))
    
    # Obj. Costs				# Cost of fixing node i.j of network k
    M_p = {(i,k,l):MpVec[i,k,l] for i in nodes for k in nets for l in commodities if i in subnodes[k] and l in com[k]} 			# Penalty for excess
    M_m = {(i,k,l):MmVec[i,k,l] for i in nodes for k in nets for l in commodities if i in subnodes[k] and l in com[k]} 			# Penalty for shortage
    c = {(i,j,k,l):cVec[i,j,k,l] for (i,j) in arcs for k in nets for l in commodities if (i,j) in subarcs[k] and l in com[k]} 		# Flow cost
    
    # Coefs & Right-hand Sid
    # Demands at nodes (fictional)
    b = {(i,k,l):bVec[i,k,l] for i in nodes for k in nets for l in commodities if i in subnodes[k] and l in com[k]}
    # arc capacities
    u = {(i,j,k,l):uVec[i,j,k,l] for (i,j) in arcs for k in nets for l in commodities if (i,j) in subarcs[k] and l in com[k]} 					# Capacity of arc i,j in network k    
    
    # gamma: is it necessary that node i (net k) k is functional for node j (net ka) to work?
    gamma = {(i,j,k,ka):0 for k in nets for ka in nets for i in nodes for j in nodes if i in subnodes[k] if j in subnodes[ka] if i!=j if k!=ka}

    for i in range(len(intArcs)):
        gamma[str(intArcs[i][0]),str(intArcs[i][1]),'P','G'] = 1  
        
    #    pprint(dam_nodes)
    #    pprint(dam_arcs)
    
    m = Model("INDP1")
    
    '''
    	VARIABLES
    '''
    x = {} 		# flow through arc (i,j) of network k, commodity l
    d_p = {} 	# flow excess at node i, network k, commodity l
    d_m = {} 	# flow defect at node i, network k, commodity l
    
    for k in nets:
    	for i in subnodes[k]:
    		for l in com[k]:
    			d_p[i,k,l] = m.addVar(vtype=GRB.CONTINUOUS, name="Mp_"+str((i,k,l)), obj=M_p[i,k,l])
    			d_m[i,k,l] = m.addVar(vtype=GRB.CONTINUOUS, name="Mm_"+str((i,k,l)), obj=M_m[i,k,l])
    	for (i,j) in subarcs[k]:
    		for l in com[k]:
    			x[i,j,k,l] = m.addVar(vtype=GRB.CONTINUOUS, name="x_"+str((i,j,k,l)), obj=c[i,j,k,l])
    m.update()
    
    
    '''
    	CONSTRAINTS
    '''
    # Flow conservation
    for k in nets:
    	for i in subnodes[k]:
    		for l in com[k]:
    			outflow = quicksum(x[i,j,k,l] for j in nodes if (i,j) in subarcs[k])
    			inflow = quicksum(x[j,i,k,l] for j in nodes if (j,i) in subarcs[k])
    			m.addConstr(outflow-inflow == b[i,k,l] - d_p[i,k,l] + d_m[i,k,l], name="flow_"+str((i,k,l)))
    
    # Flow due to component availability
    for k in nets:
        for (i,j) in subarcs[k]:
            for l in com[k]:
                arcflow = x[i,j,k,l]
                if (i,j) in dam_arcs:
                    m.addConstr(arcflow <= u[i,j,k,l] * 0.0, name="avaArc_"+str((i,j,k,l)))
                else:
                    m.addConstr(arcflow <= u[i,j,k,l], name="avaTail_"+str((i,j,k,l)))
    
    # No need for Interdependence constraints since all nodes in other network work
   
    
    
    
    '''
    	OBJ & OUTPUT
    '''
    m.update()
    m.setParam('OutputFlag',0)
    m.optimize()
    lname = "Model.lp" 
    m.write(lname)         
    
    
    if doPlot:     
         Plot.plotGraph(100, '1-Initial', coo, arcs, intArcs, nP, nG, aP, aG)
    
#    soltFOa = sum([x[i,j,k,l].x*c[i,j,k,l] for (i,j) in arcs for k in nets for l in commodities if (i,j) in subarcs[k] and l in com[k]])		# Flow cost
#    soltFOT = (m.getObjective()).getValue()
#    soltFO2 = sum([d_p[i,k,l].x*M_p[i,k,l]+d_m[i,k,l].x*M_m[i,k,l] for i in nodes for k in nets for l in commodities if i in subnodes[k] and l in com[k]]) 			# Penalty for excess
#    soltFO = soltFOT - soltFO2
    dmSum= sum([i.x for i in d_m.values()])
#    print gamma.values()
#    print [i.x for i in d_m.values()]
#    print [i.x for i in d_p.values()]
#    print x
#    print dam_nodes
#    print dam_arcs
    return dmSum
'''
This function plots which show the connectedness, average connectivity,
and percentage of met demand in the overall interdependent network
'''
def plot_Evaluation_Results(noConfig, dmRatioSum, aveConnP, allPairsConnP, aveConnG, allPairsConnG, folder):

    if not os.path.exists(folder):
        os.makedirs(folder) 
    plt.figure(figsize=(10, 6), dpi=100)     
    H, bins = np.histogram(dmRatioSum)
    width = 0.5 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, H, align='center', width=width)
    plt.title("Histogram of % of met demand for all networks")
    plt.xlabel('% met demand')
    plt.ylabel('Frequency')
    plt.savefig(folder+'HistdmRatio.png')    
    plt.clf()
    plt.bar(range(noConfig),np.mean(dmRatioSum, axis=(1,2)))
    plt.title("Average of % of met demand for each configuration")
    plt.xlabel('Configuration Number')
    plt.ylabel('% met demand')
    plt.savefig(folder+'AveragedmRatio.png')
    plt.clf()  
#    H, bins = np.histogram(aveConnP)
#    width = 0.7 * (bins[1] - bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    plt.bar(center, H, align='center', width=width)
#    plt.title("Histogram of average connectivity for all power networks")
#    plt.xlabel('average connectivity in network 1')
#    plt.ylabel('Frequency')
#    plt.savefig(folder+'HistAveConnPower.png')        
#    plt.clf()
#    H, bins = np.histogram(aveConnG)
#    width = 0.7 * (bins[1] - bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    plt.bar(center, H, align='center', width=width)
#    plt.title("Histogram of average connectivity for all gas networks")
#    plt.xlabel('average connectivity in network 2')
#    plt.ylabel('Frequency')
#    plt.savefig(folder+'HistAveConnGas.png')      
    H, bins = np.histogram(aveConnP)
    width = 0.1 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2 + 0.005
    f1 = H
    plt.bar(center,f1,color='b',align='center',label='Network 1',width=width)
    H, bins = np.histogram(aveConnG)
    center = (bins[:-1] + bins[1:]) / 2 - 0.005
    f2 = H
    plt.bar(center,f2,color='r',align='center',label='Network 2',width=width)
    plt.title("Histogram of average connectivity for all networks")
    plt.xlabel('Average connectivity')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.savefig(folder+'HistAveConn.png') 
    plt.clf()
#    plt.bar(range(noConfig),np.mean(aveConnP, axis=(1,2)))
#    plt.title("Average of average connectivity of power net for each configuration")
#    plt.xlabel('Configuration Number')
#    plt.ylabel('Average of average connectivity')
#    plt.savefig(folder+'AverageAveConnPower.png')
#    plt.clf()
#    plt.bar(range(noConfig),np.mean(aveConnG, axis=(1,2)))
#    plt.title("Average of average connectivity of gas net for each configuration")
#    plt.xlabel('Configuration Number')
#    plt.ylabel('Average of average connectivity')
#    plt.savefig(folder+'AverageAveConnGas.png')   
    plt.figure(figsize=(20, 6), dpi=100)   
    plt.bar(np.array(range(noConfig))+.1,np.mean(aveConnP,axis=(1,2)),width=0.25,
            color='b',align='center',label='Network 1')
    plt.bar(np.array(range(noConfig))-.1,np.mean(aveConnP, axis=(1,2)),width=0.25,
            color='r',align='center',label='Network 2')
    plt.title("Average of average connectivity of networks for each configuration")
    plt.xlabel('Configuration Number')
    plt.ylabel('Average of average connectivity')
    plt.xticks(range(0,noConfig,5))
    plt.legend(loc='best')
    plt.savefig(folder+'AverageAveConn.png') 
    plt.clf()
#    plt.bar(range(noConfig),np.mean(allPairsConnP, axis=(1,2)))
#    plt.title("Average of connectedness of power net for each configuration")
#    plt.xlabel('Configuration Number')
#    plt.ylabel('Average of connectedness')
#    plt.savefig(folder+'AverageConnectednessPower.png') 
#    plt.clf()
#    plt.bar(range(noConfig),np.mean(allPairsConnG, axis=(1,2)))
#    plt.title("Average of connectedness of gas net for each configuration")
#    plt.xlabel('Configuration Number')
#    plt.ylabel('Average of connectedness')
#    plt.savefig(folder+'AverageConnectednessGas.png') 
    plt.bar(np.array(range(noConfig))+0.1,np.mean(allPairsConnP,axis=(1,2)),
            width=0.25,color='b',align='center',label='Network 1')
    plt.bar(np.array(range(noConfig))-0.1,np.mean(allPairsConnG, axis=(1,2)),
            width=0.25,color='r',align='center',label='Network 2')
    plt.title("Average of connectedness of networks for each configuration")
    plt.xlabel('Configuration Number')
    plt.ylabel('Average of connectedness')
    plt.xticks(range(0,noConfig,5))
    plt.legend(loc='best')
    plt.savefig(folder+'AverageConnectedness.png') 

    
    
# Input values
noSampleSets = 10
noIterations = 30
rootfolder = 'C:\Users\ht20\Documents\Files\Generated_Network_Dataset\\' # Root folder where the database is
rootfolder += 'GridNetworks\\' # choose relevant dataset folder
#options: 'RandomNetworks\\'|'ScaleFreeNetworks\\'|'GridNetworks\\'
NetworkTypeInitial = 'GN' #Options: RN|SFN|GN

# Read configuration data
fileNameList = rootfolder + 'List_of_Configurations.txt' 
configList = np.loadtxt(fileNameList)

# 3D arrays to store evaluation results
dmRatioSum = np.zeros((len(configList),noSampleSets,noIterations))
aveConnP = np.zeros((len(configList),noSampleSets,noIterations))
allPairsConnP = np.zeros((len(configList),noSampleSets,noIterations)) 
aveConnG = np.zeros((len(configList),noSampleSets,noIterations))
allPairsConnG = np.zeros((len(configList),noSampleSets,noIterations))
noInterconnections = np.zeros(len(configList))
 
for i in configList: 
    # Reading configuration data from file 
    cnfg = int(i[0])
    noNodes = i[1]
    arcProb = i[2] # exp for scale free net; gridsizex for Grid net
    intConProb = i[3]
    damProb = i[4]
    resCap = i[5]
    foldername = NetworkTypeInitial+'Config%d_%d_%1.2f_%1.3f_%1.2f_%d/' % (cnfg,noNodes,arcProb,intConProb,damProb,resCap)
#    for s in range(noSampleSets):
#        for j in range(noIterations):
#            # Reading network data from file 
#            subfolder = rootfolder+foldername+'SampleSet_%d/IN_%d/' % (s,j)
#            PNodes = np.loadtxt(subfolder+'N1_Nodes.txt')
#            PArcs = np.loadtxt(subfolder+'N1_Arcs.txt').astype('int')
#            GNodes = np.loadtxt(subfolder+'N2_Nodes.txt')
#            GArcs = np.loadtxt(subfolder+'N2_Arcs.txt') .astype('int')
#            IntArcs = np.loadtxt(subfolder+'Interdependent_Arcs_1_2.txt', usecols = (0,1)).astype('int')
#            g = np.loadtxt(subfolder+'Zone_prep.txt').astype('int')
#    
#            # Computoing the connectedness and average connectivity for each
#            # network
#            G = nx.Graph()
#            G.add_nodes_from(PNodes[:,0])
#            G.add_edges_from(PArcs[:,0:2])
#            aveConnP[cnfg,s,j] = nx.average_node_connectivity(G)
#            allPairsConnP[cnfg,s,j] = int(nx.is_connected(G))
#            G = nx.Graph()
#            G.add_nodes_from(GNodes[:,0])
#            G.add_edges_from(GArcs[:,0:2])
#            aveConnG[cnfg,s,j] = nx.average_node_connectivity(G)
#            allPairsConnG[cnfg,s,j] = int(nx.is_connected(G))
#            # Computoing the the percentage of demand which is met
#            # for each network            
#            nP,nG,aP,aG,convPos,convNodes,convArcs,convintdpnPairs,convb,convMp,convMm,convq,convu,convc,convf,convg,zones = Network_Data_Generator.conv_Data(PNodes,PArcs,GNodes,GArcs,IntArcs,g)
#            dmSum = initialPerformanceUndamagedNetwork(nP,nG,aP,aG,convPos,convNodes,convArcs,convintdpnPairs,convb,convMp,convMm,convu,convc)
#            alldemand = sum([convb[y] for y in convb.keys() if convb[y]<0])
#            dmRatioSum[cnfg,s,j] = 1.0+dmSum/alldemand
#            
#            print 'Config %d | SampleSet_%d | IN_%d was processed' % (cnfg,s,j)
            
    # Number of Interconnections
    noInterconnections[cnfg] = round(noNodes*noNodes*intConProb)
            
            
            
#    # Plot QUality  Control results            
#    qcFolder = rootfolder+'Evaluations/'
#    plot_Evaluation_Results(len(configList),dmRatioSum, aveConnP, allPairsConnP, aveConnG, allPairsConnG,qcFolder)           
            
sns.set()     
ax = sns.distplot(noInterconnections,rug=True,kde=False)
ax.set(xlabel='Number of Interconnections', ylabel='Probability (%)')
plt.savefig('noInterconnections_'+NetworkTypeInitial+'.png',dpi=600,bbox_inches="tight")