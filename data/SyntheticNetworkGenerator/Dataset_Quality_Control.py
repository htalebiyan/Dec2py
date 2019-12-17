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
import Network_Data_Generator
import networkx as nx
import matplotlib.pyplot as plt
import os
import networkx as nx
import os 
import pandas as pd

def load_array_format_extended(BASE_DIR="C:\\Users\\ht20\Documents\\Files\Generated_Network_Dataset_v3\\RandomNetworks\\",
                               topo='RN',config=0,sample=0,cost_scale=1.0):
    file_dir = BASE_DIR+topo+'Config_'+str(config)+'\\Sample_'+str(sample)+'\\'
    with open(BASE_DIR+'List_of_Configurations.txt') as f:
            data = pd.read_csv(f, delimiter='\t')    
    config_param = data.iloc[config]
    noLayers = int(config_param.loc[' No. Layers'])
    G=nx.DiGraph()
    
    files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
    pos = {}
    dam_nodes = []
    dam_arcs = []
    z_offsetx = 0.25
    z_offsety = 3
    for k in range(1,noLayers+1):
        for file in files: 
            if file=='N'+str(k)+'_Nodes.txt':
                with open(file_dir+file) as f:
    #                print "Opened",file,"."
                    data = pd.read_csv(f, delimiter='\t',header=None)
                    for v in data.iterrows():  
                        node_data = {'b': v[1][4],'Mp': v[1][5],'Mm': v[1][6],'q': v[1][7]}
                        node_name = (int(v[1][0]),k)
                        G.add_node(node_name)
                        G.node[node_name]['data'] = node_data
                        pos[node_name] = (v[1][1]+v[1][3]*z_offsetx,v[1][2]+v[1][3]*z_offsety)
        for file in files:
            if file=='N'+str(k)+'_Arcs.txt':
                with open(file_dir+file) as f:
    #                print "Opened",file,"."
                    data = pd.read_csv(f, delimiter='\t',header=None)
                    for v in data.iterrows(): 
                        arc_data = {'u': v[1][2],'c': v[1][3],'f': v[1][4]}
                        G.add_edge((int(v[1][0]),k),(int(v[1][1]),k))
                        G[(int(v[1][0]),k)][(int(v[1][1]),k)]['data'] = arc_data
                        G.add_edge((int(v[1][1]),k),(int(v[1][0]),k))
                        G[(int(v[1][1]),k)][(int(v[1][0]),k)]['data'] = arc_data
        for kt in range(noLayers):
            if k!=kt:
                for file in files:
                    if file=='Interdependent_Arcs_'+str(k)+'_'+str(kt)+'.txt':
                        with open(file_dir+file) as f:
                #                print "Opened",file,"."
                            try:
                                data = pd.read_csv(f, delimiter='\t',header=None)
                                for v in data.iterrows():   
                                    G.add_edge((int(v[1][0]),k),(int(v[1][1]),kt)) 
                            except:
                                print('Empty file: '+ file)
        for file in files: 
            if file=='N'+str(k)+'_Damaged_Nodes.txt':
                with open(file_dir+file) as f:
                    try:
                        data = pd.read_csv(f, delimiter='\t',header=None)
                        for v in data.iterrows():               
                            dam_nodes.append((int(v[1][0]),k))
                    except:
                        print('Empty file: '+ file)
                                
        for file in files:
            if file=='N'+str(k)+'_Damaged_Arcs.txt':
                with open(file_dir+file) as f:
                    try:
                        data = pd.read_csv(f, delimiter='\t',header=None)
                        for v in data.iterrows():   
                            dam_arcs.append(((int(v[1][0]),k),(int(v[1][1]),k)))
                    except:
                        print('Empty file: '+ file)
#    pos=nx.spring_layout(G)   
##    for key,value in pos.items():                     
##        pos[key][0] += key[1]*z_offsetx
##        pos[key][1] += key[1]*z_offsety   
    return G,pos,noLayers,dam_nodes,dam_arcs

def plot_network(BASE_DIR="C:\\Users\\ht20\Documents\\Files\Generated_Network_Dataset_v3\\RandomNetworks\\",
                 topo='RN',config=0,sample=0,cost_scale=1.0):
    plt.close('all')
    plt.figure(figsize=(10,8))  
    
    G,pos,noLayers,dam_nodes,dam_arcs = load_array_format_extended(BASE_DIR,topo,config,sample,cost_scale)  
    labels = {}
    for n,d in G.nodes(data=True):
        labels[n]= "%d" % (n[0])
    pos_moved={}
    for key,value in pos.items():
        pos_moved[key] = [0,0]
        pos_moved[key][0] = pos[key][0]-0.2
        pos_moved[key][1] = pos[key][1]+0.2
    #nx.draw(G, pos,node_color='w')
    #nx.draw_networkx_labels(G,labels=labels,pos=pos,
    #                        font_color='w',font_family='CMU Serif',font_weight='bold')
    
    clr=['r','b','g','m']
    for k in range(noLayers):
        node_list = [x for x in G.nodes() if x[1]==k]
        nx.draw_networkx_nodes(G,pos,nodelist=node_list,node_color=clr[k],node_size=70,alpha=0.9)
    for k in range(noLayers):
        arc_dict = [x for x in G.edges() if x[0][1]==k and x[1][1]==k]
        nx.draw_networkx_edges(G,pos,edgelist=arc_dict,width=1,alpha=0.25,edge_color=clr[k])
        interarc_dict = [x for x in G.edges() if x[0][1]==k and x[1][1]!=k]
        nx.draw_networkx_edges(G,pos,edgelist=interarc_dict,width=1,alpha=0.25,edge_color='k')
    #nx.draw_networkx_nodes(G,pos,nodelist=dam_nodes,node_color='w',node_shape="x",node_size=35)
    #nx.draw_networkx_edges(G,pos,edgelist=dam_arcs,width=1,alpha=1,edge_color='w',style='dashed')
    plt.tight_layout()   
    plt.axis('off')
    #plt.savefig(output_dir+'/plot_net'+folderSuffix+'.png',dpi=600)

'''
This function solve INDP for two undamaged networks to find the total value of 
unmet demand in the undamaged networks
'''  
def initialPerformanceUndamagedNetwork(G, noLayers):
    '''
	INITIALIZATIONS
    '''
    # Sets
    nodes = list(G.nodes(data=False))	# All network nodes
    arcs = list(G.edges(data=False))

    nets = [x for x in range(noLayers)]
    commodities = {x:x for x in nets}
    com = {x:[x] for x in nets}
    dam_arcs = []
    dam_nodes = []
    # Subsets
    subnodes = {x:[i[0] for i in nodes if i[1]==x] for x in nets} 	# Subset of nodes that belong to each network
    noNodes = {x:len(lst) for x,lst in subnodes.items()}
    subarcs = {x:[(a[0][0],a[1][0]) for a in arcs if a[0][1]==x and a[1][1]==x] for x in nets}								# Subset of arcs that belong to each network
    
    # Obj. Costs				# Cost of fixing node i.j of network k
    M_p = {(i,k,l):G.nodes(data=True)[(i,k)]['data']['Mp'] for k in nets for l in commodities for i in subnodes[k] if l in com[k]} 			# Penalty for excess
    M_m = {(i,k,l):G.nodes(data=True)[(i,k)]['data']['Mm'] for k in nets for l in commodities for i in subnodes[k] if l in com[k]} 			# Penalty for shortage
    c = {(i,j,k,l):G[(i,k)][(j,k)]['data']['c'] for k in nets for l in commodities for (i,j) in subarcs[k] if l in com[k]} 		# Flow cost
    
    # Coefs & Right-hand Sid
    # Demands at nodes (fictional)
    b = {(i,k,l):G.nodes(data=True)[(i,k)]['data']['b'] for k in nets for l in commodities for i in subnodes[k] if l in com[k]} 
    # arc capacities
    u = {(i,j,k,l):G[(i,k)][(j,k)]['data']['u'] for k in nets for l in commodities for (i,j) in subarcs[k] if l in com[k]} 					# Capacity of arc i,j in network k    
    
    # gamma: is it necessary that node i (net k) k is functional for node j (net ka) to work?
    gamma = {(i,j,k,ka):0 for k in nets for ka in nets for i in nodes for j in nodes if i in subnodes[k] if j in subnodes[ka] if i!=j if k!=ka}

    for a in list(arcs) :
        if a[0][1] != a[1][1]:
            gamma[a[0][0],a[1][0],a[0][1],a[1][1]] = 1  
        
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
    			outflow = quicksum(x[i,j,k,l] for j in subnodes[k] if (i,j) in subarcs[k])
    			inflow = quicksum(x[j,i,k,l] for j in subnodes[k] if (j,i) in subarcs[k])
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
#    lname = "Model.lp" 
#    m.write(lname)         
#    sname = "Solution.txt"
#    fileID = open(sname, 'w')
#    for vv in m.getVars():
#        fileID.write('%s %g\n' % (vv.varName, vv.x))    
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
def plot_Evaluation_Results(noConfig, dmRatioSum, aveConn, allPairsConn,folder):

    if not os.path.exists(folder):
        os.makedirs(folder) 
    plt.figure(figsize=(8, 6), dpi=100)     
    H, bins = np.histogram(dmRatioSum)
    width = 0.5 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, H, align='center', width=width)
    plt.title("Histogram of % of met demand for all networks")
    plt.xlabel('% met demand')
    plt.ylabel('Frequency')
    plt.savefig(folder+'HistdmRatio.png')    
    plt.clf()
    plt.bar(range(noConfig),np.mean(dmRatioSum, axis=(1)))
    plt.title("Average of % of met demand for each configuration")
    plt.xlabel('Configuration Number')
    plt.ylabel('% met demand')
    plt.savefig(folder+'AveragedmRatio.png')
    plt.clf()  
 
    H, bins = np.histogram(aveConn)
    width = 0.1 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    f1 = H
    plt.bar(center,f1,color='b',align='center',label='Network 1',width=width)
    plt.title("Histogram of average connectivity for all networks")
    plt.xlabel('Average connectivity')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.savefig(folder+'HistAveConn.png') 
    plt.clf()

    plt.figure(figsize=(10, 6), dpi=100)   
    plt.bar(np.array(range(noConfig))+.1,np.mean(aveConn,axis=(1)),width=0.25,
            color='b',align='center',label='Network 1')
    plt.title("Average of average connectivity of networks for each configuration")
    plt.xlabel('Configuration Number')
    plt.ylabel('Average of average connectivity')
    plt.xticks(range(0,noConfig,5))
    plt.legend(loc='best')
    plt.savefig(folder+'AverageAveConn.png') 
    plt.clf()

    plt.bar(np.array(range(noConfig))+0.1,np.mean(allPairsConn,axis=(1)),
            width=0.25,color='b',align='center',label='Network 1')
    plt.title("Average of connectedness of networks for each configuration")
    plt.xlabel('Configuration Number')
    plt.ylabel('Average of connectedness')
    plt.xticks(range(0,noConfig,5))
    plt.legend(loc='best')
    plt.savefig(folder+'AverageConnectedness.png')
    plt.clf()

    
    
# Input values
noSamples = 30
rootfolder = "C:\\Users\ht20\Documents\\Files\\Generated_Network_Dataset_v3.1\\" # Root folder where the database is
rootfolder += 'GridNetworks\\' # choose relevant dataset folder #options: 'RandomNetworks\\'|'ScaleFreeNetworks\\'|'GridNetworks\\'
NetworkTypeInitial = 'GN' #Options: RN|SFN|GN

# Read configuration data
fileNameList = rootfolder + 'List_of_Configurations.txt' 
configList = np.loadtxt(fileNameList,skiprows=1)

# 3D arrays to store evaluation results
dmRatioSum = np.zeros((len(configList),noSamples))
aveConn = np.zeros((len(configList),noSamples))
allPairsConn = np.zeros((len(configList),noSamples)) 
noInterconnections = np.zeros(len(configList))
 
for i in configList: 
    # Reading configuration data from file 
    cnfg = int(i[0])
    noLayers = int(i[1])
    noNodes = int(i[2])
    arcProb = i[3] # exp for scale free net; gridsizex for Grid net
    intConProb = i[4]
    damProb = i[5]
    resCap = i[6]
#    foldername = NetworkTypeInitial+'Config_%d\\' % (cnfg)
    for s in range(noSamples):
        G,pos,noLayers,dam_nodes,dam_arcs = load_array_format_extended(rootfolder,NetworkTypeInitial,cnfg,s,cost_scale=1.0)
    
        # Computoing the connectedness and average connectivity for each
        # network
        aveConnect = 0.0
        isConnected = 1.0
        for k in range(noLayers):
            node_list = [x for x in G.nodes() if x[1]==k+1]
            H = G.subgraph(node_list)
            aveConnect += nx.average_node_connectivity(H)
            isConnected *= int(nx.is_connected(H.to_undirected()))
        aveConn[cnfg,s] = aveConnect/noLayers
        allPairsConn[cnfg,s] = isConnected

        # Computoing the the percentage of demand which is met
        # for all layers            
        dmSum = initialPerformanceUndamagedNetwork(G,noLayers)
        b = [G.nodes()[x]['data']['b'] for x in list(G.nodes())] 
        alldemand = sum([y for y in b if y<0])
        dmRatioSum[cnfg,s] = 1.0+dmSum/alldemand
            
        print('Config %d | Sample_%d' % (cnfg,s))
    #Number of Interconnections
    noInterconnections[cnfg] = round(noNodes*noNodes*intConProb)
#            
#            
            
# Plot QUality  Control results  
plt.close('all')          
qcFolder = rootfolder+'Evaluations/'
plot_Evaluation_Results(len(configList),dmRatioSum,aveConn, allPairsConn,qcFolder)                      
sns.set()     
ax = sns.distplot(noInterconnections,rug=True,kde=False)
ax.set(xlabel='Number of Interconnections', ylabel='Probability (%)')
#plt.savefig('noInterconnections_'+NetworkTypeInitial+'.png',dpi=600,bbox_inches="tight")

#''' Plot one network '''
#plot_network(BASE_DIR=rootfolder,topo=NetworkTypeInitial,config=0,sample=0)  