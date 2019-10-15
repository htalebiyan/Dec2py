'''
This file contains the functions to generate and store a database of 
interdependenct random networks
Hesam Talebiyan - Last updated: 03/04/2018 
'''

import Network_Data_Generator
import numpy as np
import os

# Input values
noSamples = 30 # Number of sample sets of network
noConfiguration = 100 # Number of configurations
noZones = 4 # noZones by noZones tile of zones
paramError = 0.1
rootfolder = 'C:\\Users\\ht20\Documents\\Files\\Generated_Network_Dataset_v3\\' # Root folder where the database is
rootfolder += 'ScaleFreeNetworks\\' # choose relevant dataset folder
if not os.path.exists(rootfolder):
    os.makedirs(rootfolder)
    
# The text file which stores the information of each configurations 
fileNameList = rootfolder+'List_of_Configurations.txt'
# Saving network data in the text files
fList = open(fileNameList,"a+")
fList.write('Config Number\t No. Layers\t No. Nodes\t Topology Parameter\t Interconnection Prob\t Damage Prob\t Resource Cap\n') 
fList.close()

for cnfg in range(0,noConfiguration):
    # Number of nodes of the random network (same for both networks)
    noLayers = np.random.randint(low=2, high=5) 
    # Number of nodes of the random network (same for both networks)
    noNodes = np.random.randint(low=10, high=50) 
    noNodesDict={x:int(round(noNodes*(1+np.random.normal(0, paramError)))) for x in range(1,noLayers+1)}
    # Exponent of the powerlaw of node degee distribution 
    # whose bounds correspond to Ultra-Small World regime.
    expLB = 2.001
    expUB = 2.999
    exp = np.random.uniform(low=expLB, high=expUB)
    expDict={}
    for x in range(1,noLayers+1):
        expPert = exp*(1+np.random.normal(0, paramError))
        while expPert<expLB or expPert>expUB:
            expPert = exp*(1+np.random.normal(0, paramError))
        expDict[x] = expPert
    # Existence Probability of each interconnection (among all possible pairs) 
    # in the interdependent random networks 
    intConProb = np.random.uniform(low=0.001, high=0.05/2)
    intConProbDict={}
    for k in range(1,noLayers+1): 
        for kt in range(1,noLayers+1): 
            if k!=kt:
                intConProbDict[(kt,k)]=intConProb*(1+np.random.normal(0, paramError))
    # Probability of damage ofeach node in the random networks 
    # (same for both networks)
    # Bounds are chosen based on INDP data for Shelby county associated with 
    # M6 (0.05) - M9 (0.5) scenarios
    damProb = np.random.uniform(low=0.05, high=0.5/2) 
    damProbDict={x:damProb*(1+np.random.normal(0, paramError)) for x in range(1,noLayers+1)}
    # Restoration Resource Cap for each network 
    # based on the sum of mean number of damaged nodes and mean number of 
    # damaged arcs
    meanNoDamagedNodes =noNodes*damProb
    kmin = 1
    kmax = kmin*(noNodes**(1/(exp-1)))
    meanNoDamagedArcs = meanNoDamagedNodes*(kmax/2)
    resCap = np.random.randint(low=noLayers, high=max(2*noLayers,meanNoDamagedNodes+meanNoDamagedArcs))
    
    # Saving network data in the text files
    fList = open(fileNameList,"a+")
    fList.write('%d\t%d\t%d\t%1.2f\t%1.3f\t%1.2f\t%d\n' % (cnfg,noLayers,noNodes,exp,intConProb,damProb,resCap)) 
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
    for s in range(noSamples): 
        # Making target folder to save network data          
        folder = rootfolder+'SFNConfig_%d/Sample_%d' % (cnfg,s)
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        for k in range(1,noLayers+1):                
            # Generating random networks
            nodes[k],arcs[k],pos[k]=Network_Data_Generator.Scale_Free_network(expDict[k],noNodesDict[k],k)       
    
            '''All bounds are are chosen based on INDP data for Shelby county'''
            # Supply/Demand values for each node in the networks
            b[k] = np.random.randint(low=0, high=700, size=noNodesDict[k])
            b[k] = (b[k] - np.mean(b[k])).astype('int') # Making some values negative corresponds to demand  
            b[k][0] = b[k][0]-sum(b[k]) # balancing the demand and supply
            # Over- and unde- supply penalties for each node in the networks
            Mp[k] = np.random.randint(low=5e6, high=10e6, size=noNodesDict[k])
            Mm[k] = np.random.randint(low=5e5, high=10e5, size=noNodesDict[k])
            # reconstruction cost for each node in the networks
            q[k] = np.random.randint(low=5e3, high=15e3, size=noNodesDict[k])
            # Damaged nodes in each node in the networks
            damNodes[k]= Network_Data_Generator.random_damage_Data(noNodesDict[k],damProbDict[k])    
            # Damaged arcs in each node in the networks
            # All arcs attached to a damaged node are attached
            damArcs[k]= [(i,j) for (i,j) in arcs[k] if i in damNodes[k] or j in damNodes[k]]
            # Capacity of each arc in the networks           
            u[k] = np.random.randint(low=500, high=2000, size=len(arcs[k]))
            # Flow cost of each arc in the networks      
            c[k] = np.random.randint(low=50, high=500, size=len(arcs[k]))
            # reconstruction cost of each arc in the networks      
            fa[k] = np.random.randint(low=2500, high=7e4, size=len(arcs[k]))

            # The text files which stores network data  
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
        
        noSelPairsDict={}
        for k in range(1,noLayers+1):
            for kt in range(1,noLayers+1):
                if kt!=k:
                    selPairs = Network_Data_Generator.generate_interconnctions(nodes[kt],nodes[k],intConProbDict[(kt,k)])
                    noSelPairsDict[(kt,k)] = len(selPairs)
                    fileName = folder + '/Interdependent_Arcs_%d_%d.txt' % (kt,k)
                    f = open(fileName,"w")
                    for j in selPairs:
                        f.write('%d\t%d\tN%d\tN%d\n' % (j[0],j[1],kt,k))   
                    f.close()                    
        # The text file which stores the general information of each configurations 
        fileName = folder + '/Overview.txt'
        f = open(fileName,"w")
        f.write('Number of networks = %d\n' % (noLayers))
        text = 'Number of nodes: '
        for k in range(1,noLayers+1):
            text += 'N%d = %d,' %(k,noNodesDict[k])
        f.write(text+'\n')
        text = 'Exponent of powerlaw distribution of node degree: '
        for k in range(1,noLayers+1):
            text += 'N%d = %1.4f,' %(k,expDict[k])
        f.write(text+'\n')
        text = 'Number of arcs: '
        for k in range(1,noLayers+1):
            text += 'N%d = %d,' %(k,arcs[k].shape[0])
        f.write(text+'\n')
        text = 'Existence probability of interdependent arcs: '
        for key,value in intConProbDict.items():
            text += '%s = %1.4f,' %(key,value)
        f.write(text+'\n')
        text = 'Number of interdependent arcs: '
        for key,value in noSelPairsDict.items():
            text += '%s = %d,' %(key,value)
        f.write(text+'\n') 
        text = 'Damage probability of nodes: '
        for k in range(1,noLayers+1):
            text += 'N%d = %1.4f,' %(k,damProbDict[k])
        f.write(text+'\n')
        text = 'Number of damaged nodes: '
        for k in range(1,noLayers+1):
            text += 'N%d = %d,' %(k,damNodes[k].shape[0])
        f.write(text+'\n')
        text = 'Number of damaged arcs: '
        for k in range(1,noLayers+1):
            text += 'N%d = %d,' %(k,len(damArcs[k]))
        f.write(text+'\n')
        f.write('Available resources for restoration in each time step = %d\n' % (resCap))
        f.write('Interconnections are chosen randomly\n')
        f.write('Damaged nodes are chosen randomly\n')
        f.write('Damaged arcs are those corresponding to a damaged node\n')
        f.close()
        # cost of preparation of geographical zones
        g = np.random.randint(low=2000, high=18e4, size=noZones*noZones)
        fileName = folder + '/Zone_prep.txt'
        f = open(fileName,"w")
        for j in g:
            f.write('%d\t' % j)
            f.write('\n')   
        f.close() 
    print('Configuration %d' % (cnfg))
   
   