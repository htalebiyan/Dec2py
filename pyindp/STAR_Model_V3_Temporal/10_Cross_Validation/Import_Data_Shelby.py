import numpy as np
import re
import os
import Network_Data_Converter

def importData(noScenarios,noSets,T,resCap):  
    nets = ['W','G','P']
    com = {'P':['power'], 'G':['gas'], 'W':['water']}
    
    '''Root folder where the database is'''
    rootfolderDamage = 'C:\\Users\\ht20\\Documents\\Files\\DataBase_tdINDP\\WuData\\Damage Scenarios\\'
    rootfolderResult = 'C:\\Users\\ht20\\Documents\\GitHub\\Bayesian-Hierarchical-Models\\Shelby_Example\\STAR_Model_Diff_Subproblem\\STAR_Model_V3\\outputFiles\\'
    rootfolder = 'C:\\Users\\ht20\\Documents\\Files\\DataBase_tdINDP\\MyData\\INDPScenariosNewFormat\\Base_Data\\' 
    
    '''     Input values     '''
    initialPerfTC = np.loadtxt('C:\\Users\\ht20\\Documents\\Files\\ShelbyCountyResults_V%d\\InitialPerfTC.txt' % resCap)
    
    noNet = len(nets)
    Nodes = {}
    Arcs = {}
    IntArcs = {}
    DamNodes = {}
    DamArcs = {}
    
    database='ShelbyCounty'
    # Reading network data from file 
    for k in range(1,noNet+1):
        Nodes[nets[k-1]] = np.loadtxt(rootfolder+'N%d_Nodes.txt' % k)
        Arcs[nets[k-1]] = np.loadtxt(rootfolder+'N%d_Arcs.txt' % k).astype('int')
        for ka in range(1,noNet+1):
            if os.path.isfile(rootfolder+'Interdependent_Arcs_%d_%d.txt' % (k,ka)):
                IntArcs['%s_%s' %(nets[k-1],nets[ka-1])] = np.loadtxt(rootfolder+
                        'Interdependent_Arcs_%d_%d.txt' % (k,ka), 
                        usecols = (0,1)).astype('int')
    g = np.loadtxt(rootfolder+'Zone_prep.txt').astype('int')
     
    # Convert damage data to the format of this function
    nNode,nArcs,_,convNodes,convArcs,convintdpnPairs,_,_,_,_,_,_,_,_,_, = Network_Data_Converter.conv_Data(Nodes,Arcs,IntArcs,g,nets,com,database)   
    
    # Sets
    nodes = convNodes
    arcs = convArcs                       
        
    # Subsets
    subnodes = {k:[] for k in nets} # Subset of nodes that belong to each network
    startNodeNum = 0
    for k in range(len(nets)):
        subnodes[nets[k]] = [str(i) for i in range(startNodeNum,startNodeNum+nNode[k])] 	
        startNodeNum += nNode[k]
    subarcs = {k:[] for k in nets} # Subset of arcs that belong to each network
    for k in nets:
    	for (i,j) in arcs:
    		if i in subnodes[k] and j in subnodes[k]:
    			subarcs[k].append((i,j))
    
    ''' Read data from file '''
    samples = {}
    for k in nets:
        for n in subnodes[k]:
            name = 'w_%s_%s' % (n,k)
            samples[name] = np.ones((T+1,1))
    damagedSce = []
    #    for a in subarcs[k]:
    #        name = 'y_%s_%s_%s' % (a[0],a[1],k)
    #        samples[name] = np.ones((T+1,noScenarios*noSets))
            
    for m in range(1,noSets+1):
        foldername1 = 'Set%d\\' % (m)
        for i in range(noScenarios):
            foldername2 = 'Sce%d\\' % (i)
            undamagedSceFlage = False
            indexInitialPerfTC = (m-1)*96+i
            subfolderDamage = rootfolderDamage+foldername1+foldername2
            
            intDmSum = initialPerfTC[indexInitialPerfTC,2]
            intTC = initialPerfTC[indexInitialPerfTC,3]
            
            if not os.path.exists(subfolderDamage):
                undamagedSceFlage = True
                
            elif intDmSum==0.0 and intTC==775869:
                undamagedSceFlage = True
            
            if not undamagedSceFlage:
                
                for k in range(1,noNet+1):
                    DamNodes[nets[k-1]] = np.loadtxt(subfolderDamage+
                                'N%d_Damaged_Nodes.txt' % k).astype('int')
                    DamArcs[nets[k-1]] = np.loadtxt(subfolderDamage+
                                'N%d_Damaged_Arcs.txt' % k).astype('int')
                dam_nodes,dam_arcs=Network_Data_Converter.conv_Damage_Data(nNode,DamNodes,DamArcs,nets)
                
                if dam_nodes:      
                    data = {t:[] for t in range(T)}
                    subfolderResult = rootfolderResult+'ShelbyCounty_RC%d_Set%d_Sce%d\\' % (resCap,m,i)
                    damagedSce.append('Set%d_Sce%d\\' % (m,i))
                    for k in nets:
                        for t in range(T):
                            filename = 'DecentrModel_%s_t%d_model_sol.txt' % (k,t)
                            with open(subfolderResult+filename) as f:
                                content = f.readlines()
                            data[t].append([" ".join(re.findall("[a-zA-Z0-9]+", x.strip())).split() for x in content])

                    for k in nets:
                        for n in subnodes[k]:
                            name = 'w_%s_%s' % (n,k)
                            samples[name] = np.append(samples[name],np.ones((T+1,1)),1)
                            if str(n) in dam_nodes:
                                samples[name][0,-1] = 0
                                
                    for t in range(T):
                        data[t] = [item for sublist in data[t] for item in sublist]           
                        for d in range(len(data[t])): 
                            var = data[t][d]
                            if var[0] == 'w':
                                name = 'w_%s_%s' % (var[1],var[2])
                                if (abs(float(var[3])-1.0) <= 1e-2):
                                    samples[name][t+1,-1] = 1
                                else: 
                                    samples[name][t+1,-1] = 0  
                                    
#                                for key,value in gammaFiltered.items(): 
#                                    if key[1] == var[1]:
#                                        name2 = 'w_%s_%s' % (key[0],key[2])
#                                        if samples[name2][0,-1]== 0:
#                                            samples[name][0,-1] = 0
                                
    
    for k in nets:
        for n in subnodes[k]:
            name = 'w_%s_%s' % (n,k)
            samples[name] = np.delete(samples[name], 0, 1)
            
    nNodes = len(nodes)
    A = np.zeros((nNodes,nNodes))
    for a in arcs:
        i = int(a[0])    
        j = int(a[1])
        A[i,j] = 1
            
    # Check if there is any functional sate befor an non-functional one
    for key, value in samples.items():
        for t in range(T-1):
            for i in range(value.shape[1]):
                if value[t,i]>value[t+1,i]:
                    print('%s, %d %d' % (key,t,i))
        
    print('Data Imported')
    return samples     