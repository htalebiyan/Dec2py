'''
INDP implementation
Camilo Gomez, Andres Gonzalez - 05/09/17 (Last updated: 22/09/17)

Model and notations from:
	The Interdependent Network Design Problem for Optimal Infrastructure System Restoration
	Gonzalez, Duenas-Osorio, Sanchez-Silva, Medaglia
	Computer-Aided Civil and Infrastructure Engineering 31 (2016) 334-350

Data created for toy example (see picture in folder).
'''

import numpy as np
from gurobipy import *
import networkx as nx
import os

def output_Cent(sce,T,nets):
    folder = 'CentResults\\'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    desc = ['t','Flow Cost', 'Arc Recovery Cost', 'Node Recovery Cost',\
            'Recovery Cost', 'Total Cost'\
            ,'Unbalanced Cost', 'Total-Unbalanced','Site Prep Cost',\
            'NoRepNodes','NoRepArcs','defiDem']
            
    for k in nets:
        desc.append('NoRepNodes %s' % k)
    for k in nets:
        desc.append('NoRepArcs %s' % k)
        
    timeVec = range(T)
    sepRep = np.hstack((wkNo, ykNo))
    Data = np.vstack((timeVec,soltFOa,soltFOb1,soltFOb2,soltFOb,soltFOT,\
                      soltFO2,soltFO,soltFOc,wNo,yNo,dmSum,sepRep.T))
    
    f = open(folder+'CenINDP_Sce_%d.txt'%sce,"w")
    
    for item in desc:
      f.write("%s\t" % item)
    f.write("\n")
    for row in Data.T:
        for item in row:
            f.write("%s\t" % item)
        f.write("\n")
    f.close()

#####################################Visualize the network####################
noScenarios = 1
auxfiles = 1
for sce in range(noScenarios):
    coo = {
    '1':(0,0,0),
    '2':(2,0,0),
    '3':(1,1,0),
    '4':(0,3,0),
    '5':(2,3,0),
    '6':(0,0,1),
    '7':(2,0,1),
    '8':(0,3,1),
    '9':(2,3,1)} 		# location of nodes for both networks
    
    
    
    '''	INITIALIZATIONS
    '''
    
    # Sets
    nodes = [str(i) for i in range(1,10)] # All network nodes
    dam_nodes = []  		# Nodes that will be damaged
    for i in range(1,10):
        if np.random.rand() < 0.5:
            dam_nodes.append(str(i))
    			
    arcs = [ 									# Arcs in the drawing
    ('1','2'),('1','3'),('1','4'),
    ('2','1'),('2','3'),('2','5'),
    ('3','1'),('3','2'),('3','4'),('3','5'),
    ('4','1'),('4','3'),('4','5'),
    ('5','2'),('5','3'),('5','4'),
    ('6','7'),('6','8'),
    ('7','6'),('7','9'),
    ('8','6'),('8','9'),
    ('9','7'),('9','8')]
    dam_arcs = []
    for i in range(1,10):
        for j in range(i+1,10):
            if ((str(i), str(j)) in arcs) and np.random.rand() < 0.5:
                dam_arcs.append((str(i),str(j)))
                dam_arcs.append((str(j),str(i)))
    
    zones = {1:['1','4','6','8'], 2:['2','5','7','9'], 3:['3']}
    nets = ['P','G']
    commodities = {'power':'power', 'gas':'gas'}
    com = {'P':['power'], 'G':['gas']}
    resources = ['car', 'tape']
    
    
    # Subsets
    subnodes = {'P':[str(i) for i in range(1,6)], 'G':[str(i) for i in range(6,10)]} 	# Subset of nodes that belong to each network
    subarcs = {k:[] for k in nets} 														# Subset of arcs that belong to each network
    for k in nets:
    	for (i,j) in arcs:
    		if i in subnodes[k] and j in subnodes[k]:
    			subarcs[k].append((i,j))
       
#    Pn = [i for i in range(5)]
#    Gn = [i for i in range(4)]    
#    Ppos = [coo[str(i)] for i in range(1,6)]
#    Gpos = [coo[str(i)] for i in range(6,10)]
#    sa = {k:[] for k in nets}
#    for k in nets:
#        for (i,j) in arcs:
#            if i in subnodes[k] and j in subnodes[k]:
#                if k=='G':
#                    sa[k].append((int(i)-6,int(j)-6))
#                else:
#                    sa[k].append((int(i)-1,int(j)-1))
#       
#    selPairs = [(0,0),(1,1),(3,2),(4,3)]
#    Network_Data_Generator.plot_interdependecnt_netwotks_3D(Pn, sa['P'],
#                                        Ppos,Gn,sa['G'],Gpos,selPairs)   
    # Obj. Costs
    g = {1:1, 2:1, 3:1} 																								# Cost of making an intervention in each of the zones
    f = {(i,j,k):1 for (i,j) in arcs for k in nets if (i,j) in subarcs[k]} 												# Cost of fixing arc i.j of network k
    q = {(i,k):1 for i in nodes for k in nets if i in subnodes[k]} 														# Cost of fixing node i.j of network k
    M_p = {(i,k,l):9 for i in nodes for k in nets for l in commodities if i in subnodes[k] and l in com[k]} 			# Penalty for excess
    M_m = {(i,k,l):9 for i in nodes for k in nets for l in commodities if i in subnodes[k] and l in com[k]} 			# Penalty for shortage
    c = {(i,j,k,l):1 for (i,j) in arcs for k in nets for l in commodities if (i,j) in subarcs[k] and l in com[k]} 		# Flow cost
    
    # Coefs & Right-hand Side
    b = {} 							# Demands at nodes (fictional)
    for k in nets:
    	for i in subnodes[k]:
    		for l in com[k]:
    			if i in ['1','3','5','7','9']:
    				b[i,k,l] = 2
    			elif i in ['2','4']:
    				b[i,k,l] = -3
    			elif i in ['6','8']:
    				b[i,k,l] = -2
    
    u = {(i,j,k):10 for (i,j) in arcs for k in nets if (i,j) in subarcs[k]} 					# Capacity of arc i,j in network k
    h = {(i,j,k,r):1 for (i,j) in arcs for k in nets for r in resources if (i,j) in subarcs[k]} 	# Resource needed to fix arc
    p = {(i,k,r):1 for i in nodes for k in nets for r in resources if i in subnodes[k]} 			# Resource r needed to fix node
    v = {r:2 for r in resources} 												# Resources available
    
    alpha = {(i,k,s):0 for i in nodes for k in nets for s in zones if i in subnodes[k]} 			# Is node i of network k in zone s
    for (i,k,s) in alpha:
    	if i in zones[s]:
    		alpha[i,k,s] = 1
    
    beta = {(i,j,k,s):0 for (i,j) in arcs for k in nets for s in zones if (i,j) in subarcs[k]} 		# Is arc i,j of network k in zone s
    for (i,j,k,s) in beta:
    	if i in zones[s] and j in zones[s]:
    		beta[i,j,k,s] = 1
    	else:
    		beta[i,j,k,3] = 1
    
    # gamma: is it necessary that node i (net k) k is functional for node j (net ka) to work?
    gamma = {(i,j,k,ka):0 for k in nets for ka in nets for i in nodes for j in nodes if i in subnodes[k] if j in subnodes[ka] if i!=j if k!=ka}
    gamma['1','6','P','G']=1
    gamma['4','8','P','G']=1
    gamma['2','7','P','G']=1
    gamma['5','9','P','G']=1
    
    T = 5
    
    soltFOa = np.zeros(T)
    soltFOb1 = np.zeros(T)
    soltFOb2 = np.zeros(T)
    soltFOb = np.zeros(T)
    soltFOT = np.zeros(T)
    soltFO2 = np.zeros(T)
    soltFO = np.zeros(T)
    soltFOc = np.zeros(T) # 'Site Prep Cost'
    dmSum = np.zeros(T)
    wNo = np.zeros(T)
    yNo = np.zeros(T)
    k = len(nets)
    wkNo = np.zeros((T,k)) # number of repaired nodes fro each net
    ykNo = np.zeros((T,k)) # number of repaired arcs fro each net
    
    #Write initial statues to file
    folder = 'Output\\Scenario_%d\\' % sce
    if not os.path.exists(folder):
        os.makedirs(folder)
    name = folder+"CentrModel_initial.txt"
    fileID = open(name, 'w')
    
    for ka in nets:
        for j in subnodes[ka]:
            state = 1
            if (j in dam_nodes):
                state = 0
            fileID.write('w_(%s, %s)\t%d\n' % (j, ka, state))
        for j in subarcs[ka]:
            state = 1
            if (j in dam_arcs):
                state = 0
            fileID.write('y_(%s, %s, %s)\t%d\n' % (j[0], j[1], ka, state))                       
    fileID.close()
    
    for t in range(T):
        
        m = Model("INDP1")
        
        '''
        	VARIABLES
        '''
        w = {} 		# whether node i (of network k) IS FUNCTIONAL (if damaged, investment necessary to make it functional)
        x = {} 		# flow through arc (i,j) of network k, commodity l
        y = {} 		# whether arc (i,j) of network k IS FUNCTIONAL (if damaged, investment necessary to make it functional)
        z = {} 		# whether zone s is intervened
        d_p = {} 	# flow excess at node i, network k, commodity l
        d_m = {} 	# flow defect at node i, network k, commodity l
        
        for s in zones:
        	z[s] = m.addVar(vtype=GRB.BINARY, name="z_"+str(s), obj=g[s])
        for k in nets:
        	for i in subnodes[k]:
        		if i in dam_nodes:
        			w[i,k] = m.addVar(vtype=GRB.BINARY, name="w_"+str((i,k)), obj=q[i,k])
        		else:
        			w[i,k] = m.addVar(vtype=GRB.BINARY, name="w_"+str((i,k)), obj=0)
        		for l in com[k]:
        			d_p[i,k,l] = m.addVar(vtype=GRB.CONTINUOUS, name="Mp_"+str((i,k,l)), obj=M_p[i,k,l])
        			d_m[i,k,l] = m.addVar(vtype=GRB.CONTINUOUS, name="Mm_"+str((i,k,l)), obj=M_m[i,k,l])
        	for (i,j) in subarcs[k]:
        		if (i,j) in dam_arcs:
        			y[i,j,k] = m.addVar(vtype=GRB.BINARY, name="y_"+str((i,j,k)), obj=f[i,j,k])
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
        		arcflow = quicksum(x[i,j,k,l] for l in com[k])
        		m.addConstr(arcflow <= u[i,j,k] * w[i,k], name="avaTail_"+str((i,j,k)))
        		m.addConstr(arcflow <= u[i,j,k] * w[j,k], name="avaHead_"+str((i,j,k)))
        		if (i,j) in dam_arcs:
        			m.addConstr(arcflow <= u[i,j,k] * y[i,j,k], name="avaArc_"+str((i,j,k)))
        
        # Resource availability
        for r in resources:
        	arc_ = quicksum(h[i,j,k,r]*y[i,j,k] for (i,j) in arcs for k in nets if (i,j) in subarcs[k] if (i,j) in dam_arcs)
        	nod_ = quicksum(p[i,k,r]*w[i,k] for i in nodes for k in nets if i in subnodes[k] if i in dam_nodes)
        	m.addConstr(arc_ + nod_ <= v[r], name="resource_"+str(r))
        
        # Interdependence
        for ka in nets:
        	for j in subnodes[ka]:
        		if sum(gamma[i,j,k,ka] for k in nets if k!=ka for i in nodes if i in subnodes[k]):
        			dep = quicksum(gamma[i,j,k,ka]*w[i,k] for k in nets if k!=ka for i in nodes if i in subnodes[k])
        			m.addConstr(w[j,ka] <= dep, name="interdep_"+str((ka,j)))
        
        # Zone activation
        for s in zones:
        	for k in nets:
        		for i in subnodes[k]:
        			if i in dam_nodes:
        				m.addConstr( alpha[i,k,s]*w[i,k] <= z[s], name="zone_node_"+str(s))
        		for (i,j) in subarcs[k]:
        			if (i,j) in dam_arcs:
        				m.addConstr( beta[i,j,k,s]*y[i,j,k] <= z[s], name="zone_arc_"+str(s))
        
        
        '''
        	OBJ & OUTPUT
        '''
        m.update()
        m.setParam('OutputFlag',0)
        m.optimize()
        if auxfiles:    
            lname = "outputFiles\\CentrModel_t%d.lp" % t
            m.write(lname)
       
#        print("\n")
#        print("\t\tNET.\tComponent\tSTATUS\tActive\tMass")
#        for s in zones:
#            if z[s].x:
#                print("ZONE "+str(s)+": _______________________________________________________")
#                for k in nets:
#                    for l in com[k]:
#                        for i in subnodes[k]:
#                            if alpha[i,k,s]:
#                                state = '  ok'
#                                active = ' '
#                                if i in dam_nodes:
#                                    state = 'broken'
#                                if w[i,k].x==1:
#                                    active = '  !'
#                                print("\t\t"+str(com[k][0])+"\t Node "+str(i)+", \t"+state+"\t"+active+"\t  "+str(M_p[i,k,l]-M_m[i,k,l]))
#                print("\t\t_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
#                for k in nets:
#                    for l in com[k]:
#                        for (i,j) in subarcs[k]:
#                            if beta[i,j,k,s]:
#                                state = '  ok'
#                                active = '  !'
#                                if (i,j) in dam_arcs:
#                                    state = 'broken'
#                                    if not y[i,j,k].x==1:
#                                        active = ' '
#                                if x[i,j,k,l].x:
#                                    print("\t\t"+str(com[k][0])+"\t Arc  "+str(i)+','+str(j)+", \t"+state+"\t"+active+"\t  "+str(round(x[i,j,k,l].x,2)))
#                                else:
#                                    print("\t\t"+str(com[k][0])+"\t Arc  "+str(i)+','+str(j)+", \t"+state+"\t"+active+"\t  ")
#        print("\nI spent $"+str(m.objVal)+"!\n")
        
        soltFOa[t] = sum([x[i,j,k,l].x*c[i,j,k,l] for (i,j) in arcs for k in nets for l in commodities if (i,j) in subarcs[k] and l in com[k]])		# Flow cost
        soltFOb1[t] = sum([y[i,j,k].x*f[i,j,k] for (i,j) in arcs for k in nets if (i,j) in subarcs[k] and (i,j) in dam_arcs]) 	
        soltFOb2[t] = sum([w[i,k].x*q[i,k] for i in nodes for k in nets if i in subnodes[k] and i in dam_nodes]) 
        soltFOb[t] = soltFOb1[t] + soltFOb2[t]
        soltFOT[t] = (m.getObjective()).getValue()
        soltFO2[t] = sum([d_p[i,k,l].x*M_p[i,k,l]+d_m[i,k,l].x*M_m[i,k,l] for i in nodes for k in nets for l in commodities if i in subnodes[k] and l in com[k]]) 			# Penalty for excess
        soltFO[t] = soltFOT[t] - soltFO2[t]
        soltFOc[t] = soltFOT[t] - soltFOb[t] - soltFO2[t] - soltFOa[t]
        dmSum[t]= sum([i.x for i in d_m.values()])
        wNo[t] = sum([w[i,k].x for i in nodes for k in nets if i in subnodes[k] and i in dam_nodes])
        yNo[t] = sum([y[i,j,k].x for (i,j) in arcs for k in nets if (i,j) in subarcs[k] and (i,j) in dam_arcs])      
        for k in nets:  
            kn = nets.index(k)
            wkNo[t,kn] = sum([w[i,k].x for i in nodes if i in subnodes[k] and i in dam_nodes])
            ykNo[t,kn] = sum([y[i,j,k].x for (i,j) in arcs if (i,j) in subarcs[k] and (i,j) in dam_arcs])
         
        
        # Write solution to file
        folder = 'Output\\Scenario_%d\\' % sce
        sname = folder+"CentrModel_t%d_model_sol.txt" % (t)
        fileID = open(sname, 'w')
        for vv in m.getVars():
            fileID.write('%s %g\n' % (vv.varName, vv.x))
        fileID.close() 
        for k in nets:
            for i in subnodes[k]:
                if i in dam_nodes:
                    if w[i,k].x==1:
                        dam_nodes.remove(i)
            for (i,j) in subarcs[k]:
                if (i,j) in dam_arcs:
                    if y[i,j,k].x==1:
                        dam_arcs.remove((i,j))
                        
    ## Write costs to file      
    output_Cent(sce,T,nets)
    print('Scenario %d' % sce)
