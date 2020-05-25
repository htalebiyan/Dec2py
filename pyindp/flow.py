from infrastructure import *
from indputils import *
from gurobipy import *
import string
#import platform
import networkx as nx
import matplotlib.pyplot as plt
#import copy
import random
import time
import sys
def flow_problem(N,v_r,T=1,layers=[1,3],controlled_layers=[1,3],decision_vars={},functionality={},forced_actions=False, print_cmd=True, time_limit=None):
    """INDP optimization problem. Also solves td-INDP if T > 1.
    :param N: An InfrastructureNetwork instance (created in infrastructure.py)
    :param v_r: Vector of number of resources given to each layer in each timestep. If the size of the vector is 1, it shows the total number of resources for all layers.
    :param T: Number of timesteps to optimize over.
    :param layers: Layer IDs of N included in the optimization. (Default is water (1) and power (3)).
    :param controlled_layers: Layer IDs that can be recovered in this optimization. Used for decentralized optimization.
    :param functionality: Dictionary of nodes to functionality values for non-controlled nodes. Used for decentralized optimization.
    :returns: A list of the form [actions,costs] for a successful optimization.
    """
    #print "T=",T
    #if functionality:
    #    for t in functionality:
    #        for u in functionality[t]:
    #            if functionality[t][u] == 0.0:
    #                print "At Time",str(t),", node",u,"is broken."
    start_time = time.time()
    m=Model('indp')
    m.setParam('OutputFlag',False)
    if time_limit:
        m.setParam('TimeLimit', time_limit)
    G_prime_nodes = [n[0] for n in N.G.nodes(data=True) if n[1]['data']['inf_data'].net_id in layers]
    G_prime = N.G.subgraph(G_prime_nodes)
    # Damaged nodes in whole network
    N_prime = [n for n in G_prime.nodes(data=True) if n[1]['data']['inf_data'].repaired==0.0]
    # Nodes in controlled network.
    N_hat_nodes   = [n[0] for n in G_prime.nodes(data=True) if n[1]['data']['inf_data'].net_id in controlled_layers]
    N_hat = G_prime.subgraph(N_hat_nodes)
    # Damaged nodes in controlled network.
    N_hat_prime= [n for n in N_hat.nodes(data=True) if n[1]['data']['inf_data'].repaired==0.0]
    # Damaged arcs in whole network
    A_prime = [(u,v,a) for u,v,a in G_prime.edges(data=True) if a['data']['inf_data'].functionality==0.0]
    # Damaged arcs in controlled network.
    A_hat_prime = [(u,v,a) for u,v,a in A_prime if N_hat.has_node(u) and N_hat.has_node(v)]
    #print "A_hat_prime=",A_hat_prime
    S=N.S
    #print ""
    #print "New sim."
    # Populate interdepencies. Add nodes to N' if they currently rely on a non-functional node.
    interdep_nodes={}
    for u,v,a in G_prime.edges(data=True):
        if not functionality:
            if a['data']['inf_data'].is_interdep and G_prime.nodes[u]['data']['inf_data'].functionality == 0.0:
                #print "Dependency edge goes from:",u,"to",v
                if v not in interdep_nodes:
                    interdep_nodes[v]=[]
                interdep_nodes[v].append((u,a['data']['inf_data'].gamma))
        else:
            # Should populate N_hat with layers that are controlled. Then go through N_hat.edges(data=True)
            # to find interdependencies.
            for t in range(T):
                if t not in interdep_nodes:
                    interdep_nodes[t]={}
                if N_hat.has_node(v) and a['data']['inf_data'].is_interdep:
                    if functionality[t][u] == 0.0:
                        if v not in interdep_nodes[t]:
                            interdep_nodes[t][v]=[]
                        interdep_nodes[t][v].append((u,a['data']['inf_data'].gamma))
                    
    #print "N'=",[n for (n,d) in N_prime]
    for t in range(T):
        # Add geographical space variables.
        for s in S:
            m.addVar(name='z_'+str(s.id)+","+str(t),vtype=GRB.BINARY)
        # Add over/undersupply variables for each node.
        for n,d in N_hat.nodes(data=True):
            m.addVar(name='delta+_'+str(n)+","+str(t),lb=0.0)
            m.addVar(name='delta-_'+str(n)+","+str(t),lb=0.0)
        # Add functionality binary variables for each node in N'.
        for n,d in N_hat.nodes(data=True):
            m.addVar(name='w_'+str(n)+","+str(t),vtype=GRB.BINARY)
            if T > 1:
                m.addVar(name='w_tilde_'+str(n)+","+str(t),vtype=GRB.BINARY)
        # Add flow variables for each arc.
        for u,v,a in N_hat.edges(data=True):
            m.addVar(name='x_'+str(u)+","+str(v)+","+str(t),lb=0.0)
        # Add functionality binary variables for each arc in A'.
        for u,v,a in A_hat_prime:
            m.addVar(name='y_'+str(u)+","+str(v)+","+str(t),vtype=GRB.BINARY)
            if T > 1:
                m.addVar(name='y_tilde_'+str(u)+","+str(v)+","+str(t),vtype=GRB.BINARY)
    m.update()
    for t in range(T):
        for n,d in N_hat.nodes(data=True):
            if n not in interdep_nodes.keys() or decision_vars[t]['w_'+str(n)]==0: #Beacuse repaired interdpendent nodes may beforced to become non-functional by the dependee node
                m.getVarByName('w_'+str(n)+","+str(t)).lb=decision_vars[t]['w_'+str(n)]
                m.getVarByName('w_'+str(n)+","+str(t)).ub=decision_vars[t]['w_'+str(n)]
        for u,v,a in A_hat_prime:
            m.getVarByName('y_'+str(u)+","+str(v)+","+str(t)).lb=decision_vars[t]['y_'+str(u)+","+str(v)]
            m.getVarByName('y_'+str(u)+","+str(v)+","+str(t)).ub=decision_vars[t]['y_'+str(u)+","+str(v)]
    # Populate objective function.
    objFunc=LinExpr()
    for t in range(T):
        for s in S:
            objFunc+=s.cost*m.getVarByName('z_'+str(s.id)+","+str(t))
        for u,v,a in A_hat_prime:
            if T == 1:
                objFunc+=(float(a['data']['inf_data'].reconstruction_cost)/2.0)*m.getVarByName('y_'+str(u)+","+str(v)+","+str(t))
            else:
                objFunc+=(float(a['data']['inf_data'].reconstruction_cost)/2.0)*m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t))
        for n,d in N_hat_prime:
            if T == 1:
                objFunc+=d['data']['inf_data'].reconstruction_cost*m.getVarByName('w_'+str(n)+","+str(t))
            else:
                objFunc+=d['data']['inf_data'].reconstruction_cost*m.getVarByName('w_tilde_'+str(n)+","+str(t))
        for n,d in N_hat.nodes(data=True):
            objFunc+=d['data']['inf_data'].oversupply_penalty*m.getVarByName('delta+_'+str(n)+","+str(t))
            objFunc+=d['data']['inf_data'].undersupply_penalty*m.getVarByName('delta-_'+str(n)+","+str(t))
        for u,v,a in N_hat.edges(data=True):
            objFunc+=a['data']['inf_data'].flow_cost*m.getVarByName('x_'+str(u)+","+str(v)+","+str(t))
            
            
    m.setObjective(objFunc,GRB.MINIMIZE)
    m.update()
    
    #Constraints.
    # Time-dependent constraints.
    if T > 1:
        for n,d in N_hat.nodes(data=True):
            m.addConstr(m.getVarByName('w_'+str(n)+",0"),GRB.EQUAL,0)
        for u,v,a in A_hat_prime:
            m.addConstr(m.getVarByName('y_'+str(u)+","+str(v)+",0"),GRB.EQUAL,0)
        
        
    for t in range(T):
        # Time-dependent constraint.
        for n,d in N_hat.nodes(data=True):
            if t > 0:
                wTildeSum=LinExpr()
                for t_prime in range(1,t):
                    wTildeSum+=m.getVarByName('w_tilde_'+str(n)+","+str(t_prime))
                m.addConstr(m.getVarByName('w_'+str(n)+","+str(t)),GRB.LESS_EQUAL,wTildeSum,"Time dependent recovery constraint at node "+str(n)+","+str(t))
        for u,v,a in A_hat_prime:
            if t > 0:
                yTildeSum=LinExpr()
                for t_prime in range(1,t):
                    yTildeSum+=m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t_prime))
                m.addConstr(m.getVarByName('y_'+str(u)+","+str(v)+","+str(t)),GRB.LESS_EQUAL,yTildeSum,"Time dependent recovery constraint at arc "+str(u)+","+str(v)+","+str(t))
        # Enforce a_i,j to be fixed if a_j,i is fixed (and vice versa). 
        for u,v,a in A_hat_prime:
            #print u,",",v
            m.addConstr(m.getVarByName('y_'+str(u)+","+str(v)+","+str(t)),GRB.EQUAL,m.getVarByName('y_'+str(v)+","+str(u)+","+str(t)),"Arc reconstruction equality ("+str(u)+","+str(v)+","+str(t)+")")
            if T > 1:
                m.addConstr(m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t)),GRB.EQUAL,m.getVarByName('y_tilde_'+str(v)+","+str(u)+","+str(t)),"Arc reconstruction equality ("+str(u)+","+str(v)+","+str(t)+")")
        # Conservation of flow constraint. (2) in INDP paper.
        for n,d in N_hat.nodes(data=True):
            outFlowConstr=LinExpr()
            inFlowConstr= LinExpr()
            demandConstr= LinExpr()
            for u,v,a in N_hat.out_edges(n,data=True):
                outFlowConstr+=m.getVarByName('x_'+str(u)+","+str(v)+","+str(t))
            for u,v,a in N_hat.in_edges(n,data=True):
                inFlowConstr+= m.getVarByName('x_'+str(u)+","+str(v)+","+str(t))
            demandConstr+=d['data']['inf_data'].demand - m.getVarByName('delta+_'+str(n)+","+str(t)) + m.getVarByName('delta-_'+str(n)+","+str(t))
            m.addConstr(outFlowConstr-inFlowConstr,GRB.EQUAL,demandConstr,"Flow conservation constraint "+str(n)+","+str(t))
        # Flow functionality constraints.
        if not functionality:
            interdep_nodes_list = interdep_nodes.keys() #Interdepndent nodes with a damaged dependee node 
        else:
            interdep_nodes_list = interdep_nodes[t].keys() #Interdepndent nodes with a damaged dependee node 
        for u,v,a in N_hat.edges(data=True):
            if (u in [n for (n,d) in N_hat_prime]) | (u in interdep_nodes_list):
                m.addConstr(m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*m.getVarByName('w_'+str(u)+","+str(t)),"Flow in functionality constraint("+str(u)+","+str(v)+","+str(t)+")")
            else:
                m.addConstr(m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*N.G.nodes[u]['data']['inf_data'].functionality,"Flow in functionality constraint ("+str(u)+","+str(v)+","+str(t)+")")
            if (v in [n for (n,d) in N_hat_prime]) | (v in interdep_nodes_list):
                m.addConstr(m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*m.getVarByName('w_'+str(v)+","+str(t)),"Flow out functionality constraint("+str(u)+","+str(v)+","+str(t)+")")
            else:
                m.addConstr(m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*N.G.nodes[v]['data']['inf_data'].functionality,"Flow out functionality constraint ("+str(u)+","+str(v)+","+str(t)+")")
            if (u,v,a) in A_hat_prime:
                m.addConstr(m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*m.getVarByName('y_'+str(u)+","+str(v)+","+str(t)),"Flow arc functionality constraint ("+str(u)+","+str(v)+","+str(t)+")")
            else:
                m.addConstr(m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*N.G[u][v]['data']['inf_data'].functionality,"Flow arc functionality constraint("+str(u)+","+str(v)+","+str(t)+")")

        #Resource availability constraints.
        # isSepResource = 0
        # if isinstance(v_r, (int, long)):
        #     totalResource = v_r
        # else:
        #     if len(v_r) != 1:
        #         isSepResource = 1
        #         totalResource = sum(v_r)
        #         if len(v_r) != len(layers):
        #             print "\n***ERROR: The number of resource cap values does not match the number of layers.***\n"
        #             sys.exit()
        #     else:
        #         totalResource = v_r[0]
            
        # resourceLeftConstr=LinExpr()
        # if isSepResource:
        #     resourceLeftConstrSep = [LinExpr() for i in range(len(v_r))]
                                     
        # for u,v,a in A_hat_prime:
        #     indexLayer = a['data']['inf_data'].layer - 1
        #     if T == 1:
        #         resourceLeftConstr+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_'+str(u)+","+str(v)+","+str(t))
        #         if isSepResource:
        #             resourceLeftConstrSep[indexLayer]+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_'+str(u)+","+str(v)+","+str(t))
        #     else:
        #         resourceLeftConstr+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t))
        #         if isSepResource:
        #             resourceLeftConstrSep[indexLayer]+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t))

        # for n,d in N_hat_prime:
        #     indexLayer = n[1] - 1
        #     if T == 1:
        #         resourceLeftConstr+=d['data']['inf_data'].resource_usage*m.getVarByName('w_'+str(n)+","+str(t))
        #         if isSepResource:
        #             resourceLeftConstrSep[indexLayer]+=d['data']['inf_data'].resource_usage*m.getVarByName('w_'+str(n)+","+str(t))
        #     else:
        #         resourceLeftConstr+=d['data']['inf_data'].resource_usage*m.getVarByName('w_tilde_'+str(n)+","+str(t))
        #         if isSepResource:
        #             resourceLeftConstrSep[indexLayer]+=d['data']['inf_data'].resource_usage*m.getVarByName('w_tilde_'+str(n)+","+str(t))

        # m.addConstr(resourceLeftConstr,GRB.LESS_EQUAL,totalResource,"Resource availability constraint at "+str(t)+".")
        # if isSepResource:
        #     for k in range(len(v_r)):
        #         m.addConstr(resourceLeftConstrSep[k],GRB.LESS_EQUAL,v_r[k],"Resource availability constraint at "+str(t)+ " for layer "+`k`+".")

        # Interdependency constraints
        infeasible_actions=[]
        for n,d in N_hat.nodes(data=True):
            if not functionality:
                if n in interdep_nodes:
                    interdepLConstr=LinExpr()
                    interdepRConstr=LinExpr()
                    for interdep in interdep_nodes[n]:
                        src=interdep[0]
                        gamma=interdep[1]
                        if not N_hat.has_node(src):
                            infeasible_actions.append(n)
                            interdepLConstr+=0
                        else:
                            interdepLConstr+=m.getVarByName('w_'+str(src)+","+str(t))*gamma
                    interdepRConstr+=m.getVarByName('w_'+str(n)+","+str(t))
                    m.addConstr(interdepLConstr,GRB.GREATER_EQUAL,interdepRConstr,"Interdependency constraint for node "+str(n)+","+str(t))
            else:
                if n in interdep_nodes[t]:
                    #print interdep_nodes[t]
                    interdepLConstr=LinExpr()
                    interdepRConstr=LinExpr()
                    for interdep in interdep_nodes[t][n]:
                        src=interdep[0]
                        gamma=interdep[1]
                        if not N_hat.has_node(src):
                            if print_cmd:
                                print( "Forcing",str(n),"to be 0 (dep. on",str(src),")")
                            infeasible_actions.append(n)
                            interdepLConstr+=0
                        else:
                            interdepLConstr+=m.getVarByName('w_'+str(src)+","+str(t))*gamma
                    interdepRConstr+=m.getVarByName('w_'+str(n)+","+str(t))
                    m.addConstr(interdepLConstr,GRB.GREATER_EQUAL,interdepRConstr,"Interdependency constraint for node "+str(n)+","+str(t))

        # Forced actions (if applicable)
        if forced_actions:
            recovery_sum=LinExpr()
            feasible_nodes=[(n,d) for n,d in N_hat_prime if n not in infeasible_actions] 
            if len(feasible_nodes) + len(A_hat_prime) > 0:
                for n,d in feasible_nodes:
                    if T == 1:
                        recovery_sum+=m.getVarByName('w_'+str(n)+","+str(t))
                    else:
                        recovery_sum+=m.getVarByName('w_tilde_'+str(n)+","+str(t))
                for u,v,a in A_hat_prime:
                    if T == 1:
                        recovery_sum+=m.getVarByName('y_'+str(u)+","+str(v)+","+str(t))
                    else:
                        recovery_sum+=m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t))
                m.addConstr(recovery_sum,GRB.GREATER_EQUAL,1,"Forced action constraint")
                                
        # Geographic space constraints
        for s in S:
            for n,d in N_hat_prime:
                if T == 1:
                    m.addConstr(m.getVarByName('w_'+str(n)+","+str(t))*d['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+str(s.id)+","+str(t)),"Geographical space constraint for node "+str(n)+","+str(t))
                else:
                    m.addConstr(m.getVarByName('w_tilde_'+str(n)+","+str(t))*d['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+str(s.id)+","+str(t)),"Geographical space constraint for node "+str(n)+","+str(t))
            for u,v,a in A_hat_prime:
                if T== 1:
                    m.addConstr(m.getVarByName('y_'+str(u)+","+str(v)+","+str(t))*a['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+str(s.id)+","+str(t)),"Geographical space constraint for arc ("+str(u)+","+str(v)+")")
                else:
                    m.addConstr(m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t))*a['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+str(s.id)+","+str(t)),"Geographical space constraint for arc ("+str(u)+","+str(v)+")")
      
#    print "Solving..."
    m.update()
    m.optimize()
    results=[]
    run_time = time.time()-start_time
    # Save results.
    if m.getAttr("Status")==GRB.OPTIMAL or m.status==9:
        if m.status==9:
            print ('\nOptimizer time limit, gap = %1.3f\n' % m.MIPGap)

        results=collect_results(m,controlled_layers,T,N_hat,N_hat_prime,A_hat_prime,decision_vars,S)

        results[0].add_run_time(t,run_time)  
        for l in controlled_layers:
            results[1][l].add_run_time(t,run_time) 
            
        return [m,results[0],results[1]]
    else:
        print( m.getAttr("Status"),": SOLUTION NOT FOUND. (Check data and/or violated constraints).")
        m.computeIIS()
        print('\nThe following constraint(s) cannot be satisfied:')
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
        return None
    
def collect_results(m,controlled_layers,T,N_hat,N_hat_prime,A_hat_prime,decision_vars,S):
    layers = controlled_layers
    indp_results=INDPResults()
    layer_results={l:INDPResults() for l in layers}
    # compute total demand of all layers and each layer
    total_demand = 0.0
    total_demand_layer={l:0.0 for l in layers}
    for n,d in N_hat.nodes(data=True):
        demand_value = d['data']['inf_data'].demand
        if demand_value<0:
            total_demand+=demand_value
            total_demand_layer[n[1]]+=demand_value
    for t in range(T):
        nodeCost=0.0
        arcCost=0.0
        flowCost=0.0
        overSuppCost=0.0
        underSuppCost=0.0
        underSupp=0.0
        spacePrepCost=0.0
        nodeCost_layer={l:0.0 for l in layers}
        arcCost_layer={l:0.0 for l in layers}
        flowCost_layer={l:0.0 for l in layers}
        overSuppCost_layer={l:0.0 for l in layers}
        underSuppCost_layer={l:0.0 for l in layers}
        underSupp_layer={l:0.0 for l in layers}
        spacePrepCost_layer={l:0.0 for l in layers}
        # Record node recovery actions.
        for n,d in N_hat_prime:
            nodeVar='w_tilde_'+str(n)+","+str(t)
            if T == 1:
                nodeVar='w_'+str(n)+","+str(t)
            if round(m.getVarByName(nodeVar).x)==1 or decision_vars[t]['w_'+str(n)]==1:
                action=str(n[0])+"."+str(n[1])
                indp_results.add_action(t,action)
                layer_results[n[1]].add_action(t,action)
                    #if T == 1:
                    #N.G.nodes[n]['data']['inf_data'].functionality=1.0
        # Record edge recovery actions.
        for u,v,a in A_hat_prime:
            arcVar='y_tilde_'+str(u)+","+str(v)+","+str(t)
            if T == 1:
                arcVar='y_'+str(u)+","+str(v)+","+str(t)
            if round(m.getVarByName(arcVar).x)==1 or decision_vars[t]['y_'+str(u)+","+str(v)]==1:
                action=str(u[0])+"."+str(u[1])+"/"+str(v[0])+"."+str(v[1])
                indp_results.add_action(t,action)
                layer_results[u[1]].add_action(t,action)
                #if T == 1:
                #N.G[u][v]['data']['inf_data'].functionality=1.0
        # Calculate space preparation costs.
        for s in S:
            spacePrepCost+=s.cost*m.getVarByName('z_'+str(s.id)+","+str(t)).x
        indp_results.add_cost(t,"Space Prep",spacePrepCost)
        # Calculate arc preparation costs.
        for u,v,a in A_hat_prime:
            arcVar='y_tilde_'+str(u)+","+str(v)+","+str(t)
            if T == 1:
                arcVar='y_'+str(u)+","+str(v)+","+str(t)
            arcCost+=(a['data']['inf_data'].reconstruction_cost/2.0)*m.getVarByName(arcVar).x
            arcCost_layer[u[1]]+=(a['data']['inf_data'].reconstruction_cost/2.0)*m.getVarByName(arcVar).x
            if m.getVarByName(arcVar).x==0 and decision_vars[t]['y_'+str(u)+","+str(v)]==1:
                arcCost+=a['data']['inf_data'].reconstruction_cost/2.0
                arcCost_layer[u[1]]+=a['data']['inf_data'].reconstruction_cost/2.0
        indp_results.add_cost(t,"Arc",arcCost)
        for l in layers:
            layer_results[l].add_cost(t,"Arc", arcCost_layer[l])
        # Calculate node preparation costs.
        for n,d in N_hat_prime:
            nodeVar = 'w_tilde_'+str(n)+","+str(t)
            if T == 1:
                nodeVar = 'w_'+str(n)+","+str(t)
            nodeCost+=d['data']['inf_data'].reconstruction_cost*m.getVarByName(nodeVar).x
            nodeCost_layer[n[1]]+=d['data']['inf_data'].reconstruction_cost*m.getVarByName(nodeVar).x
            if m.getVarByName(nodeVar).x==0 and decision_vars[t]['w_'+str(n)]==1:
                nodeCost+=d['data']['inf_data'].reconstruction_cost
                nodeCost_layer[n[1]]+=d['data']['inf_data'].reconstruction_cost
        indp_results.add_cost(t,"Node",nodeCost)
        for l in layers:
            layer_results[l].add_cost(t,"Node",nodeCost_layer[l])
        # Calculate under/oversupply costs.
        for n,d in N_hat.nodes(data=True):
            overSuppCost+= d['data']['inf_data'].oversupply_penalty*m.getVarByName('delta+_'+str(n)+","+str(t)).x
            overSuppCost_layer[n[1]]+= d['data']['inf_data'].oversupply_penalty*m.getVarByName('delta+_'+str(n)+","+str(t)).x
            underSupp+= m.getVarByName('delta+_'+str(n)+","+str(t)).x
            underSupp_layer[n[1]]+= m.getVarByName('delta+_'+str(n)+","+str(t)).x
            underSuppCost+=d['data']['inf_data'].undersupply_penalty*m.getVarByName('delta-_'+str(n)+","+str(t)).x
            underSuppCost_layer[n[1]]+=d['data']['inf_data'].undersupply_penalty*m.getVarByName('delta-_'+str(n)+","+str(t)).x
        indp_results.add_cost(t,"Over Supply",overSuppCost)
        indp_results.add_cost(t,"Under Supply",underSuppCost)
        indp_results.add_cost(t,"Under Supply Perc",underSupp/total_demand)
        for l in layers:
            layer_results[l].add_cost(t,"Over Supply",overSuppCost_layer[l])
            layer_results[l].add_cost(t,"Under Supply",underSuppCost_layer[l])
            layer_results[l].add_cost(t,"Under Supply Perc",underSupp_layer[l]/total_demand_layer[l])
        # Calculate flow costs.
        for u,v,a in N_hat.edges(data=True):
            flowCost+=a['data']['inf_data'].flow_cost*m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)).x
            flowCost_layer[u[1]]+=a['data']['inf_data'].flow_cost*m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)).x
        indp_results.add_cost(t,"Flow",flowCost)
        for l in layers:
            layer_results[l].add_cost(t,"Flow",flowCost_layer[l])
        # Calculate total costs.
        indp_results.add_cost(t,"Total",flowCost+arcCost+nodeCost+overSuppCost+underSuppCost+spacePrepCost)
        indp_results.add_cost(t,"Total no disconnection",spacePrepCost+arcCost+flowCost+nodeCost)           
        for l in layers:
            layer_results[l].add_cost(t,"Total",flowCost_layer[l]+arcCost_layer[l]+nodeCost_layer[l]+overSuppCost_layer[l]+underSuppCost_layer[l]+spacePrepCost_layer[l])
            layer_results[l].add_cost(t,"Total no disconnection",spacePrepCost_layer[l]+arcCost_layer[l]+flowCost+nodeCost_layer[l])
    return [indp_results,layer_results]