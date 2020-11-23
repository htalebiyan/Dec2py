from infrastructure import *
from indputils import *
from gurobipy import *
import string
#import platform
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random
import time
import sys
#HOME_DIR="/Users/Andrew/"
#if platform.system() == "Linux":
#    HOME_DIR="/home/andrew/"

def indp(N,v_r,T=1,layers=[1,3],controlled_layers=[1,3],functionality={},
         forced_actions=False, fixed_nodes={}, print_cmd=True, time_limit=None,
         co_location=True):
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
    N_hat_nodes = [n[0] for n in G_prime.nodes(data=True) if n[1]['data']['inf_data'].net_id in controlled_layers]
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
        if co_location:
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
        # Fix node values (only for iINDP)
        m.update()
        for key, val in fixed_nodes.items():
            m.getVarByName('w_'+str(key)+","+str(0)).lb=val
            m.getVarByName('w_'+str(key)+","+str(0)).ub=val
        # Add flow variables for each arc.
        for u,v,a in N_hat.edges(data=True):
            m.addVar(name='x_'+str(u)+","+str(v)+","+str(t),lb=0.0)
        # Add functionality binary variables for each arc in A'.
        for u,v,a in A_hat_prime:
            m.addVar(name='y_'+str(u)+","+str(v)+","+str(t),vtype=GRB.BINARY)
            if T > 1:
                m.addVar(name='y_tilde_'+str(u)+","+str(v)+","+str(t),vtype=GRB.BINARY)
    m.update()

    # Populate objective function.
    objFunc=LinExpr()
    for t in range(T):
        if co_location:
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
        for n,d in N_hat_prime:
            m.addConstr(m.getVarByName('w_'+str(n)+",0"),GRB.EQUAL,0, "Initial state at node "+str(n)+","+str(0))
        for u,v,a in A_hat_prime:
            m.addConstr(m.getVarByName('y_'+str(u)+","+str(v)+",0"),GRB.EQUAL,0,"Initial state at arc "+str(u)+","+str(v)+","+str(0))


    for t in range(T):
        # Time-dependent constraint.
        for n,d in N_hat_prime:
            if t > 0:
                wTildeSum=LinExpr()
                for t_prime in range(1,t+1):
                    wTildeSum+=m.getVarByName('w_tilde_'+str(n)+","+str(t_prime))
                m.addConstr(m.getVarByName('w_'+str(n)+","+str(t)),GRB.LESS_EQUAL,wTildeSum,"Time dependent recovery constraint at node "+str(n)+","+str(t))
        for u,v,a in A_hat_prime:
            if t > 0:
                yTildeSum=LinExpr()
                for t_prime in range(1,t+1):
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
        isSepResource = False
        if isinstance(v_r, int):
            totalResource = v_r
        else:
            isSepResource = True
            totalResource = sum([val for _, val in v_r.items()])
            if len(v_r.keys()) != len(layers):
                sys.exit("The number of resource cap values does not match the number of layers.\n")

        resourceLeftConstr=LinExpr()
        if isSepResource:
            resourceLeftConstrSep = {key:LinExpr() for key, _ in v_r.items()}

        for u,v,a in A_hat_prime:
            indexLayer = a['data']['inf_data'].layer
            if T == 1:
                resourceLeftConstr+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_'+str(u)+","+str(v)+","+str(t))
                if isSepResource:
                    resourceLeftConstrSep[indexLayer]+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_'+str(u)+","+str(v)+","+str(t))
            else:
                resourceLeftConstr+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t))
                if isSepResource:
                    resourceLeftConstrSep[indexLayer]+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t))

        for n,d in N_hat_prime:
            indexLayer = n[1]
            if T == 1:
                resourceLeftConstr+=d['data']['inf_data'].resource_usage*m.getVarByName('w_'+str(n)+","+str(t))
                if isSepResource:
                    resourceLeftConstrSep[indexLayer]+=d['data']['inf_data'].resource_usage*m.getVarByName('w_'+str(n)+","+str(t))
            else:
                resourceLeftConstr+=d['data']['inf_data'].resource_usage*m.getVarByName('w_tilde_'+str(n)+","+str(t))
                if isSepResource:
                    resourceLeftConstrSep[indexLayer]+=d['data']['inf_data'].resource_usage*m.getVarByName('w_tilde_'+str(n)+","+str(t))

        m.addConstr(resourceLeftConstr,GRB.LESS_EQUAL,totalResource,"Resource availability constraint at "+str(t)+".")
        if isSepResource:
            for k,_ in v_r.items():
                m.addConstr(resourceLeftConstrSep[k],GRB.LESS_EQUAL,v_r[k],"Resource availability constraint at "+str(t)+" for layer "+str(k)+".")

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
                                print("Forcing",str(n),"to be 0 (dep. on",str(src),")")
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
        if co_location:
            for s in S:
                for n,d in N_hat_prime:
                    if d['data']['inf_data'].in_space(s.id):
                        if T == 1:
                            m.addConstr(m.getVarByName('w_'+str(n)+","+str(t))*d['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+str(s.id)+","+str(t)),"Geographical space constraint for node "+str(n)+","+str(t))
                        else:
                            m.addConstr(m.getVarByName('w_tilde_'+str(n)+","+str(t))*d['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+str(s.id)+","+str(t)),"Geographical space constraint for node "+str(n)+","+str(t))
                for u,v,a in A_hat_prime:
                    if a['data']['inf_data'].in_space(s.id):
                        if T== 1:
                            m.addConstr(m.getVarByName('y_'+str(u)+","+str(v)+","+str(t))*a['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+str(s.id)+","+str(t)),"Geographical space constraint for arc ("+str(u)+","+str(v)+")")
                        else:
                            m.addConstr(m.getVarByName('y_tilde_'+str(u)+","+str(v)+","+str(t))*a['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+str(s.id)+","+str(t)),"Geographical space constraint for arc ("+str(u)+","+str(v)+")")

#    print "Solving..."
    m.update()
    m.optimize()
    run_time = time.time()-start_time
    # Save results.
    if m.getAttr("Status")==GRB.OPTIMAL or m.status==9:
        if m.status==9:
            print ('\nOptimizer time limit, gap = %1.3f\n' % m.MIPGap)
        results=collect_results(m,controlled_layers,T,N_hat,N_hat_prime,A_hat_prime,S,coloc=co_location)
        results.add_run_time(t,run_time)
        return [m,results]
    else:
        print(m.getAttr("Status"),": SOLUTION NOT FOUND. (Check data and/or violated constraints).")
        m.computeIIS()
        print ('\nThe following constraint(s) cannot be satisfied:')
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
        return None

def collect_results(m,controlled_layers,T,N_hat,N_hat_prime,A_hat_prime,S,coloc=True):
    layers = controlled_layers
    indp_results=INDPResults(layers)
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
        spacePrepCost_layer={l:0.0 for l in layers} #!!! populate this for each layer
        # Record node recovery actions.
        for n,d in N_hat_prime:
            nodeVar='w_tilde_'+str(n)+","+str(t)
            if T == 1:
                nodeVar='w_'+str(n)+","+str(t)
            if round(m.getVarByName(nodeVar).x)==1:
                action=str(n[0])+"."+str(n[1])
                indp_results.add_action(t,action)
        # Record edge recovery actions.
        for u,v,a in A_hat_prime:
            arcVar='y_tilde_'+str(u)+","+str(v)+","+str(t)
            if T == 1:
                arcVar='y_'+str(u)+","+str(v)+","+str(t)
            if round(m.getVarByName(arcVar).x)==1:
                action=str(u[0])+"."+str(u[1])+"/"+str(v[0])+"."+str(v[1])
                indp_results.add_action(t,action)
        # Calculate space preparation costs.
        if coloc:
            for s in S:
                spacePrepCost+=s.cost*m.getVarByName('z_'+str(s.id)+","+str(t)).x
        indp_results.add_cost(t,"Space Prep",spacePrepCost,spacePrepCost_layer)
        # Calculate arc preparation costs.
        for u,v,a in A_hat_prime:
            arcVar='y_tilde_'+str(u)+","+str(v)+","+str(t)
            if T == 1:
                arcVar='y_'+str(u)+","+str(v)+","+str(t)
            arcCost+=(a['data']['inf_data'].reconstruction_cost/2.0)*m.getVarByName(arcVar).x
            arcCost_layer[u[1]]+=(a['data']['inf_data'].reconstruction_cost/2.0)*m.getVarByName(arcVar).x
        indp_results.add_cost(t,"Arc",arcCost,arcCost_layer)
        # Calculate node preparation costs.
        for n,d in N_hat_prime:
            nodeVar = 'w_tilde_'+str(n)+","+str(t)
            if T == 1:
                nodeVar = 'w_'+str(n)+","+str(t)
            nodeCost+=d['data']['inf_data'].reconstruction_cost*m.getVarByName(nodeVar).x
            nodeCost_layer[n[1]]+=d['data']['inf_data'].reconstruction_cost*m.getVarByName(nodeVar).x
        indp_results.add_cost(t,"Node",nodeCost,nodeCost_layer)
        # Calculate under/oversupply costs.
        for n,d in N_hat.nodes(data=True):
            overSuppCost+= d['data']['inf_data'].oversupply_penalty*m.getVarByName('delta+_'+str(n)+","+str(t)).x
            overSuppCost_layer[n[1]]+= d['data']['inf_data'].oversupply_penalty*m.getVarByName('delta+_'+str(n)+","+str(t)).x
            underSupp+= m.getVarByName('delta-_'+str(n)+","+str(t)).x
            underSupp_layer[n[1]]+= m.getVarByName('delta-_'+str(n)+","+str(t)).x/total_demand_layer[n[1]]
            underSuppCost+=d['data']['inf_data'].undersupply_penalty*m.getVarByName('delta-_'+str(n)+","+str(t)).x
            underSuppCost_layer[n[1]]+=d['data']['inf_data'].undersupply_penalty*m.getVarByName('delta-_'+str(n)+","+str(t)).x
        indp_results.add_cost(t,"Over Supply",overSuppCost,overSuppCost_layer)
        indp_results.add_cost(t,"Under Supply",underSuppCost,underSuppCost_layer)
        indp_results.add_cost(t,"Under Supply Perc",underSupp/total_demand,underSupp_layer)
        # Calculate flow costs.
        for u,v,a in N_hat.edges(data=True):
            flowCost+=a['data']['inf_data'].flow_cost*m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)).x
            flowCost_layer[u[1]]+=a['data']['inf_data'].flow_cost*m.getVarByName('x_'+str(u)+","+str(v)+","+str(t)).x
        indp_results.add_cost(t,"Flow",flowCost,flowCost_layer)
        # Calculate total costs.
        total_lyr={}
        total_nd_lyr={}
        for l in layers:
            total_lyr[l] = flowCost_layer[l]+arcCost_layer[l]+nodeCost_layer[l]+\
                overSuppCost_layer[l]+underSuppCost_layer[l]+spacePrepCost_layer[l]
            total_nd_lyr[l] = spacePrepCost_layer[l]+arcCost_layer[l]+flowCost+nodeCost_layer[l]
        indp_results.add_cost(t,"Total",flowCost+arcCost+nodeCost+overSuppCost+\
                              underSuppCost+spacePrepCost,total_lyr)
        indp_results.add_cost(t,"Total no disconnection",spacePrepCost+arcCost+\
                              flowCost+nodeCost,total_nd_lyr)
    return indp_results

def apply_recovery(N,indp_results,t):
    for action in indp_results[t]['actions']:
        if "/" in action:
            # Edge recovery action.
            data=action.split("/")
            src=tuple([int(x) for x in data[0].split(".")])
            dst=tuple([int(x) for x in data[1].split(".")])
            N.G[src][dst]['data']['inf_data'].functionality=1.0
        else:
            # Node recovery action.
            node=tuple([int(x) for x in action.split(".")])
            #print "Applying recovery:",node
            N.G.nodes[node]['data']['inf_data'].repaired=1.0
            N.G.nodes[node]['data']['inf_data'].functionality=1.0

def create_functionality_matrix(N,T,layers,actions,strategy_type="OPTIMISTIC"):
    """Creates a functionality map for input into the functionality parameter in the indp function.
    :param N: An InfrastructureNetwork instance (created in infrastructure.py)
    :param T: Number of timesteps to optimize over.
    :param layers: Layer IDs of N included in the optimization.
    :param actions: An array of actions from a previous optimization. Likely taken from an INDPResults variable 'indp_result[t]['actions']'.
    :param strategy_type: If no actions are provided, assigns a default functionality. Options are: "OPTIMISTIC", "PESSIMISTIC" or "INFO_SHARE"
    :returns: A functionality dictionary used for input into indp.
    """
    functionality={}
    G_prime_nodes = [n[0] for n in N.G.nodes(data=True) if n[1]['data']['inf_data'].net_id in layers]
    G_prime = N.G.subgraph(G_prime_nodes)
    N_prime = [n for n in G_prime.nodes(data=True) if n[1]['data']['inf_data'].repaired==0.0]
    for t in range(T):
        functionality[t]={}
        functional_nodes=[]
        for t_p in range(t):
            for key in functionality[t_p]:
                if functionality[t_p][key]==1.0:
                    functionality[t][key]=1.0
        if strategy_type == "INFO_SHARE":
            for a in actions[t]:
                if a and not "/" in a:
                    node=int(string.split(a,".")[0])
                    layer=int(string.split(a,".")[1])
                    if layer in layers:
                        functional_nodes.append((node,layer))
        for n,d in G_prime.nodes(data=True):
            #print "layers=",layers,"n=",n
            if d['data']['inf_data'].net_id in layers:
                if (n,d) in N_prime and n in functional_nodes:
                    functionality[t][n]=1.0
                elif G_prime.has_node(n) and (n,d) not in N_prime:
                    functionality[t][n]=1.0
                else:
                    if strategy_type == "OPTIMISTIC":
                        functionality[t][n]=1.0
                    elif strategy_type == "PESSIMISTIC":
                        functionality[t][n]=0.0
                    elif strategy_type == "REALISTIC":
                        functionality[t][n]=d['data']['inf_data'].functionality
                    else:
                        if not n in functionality[t]:
                            functionality[t][n]=0.0
    return functionality

def initialize_network(BASE_DIR="../data/INDP_7-20-2015/",external_interdependency_dir=None,sim_number=1,cost_scale=1,magnitude=6,sample=0,v=3,shelby_data=True,topology='Random'):
    """ Initializes an InfrastructureNetwork from Shelby County data.
    :param BASE_DIR: Base directory of Shelby County data.
    :param sim_number: Which simulation number to use as input.
    :param cost_scale: Scales the cost to improve efficiency.
    :param magnitude: Magnitude of earthquake.
    :param v: v parameter
    :returns: An interdependent InfrastructureNetwork.
    """
    layers_temp=[]
    v_temp = 0
    if shelby_data:
    #    print "Loading Shelby County data..." #!!!
        InterdepNet=load_infrastructure_data(BASE_DIR=BASE_DIR,
                                             external_interdependency_dir=external_interdependency_dir,
                                             sim_number=sim_number, cost_scale=cost_scale,
                                             magnitude=magnitude, v=v,
                                             shelby_data=shelby_data)
    #    print "Data loaded." #!!!
    else:
        InterdepNet,v_temp,layers_temp=load_synthetic_network(BASE_DIR=BASE_DIR,topology=topology,config=magnitude,sample=sample,cost_scale=cost_scale)
    return InterdepNet,v_temp,layers_temp

def run_indp(params, layers=[1,2,3], controlled_layers=[], functionality={},T=1, validate=False,
             save=True,suffix="", forced_actions=False, saveModel=False, print_cmd_line=True,
             dynamic_params=None, co_location=True):
    """ Runs an INDP problem with specified parameters. Outputs to directory specified in params['OUTPUT_DIR'].
    :param params: Global parameters.
    :param layers: Layers to consider in the infrastructure network.
    :param T: Number of timesteps to optimize over.
    :param validate: Validate solution.
    """
    # Initialize failure scenario.
    InterdepNet=None
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=params['SIM_NUMBER'],magnitude=params["MAGNITUDE"])
    else:
        InterdepNet=params["N"]
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"] = 1
    if not controlled_layers:
        controlled_layers = layers

    v_r=params["V"]
    if isinstance(v_r, (int)):
        outDirSuffixRes = str(v_r)
    else:
        outDirSuffixRes = str(sum([val for _, val in v_r.items()]))+'_fixed_layer_Cap'

    indp_results=INDPResults(params["L"])
    if T == 1:
        print("--Running INDP (T=1) or iterative INDP.")
        if print_cmd_line:
            print("Num iters=",params["NUM_ITERATIONS"])

        # Run INDP for 1 time step (original INDP).
        output_dir=params["OUTPUT_DIR"]+'_L'+str(len(layers))+'_m'+str(params["MAGNITUDE"])+"_v"+outDirSuffixRes
        # Initial calculations.
        if dynamic_params:
            original_N = copy.deepcopy(InterdepNet) #!!! deepcopy
            dynamic_parameters(InterdepNet, original_N, 0, dynamic_params)
        results=indp(InterdepNet,0,1,layers,controlled_layers=controlled_layers,
                     functionality=functionality, co_location=co_location)
        indp_results=results[1]
        indp_results.add_components(0,INDPComponents.calculate_components(results[0],InterdepNet,layers=controlled_layers))
        for i in range(params["NUM_ITERATIONS"]):
            print("-Time Step (iINDP)",i+1,"/",params["NUM_ITERATIONS"])
            if dynamic_params:
                dynamic_parameters(InterdepNet, original_N, i+1, dynamic_params)
            results=indp(InterdepNet, v_r, T, layers, controlled_layers=controlled_layers,
                         forced_actions=forced_actions, co_location=co_location)
            indp_results.extend(results[1],t_offset=i+1)
            if saveModel:
                save_INDP_model_to_file(results[0],output_dir+"/Model",i+1)
            # Modify network to account for recovery and calculate components.
            apply_recovery(InterdepNet,indp_results,i+1)
            indp_results.add_components(i+1,INDPComponents.calculate_components(results[0],InterdepNet,layers=controlled_layers))
#            print "Num_iters=",params["NUM_ITERATIONS"]
    else:
        # td-INDP formulations. Includes "DELTA_T" parameter for sliding windows to increase
        # efficiency.
        # Edit 2/8/16: "Sliding window" now overlaps.
        num_time_windows=1
        time_window_length=T
        is_first_iteration=True
        if "WINDOW_LENGTH" in params:
            time_window_length=params["WINDOW_LENGTH"]
            num_time_windows=T
        output_dir=params["OUTPUT_DIR"]+'_L'+str(len(layers))+"_m"+str(params["MAGNITUDE"])+"_v"+outDirSuffixRes

        print("Running td-INDP (T="+str(T)+", Window size="+str(time_window_length)+")")
        # Initial percolation calculations.
        results=indp(InterdepNet,0,1,layers,controlled_layers=controlled_layers,
                     functionality=functionality, co_location=co_location)
        indp_results=results[1]
        indp_results.add_components(0,INDPComponents.calculate_components(results[0],InterdepNet,layers=controlled_layers))
        for n in range(num_time_windows):
            print("-Time window (td-INDP)",n+1,"/",num_time_windows)
            functionality_t={}
            # Slide functionality matrix according to sliding time window.
            if functionality:
                for t in functionality:
                    if t in range(n,time_window_length+n+1):
                        functionality_t[t-n]=functionality[t]
                if len(functionality_t) < time_window_length+1:
                    diff=time_window_length+1-len(functionality_t)
                    max_t=max(functionality_t.keys())
                    for d in range(diff):
                        functionality_t[max_t+d+1]=functionality_t[max_t]
            # Run td-INDP.
            results=indp(InterdepNet, v_r, time_window_length+1, layers, 
                         controlled_layers=controlled_layers, 
                         functionality=functionality_t, forced_actions=forced_actions,
                         co_location=co_location)
            if saveModel:
                save_INDP_model_to_file(results[0],output_dir+"/Model",n+1)
            if "WINDOW_LENGTH" in params:
                indp_results.extend(results[1],t_offset=n+1,t_start=1,t_end=2)
                # Modify network for recovery actions and calculate components.
                apply_recovery(InterdepNet,results[1],1)
                indp_results.add_components(n+1,
                                            INDPComponents.calculate_components(results[0],
                                                                                InterdepNet,1,
                                                                                layers=controlled_layers))
            else:
                indp_results.extend(results[1],t_offset=0)
                for t in range(1,T):
                    # Modify network to account for recovery actions.
                    apply_recovery(InterdepNet,indp_results,t)
                    indp_results.add_components(1,INDPComponents.calculate_components(results[0],InterdepNet,t,layers=controlled_layers))
    # Save results of current simulation.
    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        indp_results.to_csv(output_dir,params["SIM_NUMBER"],suffix=suffix)
        if not os.path.exists(output_dir+'/agents'):
            os.makedirs(output_dir+'/agents')
        indp_results.to_csv_layer(output_dir+'/agents',params["SIM_NUMBER"],suffix=suffix)
    return indp_results


def run_info_share(params,layers=[1,2,3],T=1,validate=False,suffix=""):
    """Applies rounds of information sharing between INDP runs. Assumes each layer is controlled by a separate player.
    params["NUM_ITERATIONS"] determines how many rounds of sharing to perform.
    NOTE: This is still not fully functional.
    :param params: Global parameters.
    :param layers: Specifies which layers to optimize over.
    :param T: Number of timesteps to optimize over.
    :param validate: (Currently not used.)
    """
    # Initialize network.
    InterdepNet=None
    num_iterations=params["NUM_ITERATIONS"]
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=params['SIM_NUMBER'],magnitude=params["MAGNITUDE"])
        params["N"]=InterdepNet
    else:
        InterdepNet=params["N"]
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"]=1
    v_r=params["V"]
    # Initialize player result variables.
    player_strategies={}
    for P in layers:
        player_strategies[P]=INDPResults()

    # Begin Sharing Process.
    for i in range(num_iterations):
        results={}
        for P in layers:
            negP=[x for x in layers if x != P]
            print("P=",str(P),"i=",str(i))
            if i == 0:
                # Create functionality matrix. This corresponds to "OPTIMISTIC" or "PESSIMISTIC" in Sharkey paper.
                # OPTIMISTIC implies that interdependencies are assumed to be fixed whenever a player needs them to be.
                # PESSIMISTIC assumes interdependencies never become fixed.
                functionality=create_functionality_matrix(InterdepNet,T,negP,actions=None,strategy_type="OPTIMISTIC")
            else:
                print( "Next iteration!")
                actions=[]
                for t in range(T):
                    for l in negP:
                        actions.append(player_strategies[l][t]["actions"])
                functionality=create_functionality_matrix(InterdepNet,T,negP,actions=actions,strategy_type="INFO_SHARE")
            params["N"]=InterdepNet.copy()
            results[P]=run_indp(params,layers,controlled_layers=[P],T=T,functionality=functionality,save=True,suffix="P"+str(P)+"_i"+str(i)+"_"+suffix,forced_actions=True)
        for P in layers:
            player_strategies[P]=results[P]

def run_inrg(params,layers=[1,2,3],validate=False,player_ordering=[3,1],suffix=""):
    InterdepNet=None
    output_dir=params["OUTPUT_DIR"]+"_m"+str(params["MAGNITUDE"])+"_v"+str(params["V"])
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=params['SIM_NUMBER'],magnitude=params["MAGNITUDE"])
        params["N"]=InterdepNet
    else:
        InterdepNet=params["N"]
    v_r=params["V"]
    # Initialize player result variables.
    player_strategies={}
    for P in layers:
        player_strategies[P]=INDPResults()
    num_iterations=params["NUM_ITERATIONS"]
    params_temp={}
    for key in params:
        params_temp[key]=params[key]
    params_temp["NUM_ITERATIONS"]=1
    for i in range(num_iterations):
        curr_player_ordering=player_ordering
        if player_ordering == "RANDOM":
            curr_player_ordering=random.sample(layers,len(layers))
        for P in curr_player_ordering:
            print("Iteration",i,", Player",P)
            #functionality=create_functionality_matrix(InterdepNet,1,[x for x in layers if x != P],strategy_type="REALISTIC")
            results=run_indp(params_temp,layers,controlled_layers=[P],T=1,save=False,suffix="P"+str(P)+"_i"+str(i),forced_actions=True)
            #print params["N"].G.node[(5,3)]['data']['inf_data'].functionality
            if i == 0:
                player_strategies[P]=results
            else:
                player_strategies[P].extend(results,t_offset=i+1,t_start=1,t_end=2)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for P in layers:
        player_strategies[P].to_csv(output_dir,params["SIM_NUMBER"],suffix="P"+str(P)+"_"+suffix)

def dynamic_parameters(N, original_N, t, dynamic_params):
    for n,d in N.G.nodes(data=True):
        data = dynamic_params[d['data']['inf_data'].net_id]
        if d['data']['inf_data'].demand<0:
            current_pop = data.loc[(data['node']==n[0])&(data['time']==t), 'current pop'].iloc[0]
            total_pop = data.loc[(data['node']==n[0])&(data['time']==t), 'total pop'].iloc[0]
            original_demand = original_N.G.nodes[n]['data']['inf_data'].demand
            d['data']['inf_data'].demand = original_demand*current_pop/total_pop

def save_INDP_model_to_file(model,outModelDir,t,l=0,suffix=''):
    if not os.path.exists(outModelDir):
        os.makedirs(outModelDir)
    # Write models to file
    lname = "/Model_t%d_l%d_%s.lp" % (t,l,suffix)
    model.write(outModelDir+lname)
    model.update()
     # Write solution to file
    sname = "/Solution_t%d_l%d_%s.txt" % (t,l,suffix)
    fileID = open(outModelDir+sname, 'w')
    for vv in model.getVars():
        fileID.write('%s %g\n' % (vv.varName, vv.x))
    fileID.write('Obj: %g' % model.objVal)
    fileID.close()

def initialize_sample_network(layers=[1,2]):
    """ Initializes sample network
    :param layers: (Currently not used).
    :returns: An interdependent InfrastructureNetwork.
    """
    InterdepNet=InfrastructureNetwork("sample_network")
    node_to_demand_dict={(1,1):5,(2,1):-1,(3,1):-2,(4,1):-2,(5,1):-4,(6,1):4,
                         (7,2):-2,(8,2):6,(9,2):1,(10,2):-5,(11,2):4,(12,2):-4}
    space_to_nodes_dict={1:[(1,1),(7,2)],2:[(2,1),(8,2)],
                    3:[(3,1),(5,1),(9,2),(11,2)],4:[(4,1),(6,1),(10,2),(12,2)]}
    arc_list= [((1,1),(2,1)),((1,1),(4,1)),((1,1),(3,1)),((6,1),(4,1)),((6,1),(5,1)),
               ((8,2),(7,2)),((8,2),(10,2)),((9,2),(7,2)),((9,2),(10,2)),((9,2),(12,2)),
               ((11,2),(12,2))]
    interdep_list=[((1,1),(7,2)),((2,1),(8,2)),((9,2),(3,1)),((4,1),(10,2))]
    failed_nodes=[(1,1),(2,1),(3,1),(5,1),(6,1),
                  (7,2),(8,2),(9,2),(11,2),(12,2)]
    if 3 in layers:
        node_to_demand_dict.update({(13,3):3,(14,3):6,(15,3):-5,(16,3):-6,
                                    (17,3):4,(18,3):-2})
        space_to_nodes_dict[1].extend([(13,3),(14,3),(15,3)])
        space_to_nodes_dict[2].extend([(16,3),(17,3),(18,3)])
        arc_list.extend([((13,3),(15,3)),((14,3),(15,3)),((14,3),(16,3)),
                         ((17,3),(15,3)),((17,3),(16,3)),((17,3),(18,3))])
        interdep_list.extend([((11,2),(17,3)),((9,2),(15,3)),((14,3),(8,2)),((14,3),(9,2))])
        failed_nodes.extend([(14,3),(15,3),(16,3),(17,3),(18,3)]) 
    global_index=1
    for n in node_to_demand_dict:
        nn=InfrastructureNode(global_index,n[1],n[0])
        nn.demand=node_to_demand_dict[n]
        nn.reconstruction_cost=abs(nn.demand)
        nn.oversupply_penalty=50
        nn.undersupply_penalty=50
        nn.resource_usage=1
        if n in failed_nodes:
            nn.functionality=0.0
            nn.repaired=0.0
        InterdepNet.G.add_node((nn.local_id,nn.net_id),data={'inf_data':nn})
        global_index+=1
    for s in space_to_nodes_dict:
        InterdepNet.S.append(InfrastructureSpace(s,0))
        for n in space_to_nodes_dict[s]:
            InterdepNet.G.nodes[n]['data']['inf_data'].space=s
    for a in arc_list:
        aa=InfrastructureArc(a[0][0],a[1][0],a[0][1])
        aa.flow_cost=1
        aa.capacity=50
        InterdepNet.G.add_edge((aa.source,aa.layer),(aa.dest,aa.layer),data={'inf_data':aa})
    for g in interdep_list:
        aa=InfrastructureInterdepArc(g[0][0],g[1][0],g[0][1],g[1][1],1.0)
        InterdepNet.G.add_edge((aa.source,aa.source_layer),(aa.dest,aa.dest_layer),data={'inf_data':aa})
    return InterdepNet

def plot_indp_sample(params,folderSuffix="",suffix=""):
    plt.figure(figsize=(16,8))
    if 3 in params["L"]:
        plt.figure(figsize=(16,10))
    InterdepNet=initialize_sample_network(layers=params["L"])
    pos=nx.spring_layout(InterdepNet.G)
    pos[(1,1)][0] =  0.5
    pos[(7,2)][0] =  0.5
    pos[(2,1)][0] =  0.0
    pos[(8,2)][0] =  0.0
    pos[(3,1)][0] =  2.0
    pos[(9,2)][0] =  2.0
    pos[(4,1)][0] =  1.5
    pos[(10,2)][0] =  1.5
    pos[(5,1)][0] =  3.0
    pos[(11,2)][0] =  3.0
    pos[(6,1)][0] =  2.5
    pos[(12,2)][0] =  2.5
    pos[(2,1)][1] =  0.0
    pos[(4,1)][1] =  0.0
    pos[(6,1)][1] =  0.0
    pos[(1,1)][1] =  1.0
    pos[(3,1)][1] =  1.0
    pos[(5,1)][1] =  1.0
    pos[(8,2)][1] =  2.0
    pos[(10,2)][1] =  2.0
    pos[(12,2)][1] =  2.0
    pos[(7,2)][1] =  3.0
    pos[(9,2)][1] =  3.0
    pos[(11,2)][1] =  3.0
    node_dict={1:[(1,1),(2,1),(3,1),(4,1),(5,1),(6,1)], 
               11:[(4,1)], #Undamaged
               12:[(1,1),(2,1),(3,1),(5,1),(6,1)], #Damaged
               2:[(7,2),(8,2),(9,2),(10,2),(11,2),(12,2)], 
               21:[(10,2)], 
               22:[(7,2),(8,2),(9,2),(11,2),(12,2)]}
    arc_dict= {1: [((1,1),(2,1)),((1,1),(3,1)),((1,1),(4,1)),((6,1),(4,1)),
                   ((6,1),(5,1))],
               2: [((8,2),(7,2)),((8,2),(10,2)),((9,2),(7,2)),((9,2),(10,2)),
                   ((9,2),(12,2)),((11,2),(12,2))]}
    if 3 in params["L"]:
         pos[(13,3)][0] =  0.5
         pos[(14,3)][0] =  0.0
         pos[(15,3)][0] =  2.0
         pos[(16,3)][0] =  1.5
         pos[(17,3)][0] =  3.0
         pos[(18,3)][0] =  2.5
         pos[(13,3)][1] =  5.0
         pos[(14,3)][1] =  4.0
         pos[(15,3)][1] =  5.0
         pos[(16,3)][1] =  4.0
         pos[(17,3)][1] =  5.0
         pos[(18,3)][1] =  4.0   
         node_dict[3] = [(13,3),(14,3),(15,3),(16,3),(17,3),(18,3)]
         node_dict[31] = [(13,3)]
         node_dict[32] = [(14,3),(15,3),(16,3),(17,3),(18,3)]
         arc_dict[3] = [((13,3),(15,3)),((14,3),(15,3)),((14,3),(16,3)),
                         ((17,3),(15,3)),((17,3),(16,3)),((17,3),(18,3))]

    labels = {}
    for n,d in InterdepNet.G.nodes(data=True):
        labels[n]= "%d[%d]" % (n[0],d['data']['inf_data'].demand)
    pos_moved={}
    for key,value in pos.items():
        pos_moved[key] = [0,0]
        pos_moved[key][0] = pos[key][0]-0.2
        pos_moved[key][1] = pos[key][1]+0.2

    v_r = params["V"]
    if isinstance(v_r, (int)):
        totalResource = v_r
    else:
        totalResource = sum([val for _, val in v_r.items()])

    output_dir=params["OUTPUT_DIR"]+'_L'+str(len(params["L"]))+'_m'+str(params["MAGNITUDE"])+"_v"+str(totalResource)+folderSuffix
    action_file =output_dir+"/actions_"+str(params["SIM_NUMBER"])+"_"+suffix+".csv"
    actions = {0:[]}
    if os.path.isfile(action_file):
        with open(action_file) as f:
            lines=f.readlines()[1:]
            for line in lines:
                data=line.split(",")
                t=int(data[0])
                action=str.strip(data[1])
                if t not in actions:
                    actions[t]=[]
                actions[t].append(action)

    T = max(actions.keys())
    for t, value in actions.items():
        plt.subplot(2, (T+1)/2+1 ,t+1, aspect='equal')
        plt.title('Time = %d' % t)
        for a in value:
            data=a.split(".")
            node_dict[int(data[1])*10+1].append((int(data[0]),int(data[1])))
            node_dict[int(data[1])*10+2].remove((int(data[0]),int(data[1])))
        nx.draw(InterdepNet.G, pos,node_color='w')
        nx.draw_networkx_labels(InterdepNet.G,labels=labels,pos=pos,
                                font_color='w',font_family='CMU Serif')#,font_weight='bold'
        nx.draw_networkx_nodes(InterdepNet.G,pos,nodelist=node_dict[1],node_color='#b51717',node_size=1100,alpha=0.7)
        nx.draw_networkx_nodes(InterdepNet.G,pos,nodelist=node_dict[2],node_color='#005f98',node_size=1100,alpha=0.7)
        nx.draw_networkx_nodes(InterdepNet.G,pos_moved,nodelist=node_dict[12],node_color='k',node_shape="X",node_size=150)
        nx.draw_networkx_nodes(InterdepNet.G,pos_moved,nodelist=node_dict[22],node_color='k',node_shape="X",node_size=150)
        nx.draw_networkx_edges(InterdepNet.G,pos,edgelist=arc_dict[1],width=1,alpha=0.9,edge_color='r')
        nx.draw_networkx_edges(InterdepNet.G,pos,edgelist=arc_dict[2],width=1,alpha=0.9,edge_color='b')
        if 3 in params["L"]:
            nx.draw_networkx_nodes(InterdepNet.G,pos,nodelist=node_dict[3],node_color='#009181',node_size=1100,alpha=0.7)
            nx.draw_networkx_nodes(InterdepNet.G,pos_moved,nodelist=node_dict[32],node_color='k',node_shape="X",node_size=150)
            nx.draw_networkx_edges(InterdepNet.G,pos,edgelist=arc_dict[3],width=1,alpha=0.9,edge_color='g')
    plt.tight_layout()
    plt.savefig(output_dir+'/plot_net'+suffix+'.png',dpi=300)