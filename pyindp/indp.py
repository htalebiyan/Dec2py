from infrastructure import *
from indputils import *
from gurobipy import *
import string
import platform
import networkx as nx
import copy
import random
import sys

#HOME_DIR="/Users/Andrew/"
#if platform.system() == "Linux":
#    HOME_DIR="/home/andrew/"

def indp(N,v_r,T=1,layers=[1,3],controlled_layers=[1,3],functionality={},forced_actions=False):
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
    #                print "At Time",`t`,", node",u,"is broken."
    m=Model('indp')
    m.setParam('OutputFlag',False)
    G_prime_nodes = [n[0] for n in N.G.nodes_iter(data=True) if n[1]['data']['inf_data'].net_id in layers]
    G_prime = N.G.subgraph(G_prime_nodes)
    # Damaged nodes in whole network
    N_prime = [n for n in G_prime.nodes_iter(data=True) if n[1]['data']['inf_data'].functionality==0.0]
    # Nodes in controlled network.
    N_hat_nodes   = [n[0] for n in G_prime.nodes_iter(data=True) if n[1]['data']['inf_data'].net_id in controlled_layers]
    N_hat = G_prime.subgraph(N_hat_nodes)
    # Damaged nodes in controlled network.
    N_hat_prime= [n for n in N_hat.nodes_iter(data=True) if n[1]['data']['inf_data'].functionality==0.0]
    # Damaged arcs in whole network
    A_prime = [(u,v,a) for u,v,a in G_prime.edges_iter(data=True) if a['data']['inf_data'].functionality==0.0]
    # Damaged arcs in controlled network.
    A_hat_prime = [(u,v,a) for u,v,a in A_prime if N_hat.has_node(u) and N_hat.has_node(v)]
    #print "A_hat_prime=",A_hat_prime
    S=N.S
    #print ""
    #print "New sim."
    # Populate interdepencies. Add nodes to N' if they currently rely on a non-functional node.
    interdep_nodes={}
    for u,v,a in G_prime.edges_iter(data=True):
        if not functionality:
            if a['data']['inf_data'].is_interdep and G_prime.node[u]['data']['inf_data'].functionality == 0.0:
                #print "Dependency edge goes from:",u,"to",v
                if v not in interdep_nodes:
                    interdep_nodes[v]=[]
                interdep_nodes[v].append((u,a['data']['inf_data'].gamma))
        else:
            # Should populate N_hat with layers that are controlled. Then go through N_hat.edges_iter(data=True)
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
        for n,d in N_hat.nodes_iter(data=True):
            m.addVar(name='delta+_'+str(n)+","+str(t),lb=0.0)
            m.addVar(name='delta-_'+str(n)+","+str(t),lb=0.0)
        # Add functionality binary variables for each node in N'.
        for n,d in N_hat.nodes_iter(data=True):
            m.addVar(name='w_'+str(n)+","+str(t),vtype=GRB.BINARY)
            if T > 1:
                m.addVar(name='w_tilde_'+str(n)+","+str(t),vtype=GRB.BINARY)
        # Add flow variables for each arc.
        for u,v,a in N_hat.edges_iter(data=True):
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
        for n,d in N_hat.nodes_iter(data=True):
            objFunc+=d['data']['inf_data'].oversupply_penalty*m.getVarByName('delta+_'+str(n)+","+str(t))
            objFunc+=d['data']['inf_data'].undersupply_penalty*m.getVarByName('delta-_'+str(n)+","+str(t))
        for u,v,a in N_hat.edges_iter(data=True):
            objFunc+=a['data']['inf_data'].flow_cost*m.getVarByName('x_'+str(u)+","+str(v)+","+str(t))
            
            
    m.setObjective(objFunc,GRB.MINIMIZE)
    m.update()
    
    #Constraints.
    # Time-dependent constraints.
    if T > 1:
        for n,d in N_hat.nodes_iter(data=True):
            m.addConstr(m.getVarByName('w_'+str(n)+",0"),GRB.EQUAL,0)
        for u,v,a in A_hat_prime:
            m.addConstr(m.getVarByName('y_'+str(u)+","+str(v)+",0"),GRB.EQUAL,0)
        
        
    for t in range(T):
        # Time-dependent constraint.
        for n,d in N_hat.nodes_iter(data=True):
            if t > 0:
                wTildeSum=LinExpr()
                for t_prime in range(1,t):
                    wTildeSum+=m.getVarByName('w_tilde_'+str(n)+","+str(t_prime))
                m.addConstr(m.getVarByName('w_'+str(n)+","+str(t)),GRB.LESS_EQUAL,wTildeSum,"Time dependent recovery constraint at node "+str(n)+","+str(t))
        for u,v,a in A_hat_prime:
            if t > 0:
                yTildeSum=LinExpr()
                for t_prime in range(1,t):
                    yTildeSum+=m.getVarByName('y_tilde_'+`u`+","+`v`+","+`t_prime`)
                m.addConstr(m.getVarByName('y_'+`u`+","+`v`+","+`t`),GRB.LESS_EQUAL,yTildeSum,"Time dependent recovery constraint at arc "+`u`+","+`v`+","+`t`)
        # Enforce a_i,j to be fixed if a_j,i is fixed (and vice versa). 
        for u,v,a in A_hat_prime:
            #print u,",",v
            m.addConstr(m.getVarByName('y_'+`u`+","+`v`+","+`t`),GRB.EQUAL,m.getVarByName('y_'+`v`+","+`u`+","+`t`),"Arc reconstruction equality ("+`u`+","+`v`+","+`t`+")")
            if T > 1:
                m.addConstr(m.getVarByName('y_tilde_'+`u`+","+`v`+","+`t`),GRB.EQUAL,m.getVarByName('y_tilde_'+`v`+","+`u`+","+`t`),"Arc reconstruction equality ("+`u`+","+`v`+","+`t`+")")
        # Conservation of flow constraint. (2) in INDP paper.
        for n,d in N_hat.nodes_iter(data=True):
            outFlowConstr=LinExpr()
            inFlowConstr= LinExpr()
            demandConstr= LinExpr()
            for u,v,a in N_hat.out_edges(n,data=True):
                outFlowConstr+=m.getVarByName('x_'+`u`+","+`v`+","+`t`)
            for u,v,a in N_hat.in_edges(n,data=True):
                inFlowConstr+= m.getVarByName('x_'+`u`+","+`v`+","+`t`)
            demandConstr+=d['data']['inf_data'].demand - m.getVarByName('delta+_'+`n`+","+`t`) + m.getVarByName('delta-_'+`n`+","+`t`)
            m.addConstr(outFlowConstr-inFlowConstr,GRB.EQUAL,demandConstr,"Flow conservation constraint "+`n`+","+`t`)
        # Flow functionality constraints.
        for u,v,a in N_hat.edges_iter(data=True):
            if u in [n for (n,d) in N_hat_prime]:
                m.addConstr(m.getVarByName('x_'+`u`+","+`v`+","+`t`),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*m.getVarByName('w_'+`u`+","+`t`),"Flow in functionality constraint("+`u`+","+`v`+","+`t`+")")
            else:
                m.addConstr(m.getVarByName('x_'+`u`+","+`v`+","+`t`),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*N.G.node[u]['data']['inf_data'].functionality,"Flow in functionality constraint ("+`u`+","+`v`+","+`t`+")")
            if v in [n for (n,d) in N_hat_prime]:
                m.addConstr(m.getVarByName('x_'+`u`+","+`v`+","+`t`),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*m.getVarByName('w_'+`v`+","+`t`),"Flow out functionality constraint("+`u`+","+`v`+","+`t`+")")
            else:
                m.addConstr(m.getVarByName('x_'+`u`+","+`v`+","+`t`),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*N.G.node[v]['data']['inf_data'].functionality,"Flow out functionality constraint ("+`u`+","+`v`+","+`t`+")")
            if (u,v,a) in A_hat_prime:
                m.addConstr(m.getVarByName('x_'+`u`+","+`v`+","+`t`),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*m.getVarByName('y_'+`u`+","+`v`+","+`t`),"Flow arc functionality constraint ("+`u`+","+`v`+","+`t`+")")
            else:
                m.addConstr(m.getVarByName('x_'+`u`+","+`v`+","+`t`),GRB.LESS_EQUAL,a['data']['inf_data'].capacity*N.G[u][v]['data']['inf_data'].functionality,"Flow arc functionality constraint("+`u`+","+`v`+","+`t`+")")

        #Resource availability constraints.
        isSepResource = 0
        if isinstance(v_r, (int, long)):
            totalResource = v_r
        else:
            if len(v_r) != 1:
                isSepResource = 1
                totalResource = sum(v_r)
                if len(v_r) != len(layers):
                    print "\n***ERROR: The number of resource cap values does not match the number of layers.***\n"
                    sys.exit()
            else:
                totalResource = v_r[0]
            
        resourceLeftConstr=LinExpr()
        if isSepResource:
            resourceLeftConstrSep = [LinExpr() for i in range(len(v_r))]
                                     
        for u,v,a in A_hat_prime:
            indexLayer = a['data']['inf_data'].layer - 1
            if T == 1:
                resourceLeftConstr+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_'+`u`+","+`v`+","+`t`)
                if isSepResource:
                    resourceLeftConstrSep[indexLayer]+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_'+`u`+","+`v`+","+`t`)
            else:
                resourceLeftConstr+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_tilde_'+`u`+","+`v`+","+`t`)
                if isSepResource:
                    resourceLeftConstrSep[indexLayer]+=0.5*a['data']['inf_data'].resource_usage*m.getVarByName('y_tilde_'+`u`+","+`v`+","+`t`)

        for n,d in N_hat_prime:
            indexLayer = n[1] - 1
            if T == 1:
                resourceLeftConstr+=d['data']['inf_data'].resource_usage*m.getVarByName('w_'+`n`+","+`t`)
                if isSepResource:
                    resourceLeftConstrSep[indexLayer]+=d['data']['inf_data'].resource_usage*m.getVarByName('w_'+`n`+","+`t`)
            else:
                resourceLeftConstr+=d['data']['inf_data'].resource_usage*m.getVarByName('w_tilde_'+`n`+","+`t`)
                if isSepResource:
                    resourceLeftConstrSep[indexLayer]+=d['data']['inf_data'].resource_usage*m.getVarByName('w_tilde_'+`n`+","+`t`)

        m.addConstr(resourceLeftConstr,GRB.LESS_EQUAL,totalResource,"Resource availability constraint at "+`t`+".")
        if isSepResource:
            for k in range(len(v_r)):
                m.addConstr(resourceLeftConstrSep[k],GRB.LESS_EQUAL,v_r[k],"Resource availability constraint at "+`t`+ " for layer "+`k`+".")

        # Interdependency constraints
        infeasible_actions=[]
        for n,d in N_hat.nodes_iter(data=True):
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
                            interdepLConstr+=m.getVarByName('w_'+`src`+","+`t`)*gamma
                    interdepRConstr+=m.getVarByName('w_'+`n`+","+`t`)
                    m.addConstr(interdepLConstr,GRB.GREATER_EQUAL,interdepRConstr,"Interdependency constraint for node "+`n`+","+`t`)
            else:
                if n in interdep_nodes[t]:
                    #print interdep_nodes[t]
                    interdepLConstr=LinExpr()
                    interdepRConstr=LinExpr()
                    for interdep in interdep_nodes[t][n]:
                        src=interdep[0]
                        gamma=interdep[1]
                        if not N_hat.has_node(src):
                            print "Forcing",`n`,"to be 0 (dep. on",`src`,")"
                            infeasible_actions.append(n)
                            interdepLConstr+=0
                        else:
                            interdepLConstr+=m.getVarByName('w_'+`src`+","+`t`)*gamma
                    interdepRConstr+=m.getVarByName('w_'+`n`+","+`t`)
                    m.addConstr(interdepLConstr,GRB.GREATER_EQUAL,interdepRConstr,"Interdependency constraint for node "+`n`+","+`t`)

        # Forced actions (if applicable)
        if forced_actions:
            recovery_sum=LinExpr()
            feasible_nodes=[(n,d) for n,d in N_hat_prime if n not in infeasible_actions] 
            if len(feasible_nodes) + len(A_hat_prime) > 0:
                for n,d in feasible_nodes:
                    if T == 1:
                        recovery_sum+=m.getVarByName('w_'+`n`+","+`t`)
                    else:
                        recovery_sum+=m.getVarByName('w_tilde_'+`n`+","+`t`)
                for u,v,a in A_hat_prime:
                    if T == 1:
                        recovery_sum+=m.getVarByName('y_'+`u`+","+`v`+","+`t`)
                    else:
                        recovery_sum+=m.getVarByName('y_tilde_'+`u`+","+`v`+","+`t`)
                m.addConstr(recovery_sum,GRB.GREATER_EQUAL,1,"Forced action constraint")

        # Geographic space constraints
        for s in S:
            for n,d in N_hat_prime:
                if T == 1:
                    m.addConstr(m.getVarByName('w_'+`n`+","+`t`)*d['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+`s.id`+","+`t`),"Geographical space constraint for node "+`n`+","+`t`)
                else:
                    m.addConstr(m.getVarByName('w_tilde_'+`n`+","+`t`)*d['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+`s.id`+","+`t`),"Geographical space constraint for node "+`n`+","+`t`)
            for u,v,a in A_hat_prime:
                if T== 1:
                    m.addConstr(m.getVarByName('y_'+`u`+","+`v`+","+`t`)*a['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+`s.id`+","+`t`),"Geographical space constraint for arc ("+`u`+","+`v`+")")
                else:
                    m.addConstr(m.getVarByName('y_tilde_'+`u`+","+`v`+","+`t`)*a['data']['inf_data'].in_space(s.id),GRB.LESS_EQUAL,m.getVarByName('z_'+`s.id`+","+`t`),"Geographical space constraint for arc ("+`u`+","+`v`+")")
      
#    print "Solving..."
    m.update()
    m.optimize()
    indp_results=INDPResults()
    # Save results.
    if m.getAttr("Status")==GRB.OPTIMAL:
        for t in range(T):
            nodeCost=0.0
            arcCost=0.0
            flowCost=0.0
            overSuppCost=0.0
            underSuppCost=0.0
            spacePrepCost=0.0
            # Record node recovery actions.
            for n,d in N_hat_prime:
                nodeVar='w_tilde_'+`n`+","+`t`
                if T == 1:
                    nodeVar='w_'+`n`+","+`t`
                if round(m.getVarByName(nodeVar).x)==1:
                    action=`n[0]`+"."+`n[1]`
                    indp_results.add_action(t,action)
                    #if T == 1:
                    #N.G.node[n]['data']['inf_data'].functionality=1.0
            # Record edge recovery actions.
            for u,v,a in A_hat_prime:
                arcVar='y_tilde_'+`u`+","+`v`+","+`t`
                if T == 1:
                    arcVar='y_'+`u`+","+`v`+","+`t`
                if round(m.getVarByName(arcVar).x)==1:
                    action=`u[0]`+"."+`u[1]`+"/"+`v[0]`+"."+`v[1]`
                    indp_results.add_action(t,action)
                    #if T == 1:
                    #N.G[u][v]['data']['inf_data'].functionality=1.0
            # Calculate space preparation costs.
            for s in S:
                spacePrepCost+=s.cost*m.getVarByName('z_'+`s.id`+","+`t`).x
            indp_results.add_cost(t,"Space Prep",spacePrepCost)
            # Calculate arc preparation costs.
            for u,v,a in A_hat_prime:
                arcVar='y_tilde_'+`u`+","+`v`+","+`t`
                if T == 1:
                    arcVar='y_'+`u`+","+`v`+","+`t`
                arcCost+=(a['data']['inf_data'].reconstruction_cost/2.0)*m.getVarByName(arcVar).x
            indp_results.add_cost(t,"Arc",arcCost)
            # Calculate node preparation costs.
            for n,d in N_hat_prime:
                nodeVar = 'w_tilde_'+`n`+","+`t`
                if T == 1:
                    nodeVar = 'w_'+`n`+","+`t`
                nodeCost+=d['data']['inf_data'].reconstruction_cost*m.getVarByName(nodeVar).x
            indp_results.add_cost(t,"Node",nodeCost)

            # Calculate under/oversupply costs.
            for n,d in N_hat.nodes_iter(data=True):
                overSuppCost+= d['data']['inf_data'].oversupply_penalty*m.getVarByName('delta+_'+`n`+","+`t`).x
                underSuppCost+=d['data']['inf_data'].undersupply_penalty*m.getVarByName('delta-_'+`n`+","+`t`).x
            indp_results.add_cost(t,"Over Supply",overSuppCost)
            indp_results.add_cost(t,"Under Supply",underSuppCost)
            # Calculate flow costs.
            for u,v,a in N_hat.edges_iter(data=True):
                flowCost+=a['data']['inf_data'].flow_cost*m.getVarByName('x_'+`u`+","+`v`+","+`t`).x
            indp_results.add_cost(t,"Flow",flowCost)
            # Calculate total costs.
            indp_results.add_cost(t,"Total",flowCost+arcCost+nodeCost+overSuppCost+underSuppCost+spacePrepCost)
            indp_results.add_cost(t,"Total no disconnection",spacePrepCost+arcCost+flowCost+nodeCost)
	                
        return [m,indp_results]
    else:
        print m.getAttr("Status"),": SOLUTION NOT FOUND. (Check data and/or violated constraints)."
        m.computeIIS()
        print ('\nThe following constraint(s) cannot be satisfied:')
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
        return None

def apply_recovery(N,indp_results,t):
    for action in indp_results[t]['actions']:
        if "/" in action:
            # Edge recovery action.
            data=string.split(action,"/")
            src=tuple([int(x) for x in string.split(data[0],".")])
            dst=tuple([int(x) for x in string.split(data[1],".")])
            N.G[src][dst]['data']['inf_data'].functionality=1.0
        else:
            # Node recovery action.
            node=tuple([int(x) for x in string.split(action,".")])
            #print "Applying recovery:",node
            N.G.node[node]['data']['inf_data'].functionality=1.0

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
    G_prime_nodes = [n[0] for n in N.G.nodes_iter(data=True) if n[1]['data']['inf_data'].net_id in layers]
    G_prime = N.G.subgraph(G_prime_nodes)
    N_prime = [n for n in G_prime.nodes_iter(data=True) if n[1]['data']['inf_data'].functionality==0.0]
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
        for n,d in G_prime.nodes_iter(data=True):
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
            

def initialize_network(BASE_DIR="../data/INDP_7-20-2015/",external_interdependency_dir=None,sim_number=1,cost_scale=0.1,magnitude=6,v=3):
    """ Initializes an InfrastructureNetwork from Shelby County data.
    :param BASE_DIR: Base directory of Shelby County data.
    :param sim_number: Which simulation number to use as input.
    :param cost_scale: Scales the cost to improve efficiency.
    :param magnitude: Magnitude of earthquake.
    :param v: v parameter
    :returns: An interdependent InfrastructureNetwork.
    """
    print "Loading Shelby County data..."
    InterdepNet=load_infrastructure_data(BASE_DIR=BASE_DIR,external_interdependency_dir=external_interdependency_dir,sim_number=sim_number,cost_scale=1.0,magnitude=magnitude,v=v)
    print "Data loaded."
    return InterdepNet


def initialize_sample_network(params,layers=[1,2]):
    """ Initializes sample 2x8 network (used in INRG examples)
    :param params: (Currently not used).
    :param layers: (Currently not used).
    :returns: An interdependent InfrastructureNetwork.
    """
    InterdepNet=InfrastructureNetwork("2x8_centralized")
    node_to_demand_dict={(1,1):5,(2,1):-1,(3,1):-2,(4,1):-2,(5,2):-2,(6,2):6,(7,2):1,(8,2):-5}
    space_to_nodes_dict={1:[(1,1),(5,2)],2:[(2,1),(6,2)],3:[(3,1),(7,2)],4:[(4,1),(8,2)]}
    arc_list= [((1,1),(2,1)),((1,1),(4,1)),((1,1),(3,1)),((6,2),(5,2)),((6,2),(8,2)),((7,2),(5,2)),((7,2),(8,2))]
    interdep_list=[((1,1),(5,2)),((2,1),(6,2)),((3,1),(7,2)),((4,1),(8,2))]
    failed_nodes=[(1,1),(2,1),(3,1),(5,2),(6,2),(7,2)]
    global_index=1
    for n in node_to_demand_dict:
        nn=InfrastructureNode(global_index,n[1],n[0])
        nn.demand=node_to_demand_dict[n]
        nn.reconstruction_cost=abs(nn.demand)
        nn.oversupply_penalty=0.0
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
            InterdepNet.G.node[n]['data']['inf_data'].space=s
    for a in arc_list:
        aa=InfrastructureArc(a[0][0],a[1][0],a[0][1])
        aa.flow_cost=1
        aa.capacity=50
        InterdepNet.G.add_edge((aa.source,aa.layer),(aa.dest,aa.layer),data={'inf_data':aa})
    for g in interdep_list:
        aa=InfrastructureInterdepArc(g[0][0],g[1][0],g[0][1],g[1][1],1.0)
        InterdepNet.G.add_edge((aa.source,aa.source_layer),(aa.dest,aa.dest_layer),data={'inf_data':aa})
    return InterdepNet
        
def run_sample(params):
    """ Runs the sample network generated in initialize_sample_network through indp.
    :param params: Global parameters.
    """
    N=initialize_sample_network(params)
    params["N"]=N
    params["V"]=2
    run_indp(params)

def run_indp(params,layers=[1,2,3],controlled_layers=[],functionality={},T=1,validate=False,save=True,suffix="",forced_actions=False,saveModel=False,print_cmd_line=True):
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
        params["NUM_ITERATIONS"]=1
    if not controlled_layers:
        controlled_layers = layers
        
    v_r=params["V"]
    if isinstance(v_r, (int, long)):
        outDirSuffixRes = `v_r`
    elif len(v_r)==1:
        outDirSuffixRes = `v_r[0]`
    else:
        outDirSuffixRes = `sum(v_r)`+'_Layer_Res_Cap'
            
    indp_results=INDPResults()
    if T == 1:
        if print_cmd_line:
            print "Running INDP (T=1) or iterative INDP."
            print "Num iters=",params["NUM_ITERATIONS"]
            
        # Run INDP for 1 time step (original INDP).
        output_dir=params["OUTPUT_DIR"]+'_m'+`params["MAGNITUDE"]`+"_v"+outDirSuffixRes
        # Initial calculations.
        results=indp(InterdepNet,0,1,layers,controlled_layers=controlled_layers)
        indp_results=results[1]
        indp_results.add_components(0,INDPComponents.calculate_components(results[0],InterdepNet,layers=controlled_layers))

        for i in range(params["NUM_ITERATIONS"]):
            if print_cmd_line:
                print "Time Step (iINDP)=",i,"/",params["NUM_ITERATIONS"]
            results=indp(InterdepNet,v_r,T,layers,controlled_layers=controlled_layers,forced_actions=forced_actions)
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
        output_dir=params["OUTPUT_DIR"]+"_m"+`params["MAGNITUDE"]`+"_v"+outDirSuffixRes
        if print_cmd_line:
            print "Running td-INDP (T="+`T`+", Window size="+`time_window_length`+")"
        # Initial percolation calculations.
        results=indp(InterdepNet,0,1,layers,controlled_layers=controlled_layers,functionality=functionality)
        indp_results=results[1]
        indp_results.add_components(0,INDPComponents.calculate_components(results[0],InterdepNet,layers=controlled_layers))
        for n in range(num_time_windows):
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
            results=indp(InterdepNet,v_r,time_window_length+1,layers,controlled_layers=controlled_layers,functionality=functionality_t,forced_actions=forced_actions)
            if saveModel:
                save_INDP_model_to_file(results[0],output_dir+"/Model",n+1)
            if "WINDOW_LENGTH" in params:
                indp_results.extend(results[1],t_offset=n+1,t_start=1,t_end=2)
                # Modify network for recovery actions and calculate components.
                apply_recovery(InterdepNet,results[1],1)
                indp_results.add_components(n+1,INDPComponents.calculate_components(results[0],InterdepNet,1,layers=controlled_layers))
            else:
                indp_results.extend(results[1],t_offset=1)
                for t in range(1,T):
                    # Modify network to account for recovery actions.
                    apply_recovery(InterdepNet,indp_results,t)
                    indp_results.add_components(1,INDPComponents.calculate_components(results[0],InterdepNet,t,layers=controlled_layers))
    # Save results of INDP run.
    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        indp_results.to_csv(output_dir,params["SIM_NUMBER"],suffix=suffix)
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
            print "P=",`P`,"i=",`i`
            if i == 0:
                # Create functionality matrix. This corresponds to "OPTIMISTIC" or "PESSIMISTIC" in Sharkey paper.
                # OPTIMISTIC implies that interdependencies are assumed to be fixed whenever a player needs them to be.
                # PESSIMISTIC assumes interdependencies never become fixed.
                functionality=create_functionality_matrix(InterdepNet,T,negP,actions=None,strategy_type="OPTIMISTIC")
            else:
                print "Next iteration!"
                actions=[]
                for t in range(T):
                    for l in negP:
                        actions.append(player_strategies[l][t]["actions"])
                functionality=create_functionality_matrix(InterdepNet,T,negP,actions=actions,strategy_type="INFO_SHARE")
            params["N"]=InterdepNet.copy()
            results[P]=run_indp(params,layers,controlled_layers=[P],T=T,functionality=functionality,save=True,suffix="P"+`P`+"_i"+`i`+"_"+suffix,forced_actions=True)
        for P in layers:
            player_strategies[P]=results[P]

def run_inrg(params,layers=[1,2,3],validate=False,player_ordering=[3,1],suffix=""):
    InterdepNet=None
    output_dir=params["OUTPUT_DIR"]+"_m"+`params["MAGNITUDE"]`+"_v"+`params["V"]`
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
            print "Iteration",i,", Player",P
            #functionality=create_functionality_matrix(InterdepNet,1,[x for x in layers if x != P],strategy_type="REALISTIC")
            results=run_indp(params_temp,layers,controlled_layers=[P],T=1,save=False,suffix="P"+`P`+"_i"+`i`,forced_actions=True)
            #print params["N"].G.node[(5,3)]['data']['inf_data'].functionality
            if i == 0:
                player_strategies[P]=results
            else:
                player_strategies[P].extend(results,t_offset=i+1,t_start=1,t_end=2)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for P in layers:
        player_strategies[P].to_csv(output_dir,params["SIM_NUMBER"],suffix="P"+`P`+"_"+suffix)
    

def run_seqeq(params,layers=[1,2,3],validate=False,player_ordering=[3,1],suffix=""):
    InterdepNet=None
    output_dir=params["OUTPUT_DIR"]+"_m"+`params["MAGNITUDE"]`+"_v"+`params["V"]`
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=params['SIM_NUMBER'],magnitude=params["MAGNITUDE"])
        params["N"]=InterdepNet
    else:
        InterdepNet=params["N"]
    v_r=params["V"]
    
                            
    
def baseline_metrics(BASE_DIR="/Users/Andrew/Dropbox/iINDP",layers=[1,2,3]):
    """ Calculate and print baseline metrics of undamaged network.
    :param BASE_DIR: Base directory for INDP/Shelby County data.
    :param layers: Specifies which layers to calculate metrics for.
    """
    N=initialize_network(BASE_DIR,sim_number=0)
    G=N.G
    costs_disconnection=0.0
    costs_no_disconnection=0.0
    opt_flow_cost=0.0
    for n,d in G.nodes_iter(data=True):
        if d['data']['inf_data'].net_id in layers:
            demand=d['data']['inf_data'].demand
            if demand > 0.0:
                costs_disconnection+=d['data']['inf_data'].oversupply_penalty*demand
            else:
                costs_disconnection+=d['data']['inf_data'].undersupply_penalty*(-demand)
            costs_disconnection+=d['data']['inf_data'].reconstruction_cost
            costs_no_disconnection+=d['data']['inf_data'].reconstruction_cost
    for u,v,a in G.edges_iter(data=True):
        if u[1] in layers and v[1] in layers:
            costs_disconnection+=(a['data']['inf_data'].reconstruction_cost/2.0)
            costs_no_disconnection+=(a['data']['inf_data'].reconstruction_cost/2.0)
    for s in N.S:
        costs_disconnection+=s.cost
        costs_no_disconnection+=s.cost
    results=indp(N,1)
    opt_flow_cost=results[1]["Flow"]
    print "Costs with disconnection:",costs_disconnection
    print "Costs w/o disconnection: ",costs_no_disconnection
    print "Nominal flow cost:       ",opt_flow_cost
#baseline_metrics(layers=[1,3],BASE_DIR=HOME_DIR+"/Dropbox/iINDP")

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