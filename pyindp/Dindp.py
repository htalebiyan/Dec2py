import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from indp import *
import os.path
import operator
import networkx as nx
from infrastructure import *
from indputils import *
import copy
from gurobipy import *
import itertools 
import scipy
def run_judgment_call(params,layers,T=1,saveJC=True,print_cmd=True,saveJCModel=False,validate=False):
    """ Solves an INDP problem with specified parameters using a decentralized hueristic called Judgment Call . Outputs to directory specified in params['OUTPUT_DIR'].
    :param params: Global parameters.
    :param layers: Layers to consider in the infrastructure network.
    :param T: Number of time steps per analyses (1 for D-iINDP and T>1 for D-tdINDP)
    :param saveJC: If true, the results are saved to files
    :param print_cmd: If true, the results are printed to console
    :param saveJCModel: If true, optimization models and their solutions are printed to file
    :param validate: (Currently not used.)
    """
    
    judgment_type = params["JUDGMENT_TYPE"]
    auction_type = params["AUCTION_TYPE"]
    valuation_type = params["VALUATION_TYPE"]
    
    # Initialize failure scenario.
    InterdepNet=None
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=params['SIM_NUMBER'],magnitude=params["MAGNITUDE"])
    else:
        InterdepNet=params["N"]
        
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"]=1
 
    v_r = params["V"]
    num_iterations = params["NUM_ITERATIONS"]
    if auction_type:
        output_dir = params["OUTPUT_DIR"]+'_m'+`params["MAGNITUDE"]`+"_v"+`sum(v_r)`+'_auction_'+auction_type+'_'+valuation_type
    else:
        output_dir = params["OUTPUT_DIR"]+'_m'+`params["MAGNITUDE"]`+"_v"+`sum(v_r)`+'_fixed_layer_cap'
    
    Dindp_results={P:INDPResults() for P in layers}   
    Dindp_results_Real={P:INDPResults() for P in layers} 
    currentTotalCost = {}
    if T == 1: 
        if auction_type:
            print "***\nRunning Judgment Call with type "+judgment_type +" with auction "+auction_type+ ' & valuation '+ valuation_type
        else:
            print "***\nRunning Judgment Call with type "+judgment_type 
        if print_cmd:
            print "Num iters=",params["NUM_ITERATIONS"]
        # Initial calculations.
        indp_results_initial=indp(InterdepNet,0,1,layers,controlled_layers=layers)
        # Initial costs are just saved to cost_#_sum.csv not to cost_#_P.csv (to be corrected)
        # Initial components are not saved (to be corrected)
        # Components of all layers are not saved to components_#_sum.csv(to be corrected)
        # Actions of all layers are not saved to actions_#_sum.csv(to be corrected)
        # Percolation of all layers are not saved to percolation_#_sum.csv(to be corrected)
        for P in layers:
            Dindp_results[P].add_cost(0,'Total',0.0) #Add a zero entry for t=0 for each layer
            Dindp_results_Real[P].add_cost(0,'Total',0.0) #Add a zero entry for t=0 for each layer
        
        res_allocate = {}
        PoA = {}
        valuations={}
        for i in range(num_iterations):
            print "\n--Iteration "+`i`+"/"+`num_iterations-1`
            
            v_r_applied = []
            if auction_type:
                res_allocate[i],PoA[i],valuations[i]=auction_resources(sum(v_r),params,
                    layers=layers,T=1,print_cmd=print_cmd,judgment_type=judgment_type,
                    auction_type=auction_type,valuation_type=valuation_type)
               
                for key, value in res_allocate[i].items():
                    v_r_applied.append(len(value))
            else:
                v_r_applied = v_r
             
            functionality = {p:{} for p in layers}
            uncorrectedResults = {}  
            if print_cmd:
                print "Judgment: "                
            for P in layers:
                if print_cmd:
                    print "Layer-%d"%(P)
                    
                negP=[x for x in layers if x != P]    
                functionality[P] = create_judgment_matrix(InterdepNet,T,negP,v_r_applied,
                                        actions=None,judgment_type=judgment_type) 
                
                # Make decision based on judgments before communication
                indp_results = indp(InterdepNet,v_r_applied[P-1],1,layers=layers,
                                controlled_layers=[P],functionality= functionality[P],
                                print_cmd=print_cmd)
                        
                # Save models for re-evaluation after communication
                uncorrectedResults[P] = indp_results[1]
                
                # Save results of decisions based on judgments 
                Dindp_results[P].extend(indp_results[1],t_offset=i+1)
                # Save models to file
                if saveJCModel:
                    save_INDP_model_to_file(indp_results[0],output_dir+"/Model",i+1,P)

                # Modify network to account for recovery and calculate components.
                apply_recovery(InterdepNet,Dindp_results[P],i+1)
                Dindp_results[P].add_components(i+1,INDPComponents.calculate_components(indp_results[0],InterdepNet,layers=[P]))
            
            # Re-evaluate judgments based on other agents' decisions
            if print_cmd:
                print "Re-evaluation: "
            for P in layers:
                if print_cmd:
                    print "Layer-%d"%(P)  
                                     
                indp_results_Real,realizations = Decentralized_INDP_Realized_Performance(InterdepNet,i+1,
                                uncorrectedResults[P],functionality= functionality[P],
                                T=1,layers=layers,controlled_layers=[P],
                                print_cmd=print_cmd,saveJCModel=saveJCModel)  
                             
                Dindp_results_Real[P].extend(indp_results_Real[1],t_offset=i+1)
                
                if saveJCModel:
                    save_INDP_model_to_file(indp_results_Real[0],output_dir+"/Model",i+1,P,suffix='Real')  
                    output_dir_judgments= output_dir + '/judgments'
                    write_judgments_csv(InterdepNet,output_dir_judgments,functionality[P],realizations,
                                        sample_num=params["SIM_NUMBER"],
                                        agent=P,time=i+1,suffix="")       
        # Calculate sum of costs    
        Dindp_results_sum = INDPResults()
        Dindp_results_Real_sum = INDPResults()
        cost_types = Dindp_results[1][0]['costs'].keys()
        for i in range(num_iterations+1):                
            for cost_type in cost_types:
                sumTemp = 0.0
                sumTemp_Real = 0.0
                if i==0:
                    sumTemp = indp_results_initial[1][0]['costs'][cost_type]
                    sumTemp_Real = indp_results_initial[1][0]['costs'][cost_type]
                else:
                    for P in layers:
                        sumTemp += Dindp_results[P][i]['costs'][cost_type]
                        sumTemp_Real += Dindp_results_Real[P][i]['costs'][cost_type]
                Dindp_results_sum.add_cost(i,cost_type,sumTemp)
                Dindp_results_Real_sum.add_cost(i,cost_type,sumTemp_Real)
            
            for P in layers:
                for a in Dindp_results[P][i]['actions']:
                    Dindp_results_sum.add_action(i,a) 
                
        if auction_type:
            output_dir_auction = output_dir + '/auctions'
            write_auction_csv(output_dir_auction,res_allocate,PoA,valuations,sample_num=params["SIM_NUMBER"],suffix="")    
        # Save results of D-iINDP run to file.
        if saveJC:   
            output_dir_agents = output_dir + '/agents'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(output_dir_agents):
                os.makedirs(output_dir_agents)
            for P in layers:
                Dindp_results[P].to_csv(output_dir_agents,params["SIM_NUMBER"],suffix=`P`)
                Dindp_results_Real[P].to_csv(output_dir_agents,params["SIM_NUMBER"],suffix='Real_'+`P`)
            Dindp_results_sum.to_csv(output_dir,params["SIM_NUMBER"],suffix='sum')
            Dindp_results_Real_sum.to_csv(output_dir,params["SIM_NUMBER"],suffix='Real_sum')
    else:
        # td-INDP formulations. Includes "DELTA_T" parameter for sliding windows to increase
        # efficiency.
        # Edit 2/8/16: "Sliding window" now overlaps.
        print 'hahahaha'


'''
This function computes the realized values of flow cost, unbalanced cost, and 
demand deficit at the end of each step according to the what the other agent
actually decides (as opposed to according to the guess)
For output items, look at the description of Decentralized_INDP()
'''
def Decentralized_INDP_Realized_Performance(N,iteration,indp_results,functionality,
                                            layers,T=1,controlled_layers=[1],
                                            print_cmd=False,saveJCModel=False):
    
    G_prime_nodes = [n[0] for n in N.G.nodes_iter(data=True) if n[1]['data']['inf_data'].net_id in layers]
    G_prime = N.G.subgraph(G_prime_nodes)
    
    interdep_nodes={}
    for u,v,a in G_prime.edges_iter(data=True):
        if a['data']['inf_data'].is_interdep and G_prime.node[v]['data']['inf_data'].net_id in controlled_layers:
            if v not in interdep_nodes:
                interdep_nodes[v]=[]
            interdep_nodes[v].append((u,G_prime.node[u]['data']['inf_data'].functionality))
    
    functionality_realized = copy.deepcopy(functionality)       
    for t in range(T):     
        for u, value in functionality[t].iteritems():     
            if functionality[t][u]==1.0 and G_prime.node[u]['data']['inf_data'].functionality==0.0:
                functionality_realized[t][u] = 0.0
             
        realCount = 0  
        realizations = {t:{}}         
        for v, value in interdep_nodes.iteritems():
            sum_u_functioanlity = 0.0
            for u in value:
                sum_u_functioanlity += u[1]
            
            realCount += 1 
            realizations[t][realCount]={'uValue':u[1],
                        'vValue':G_prime.node[v]['data']['inf_data'].functionality,
                        'vRepaired':G_prime.node[v]['data']['inf_data'].repaired,
                        'vName':v,'uName':u[0],
                        'vCorrected':False}    
            if sum_u_functioanlity == 0.0 and G_prime.node[v]['data']['inf_data'].functionality==1.0:
                realizations[t][realCount]['vCorrected'] = True
                if print_cmd:
                    print 'Correcting '+`v`+' to 0 (dep. on '+`value`+')'     
                     
    indp_results_Real = indp(N,0,1,layers=layers,controlled_layers=controlled_layers,
                             functionality=functionality_realized,
                                print_cmd=print_cmd)  
    for t in range(T):
        costs = indp_results.results[t]['costs']    
        nodeCost=costs["Node"]
        indp_results_Real[1][t]['costs']["Node"]=nodeCost
        arcCost=costs["Arc"]
        indp_results_Real[1][t]['costs']["Arc"]=arcCost
        spacePrepCost=costs["Space Prep"]
        indp_results_Real[1][t]['costs']["Space Prep"]=spacePrepCost
        flowCost=indp_results_Real[1][t]['costs']["Flow"]
        overSuppCost=indp_results_Real[1][t]['costs']["Over Supply"]
        underSuppCost=indp_results_Real[1][t]['costs']["Under Supply"]
        # Calculate total costs.
        indp_results_Real[1][t]['costs']["Total"]=flowCost+arcCost+nodeCost+overSuppCost+underSuppCost+spacePrepCost
        indp_results_Real[1][t]["Total no disconnection"]=spacePrepCost+arcCost+flowCost+nodeCost
            
    return indp_results_Real,realizations    


def create_judgment_matrix(N,T,layers,v_r=[],actions=[],judgment_type="OPTIMISTIC"):
    """Creates a functionality map for input into the functionality parameter in the indp function.
    :param N: An InfrastructureNetwork instance (created in infrastructure.py)
    :param T: Number of timesteps to optimize over.
    :param layers: Layer IDs of N included in the optimization.
    :param actions: An array of actions from a previous optimization. Likely taken from an INDPResults variable 'indp_result[t]['actions']'.
    :param judgment_type: If no actions are provided, assigns a default functionality. Options are: "OPTIMISTIC", "PESSIMISTIC" or ...
    :returns: A functionality dictionary used for input into indp.
    """
    functionality={}
    G_prime_nodes = [n[0] for n in N.G.nodes_iter(data=True) if n[1]['data']['inf_data'].net_id in layers]
    G_prime = N.G.subgraph(G_prime_nodes)
    N_prime = [n for n in G_prime.nodes_iter(data=True) if n[1]['data']['inf_data'].functionality==0.0]
    N_prime_nodes = [n[0] for n in G_prime.nodes_iter(data=True) if n[1]['data']['inf_data'].functionality==0.0]
    
    for t in range(T):
        functionality[t]={}
        functional_nodes = []
        # Generate resoration probabilities and corresponding bernoulli experiments 
        # for demand-based and deterministic demand-based judgments
        # Updates the bernoulli experiments (not the resoration probabilities) for each t
        interdepSrc = []
        detPriority = []
        if judgment_type == 'DEMAND' or judgment_type == 'DET-DEMAND':
            priorityList = demand_based_priority_List(N,layers)  
            if judgment_type == 'DET-DEMAND':
                sortedpriorityList = sorted(priorityList.items(), 
                    key=operator.itemgetter(1), reverse=True)
                
                num_layers = 1 #len(layers)
                if isinstance(v_r, (int, long)):
                    resCap = int(v_r/num_layers)
                else:
                    resCap = int(sum(v_r)/num_layers)  
                           
                for u,v,a in N.G.edges_iter(data=True):
                    if a['data']['inf_data'].is_interdep and u[1] in layers:
                        interdepSrc.append(u)
                        
                for i in sortedpriorityList:
                    if (i[0] in N_prime_nodes) and (len(detPriority)<(t+1)*resCap) and (i[0] in interdepSrc):
                        detPriority.append(i[0])
        # Nodes that are judged/known to be functional for t_p<t    
        for t_p in range(t):
            for key in functionality[t_p]:
                if functionality[t_p][key]==1.0:
                    functional_nodes.append(key)                     
        for n,d in G_prime.nodes_iter(data=True):
            #print "layers=",layers,"n=",n
            if d['data']['inf_data'].net_id in layers:
                # Undamged Nodes
                if G_prime.has_node(n) and (n,d) not in N_prime:
                    functionality[t][n]=1.0
                # Nodes that are judged/known to be functional for t_p<t
                elif n in functional_nodes:
                    functionality[t][key]=1.0   
                # Judgments
                else:
                    if judgment_type == "OPTIMISTIC":
                        functionality[t][n]=1.0
                    elif judgment_type == "PESSIMISTIC":
                        functionality[t][n]=0.0
                    elif judgment_type == "DEMAND":
                        functionality[t][n]=priorityList[n][1]
                    elif judgment_type == "DET-DEMAND":                        
                        if n in detPriority:                        
                            functionality[t][n]=1.0
                        else:
                            functionality[t][n]=0.0
                    elif judgment_type == "RANDOM":
                        functionality[t][n]=np.random.choice([0, 1], p=[0.5, 0.5])
                    elif judgment_type == "REALISTIC":
                        functionality[t][n]=d['data']['inf_data'].functionality
                    else:
                        if not n in functionality[t]:
                            functionality[t][n]=0.0                    
    return functionality

'''
This function generates the prioirt list for the demand-based guess
Here, an agent guesses that the (dependee) node in the 
other network is repaired until the next time step with the probability that 
is equal to the demand/supply value of the node divided by the maximum value
of demand/supply in the other network
Also, based on the above probability, a guess is generated for the node in the
other network.
The output of this functon is employed by guess_generator()
'''
def demand_based_priority_List(N,layers):
    G_prime_nodes = {}
    G_prime = {}
    com={}
    maxValues = {}
    prob = {}
    for P in layers:
        com[P] = [0] #assuming single commodity for all layers
        for l in com[P]:
            G_prime_nodes[P] = [n[0] for n in N.G.nodes_iter(data=True) if n[1]['data']['inf_data'].net_id==P]
            G_prime[P] = N.G.subgraph(G_prime_nodes[P])
            maxValues[P,l,'Demand'] = min([n[1]['data']['inf_data'].demand for n in G_prime[P].nodes_iter(data=True)])
            maxValues[P,l,'Supply'] = max([n[1]['data']['inf_data'].demand for n in G_prime[P].nodes_iter(data=True)])
            for n in G_prime[P].nodes_iter(data=True):
                if not n[0] in prob.keys():
                    p = []
                    for l in com[P]:
                        value = n[1]['data']['inf_data'].demand
                        if value>0:
                            p.append(value/maxValues[P,l,'Supply'])
                        elif value<=0:
                            p.append(value/maxValues[P,l,'Demand'])
                        else:
                            print 'Are you kidding me???'
                    prob[n[0]] = [max(p),np.random.choice([0, 1], p=[1-max(p), max(p)])]                
    return prob

def auction_resources(v_r,params,layers,T=1,print_cmd=True,judgment_type="OPTIMISTIC",auction_type="MDA",valuation_type='DTC_uniform'):
    """ allocate resources based on different types of auction and valuatoin.
    :param auction_type: Type of the auction: MDA(Multiunit Descending (First-price, Dutch) auction), MAA(Multiunit Ascending (Second-price, English) auction), MCA(Multiunit Combinatorial auction).
    :param valuation_type: Type of the valuation: DTC (Differential Total Cost), DTC_unifrom (DTC with uniform distribution), MDDN (Max Demand Damaged Nodes).
    """
    # Initialize failure scenario.
    InterdepNet=None
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=params['SIM_NUMBER'],magnitude=params["MAGNITUDE"])
    else:
        InterdepNet=params["N"]
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"]=1
    num_iterations = params["NUM_ITERATIONS"]   
                     
    #Compute Valuations
    if print_cmd:
        print "Compute Valuations (" + valuation_type + ")"
    valuation, optimal_valuation = compute_valuations(v_r,InterdepNet,layers=layers,
                                T=1,print_cmd=print_cmd,judgment_type=judgment_type,
                                valuation_type=valuation_type)
    
    #Auctioning
    if print_cmd:
        print "Auction (" + auction_type + ")" 
    resource_allocation = {P:[] for P in layers}
    PoA = {}
    PoA['optimal'] = optimal_valuation  
    PoA['winner'] = [] 
    sum_valuation = 0
    if auction_type=="MDA":
        cur_valuation = {v+1:{} for v in range(v_r)}
        for v in range(v_r):
            if print_cmd:
                print "Resource-%d"%(v+1),
            for P in layers:
                cur_valuation[v+1][P]= valuation[P][len(resource_allocation[P])]
            winner = max(cur_valuation[v+1].iteritems(), key=operator.itemgetter(1))[0]
            PoA['winner'].append(cur_valuation[v+1][winner])
            if cur_valuation[v+1][winner]!=0:
                if print_cmd:
                    print "Player %d wins!" % winner
                sum_valuation += cur_valuation[v+1][winner]
                resource_allocation[winner].append(v+1) 
            else:
                if print_cmd:
                    print "No auction winner!"
    if auction_type=="MAA":
        all_valuations = []
        for p,value in valuation.items():
            all_valuations.extend(value)
        all_valuations.sort()
        Q = {0:v_r*len(layers)} 
        q = {P:{0:v_r} for P in layers}
        p = {0: 0.0}
        t = 0
        while Q[t]>v_r:
            t += 1
            p[t] = all_valuations[t-1]
            Q[t] = 0.0
            for P in layers:
                q[P][t] = 0
                for i in range(len(valuation[P])):
                    if valuation[P][i] > p[t]:
                        q[P][t] += 1
                    else:
                        break
                Q[t] += q[P][t]
            
        sum_valuation = p[t]*Q[t]
        PoA['winner'] = [p[t] for v in range(int(Q[t]))]
        if Q[t]<v_r:
            for v in range(int(v_r-Q[t])):
                PoA['winner'].append(0.0)
                if print_cmd:
                    print "No auction winner for resource %d!" %(Q[t]+v+1)                
        for P in layers:
            resource_allocation[P] = range(1,q[P][t]+1)
            
    if auction_type=="MCA":
        m=Model('auction')
        m.setParam('OutputFlag',False)    
        
        # Add allocation variables and populate objective function.
        for P in layers:
            for v in range(v_r):
                m.addVar(name='y_'+`v+1`+","+`P`,vtype=GRB.BINARY,
                         obj=sum([-valuation[P][vv] for vv in range(v+1)]))
        m.update()
        # Add constraints
        numAllocatedResources=LinExpr()
        for P in layers:
            eachBidderAllocation=LinExpr()
            for v in range(v_r):
                numAllocatedResources+=m.getVarByName('y_'+`v+1`+","+`P`)*(v+1)
                eachBidderAllocation+=m.getVarByName('y_'+`v+1`+","+`P`)
            m.addConstr(eachBidderAllocation,GRB.LESS_EQUAL,1.0,"Bidder "+`P`+" allocation")   
        m.addConstr(numAllocatedResources,GRB.LESS_EQUAL,v_r,"Total number of resources")  
        #    print "Solving..."
        m.update()
        m.optimize()   
        for P in layers:
            for v in range(v_r):
                if m.getVarByName('y_'+`v+1`+","+`P`).x==1:
                    resource_allocation[P] = range(1,v+2)
                    for vv in range(v+1):
                        PoA['winner'].append(valuation[P][vv])
        sum_valuation = sum(PoA['winner'])  
#        m.write('model.lp')
#        m.write('model.sol')
    if sum_valuation!=0:
        PoA['poa'] = optimal_valuation/sum_valuation
    else:
        PoA['poa'] = -10
    
    return resource_allocation,PoA,valuation

def compute_valuations(v_r,InterdepNet,layers,T=1,print_cmd=True,judgment_type="OPTIMISTIC",valuation_type='DTC_uniform'):
    """ computes bidders' valuations for different number of resources
    :param valuation_type: Type of the valuation: DTC (Differential Total Cost), DTC_unifrom (DTC with uniform distribution), MDDN (Max Demand Damaged Nodes).
    """
    
    """ Calculating current total cost """
    currentTotalCost={}
    for P in layers:     
        '''!!! check what v_r must be for det demand JC'''
        indp_results = indp(InterdepNet,v_r=0,T=1,layers=layers,
                            controlled_layers=[P])
        currentTotalCost[P] = indp_results[1][0]['costs']['Total']
            
    """ Optimal Valuation """      
    indp_results = indp(InterdepNet,v_r=0,T=1,layers=layers,
                                controlled_layers=layers)
    optimal_total_cost_current = indp_results[1][0]['costs']['Total']
    indp_results = indp(InterdepNet,v_r=v_r,T=1,layers=layers,
                                controlled_layers=layers)
    optimal_total_cost = indp_results[1][0]['costs']['Total']
    optimal_valuation = optimal_total_cost_current - optimal_total_cost
    
    valuation={P:[] for P in layers}     
    if T == 1: # For iterative INDP formulation
        for P in layers:
            if print_cmd:
                print "Bidder-%d"%(P)
            if valuation_type=='DTC':
                for v in range(v_r):  
                    indp_results={}
                    negP=[x for x in layers if x != P]
                    functionality = create_judgment_matrix(InterdepNet,T,negP,v_r,
                                            actions=None,judgment_type=judgment_type) 
                    '''!!! check what v_r must be for det demand JC'''
                    indp_results = indp(InterdepNet,v_r=v+1,
                                T=1,layers=layers,controlled_layers=[P],
                                functionality=functionality,
                                print_cmd=print_cmd)
                    newTotalCost = indp_results[1][0]['costs']['Total']
                    if indp_results[1][0]['actions']!=[]:
                        valuation[P].append(currentTotalCost[P]-newTotalCost)
                        currentTotalCost[P] = newTotalCost
                    else:
                        valuation[P].append(0.0)
                    
            elif valuation_type=='DTC_uniform':
                for v in range(v_r):  
                    indp_results={}
                    totalCostBounds = []
                    for jt in ["PESSIMISTIC","OPTIMISTIC"]:
                        negP=[x for x in layers if x != P]
                        functionality = create_judgment_matrix(InterdepNet,T,negP,v_r,
                                                actions=None,judgment_type=jt) 
                        '''!!! check what v_r must be for det demand JC'''
                        indp_results = indp(InterdepNet,v_r=v+1,
                                    T=1,layers=layers,controlled_layers=[P],
                                    functionality=functionality,
                                    print_cmd=print_cmd)
                        totalCostBounds.append(indp_results[1][0]['costs']['Total'])
                    newTotalCost = np.random.uniform(min(totalCostBounds),
                                                    max(totalCostBounds),1)[0]
                    if indp_results[1][0]['actions']!=[]:
                        valuation[P].append(currentTotalCost[P]-newTotalCost)
                    else:
                        valuation[P].append(0.0)
                        
            elif valuation_type=='MDDN':
                G_prime_nodes = [n[0] for n in InterdepNet.G.nodes_iter(data=True) if n[1]['data']['inf_data'].net_id==P]
                G_prime = InterdepNet.G.subgraph(G_prime_nodes)
                dem_damaged_nodes = [abs(n[1]['data']['inf_data'].demand) for n in G_prime.nodes_iter(data=True) if n[1]['data']['inf_data'].repaired==0.0]
                dem_damaged_nodes_reverse_sorted = np.sort(dem_damaged_nodes)[::-1]
                if len(dem_damaged_nodes)>=v_r:
                    valuation[P] = dem_damaged_nodes_reverse_sorted[0:v_r].tolist()
                if len(dem_damaged_nodes)<v_r:
                    valuation[P] = dem_damaged_nodes_reverse_sorted[:].tolist()
                    for vv in range(v_r-len(dem_damaged_nodes)):
                        valuation[P].append(0.0)
            else:
                import sys
                sys.exit( "Wrong valuation type!!!")
    return valuation, optimal_valuation
                
def write_auction_csv(outdir,res_allocate,PoA,valuations,sample_num=1,suffix=""):
    if not os.path.exists(outdir):
        os.makedirs(outdir)        
    auction_file=outdir+"/auctions_"+`sample_num`+"_"+suffix+".csv"
    header = "t,"
    for key,value in res_allocate[0].items():
        header += "P"+`key`+","
    header += "PoA,optimal_val,winner_val"
    for p,value in valuations[0].items(): 
        header +=  ",bidder_"+`p`+"_valuation"
    with open(auction_file,'w') as f:
        f.write(header+"\n")
        for t,value in res_allocate.items():
            row = `t+1`+","
            for p,pvalue in value.items():
                row += `len(pvalue)`+','
            row += `PoA[t]['poa']`+','+`PoA[t]['optimal']`+','
            for pitem in PoA[t]['winner']:
                row += `pitem`+"|"            
            for p,pvalue in valuations[t].items():
                row += ','
                for pitem in pvalue:
                    row += `pitem`+"|"
            f.write(row+"\n")
            
def read_resourcec_allocation(df,sample_range,layers,T=1,L=3,suffix="",ref_method='indp',ci=None,listHDadd=None):  
    no_resources = df.no_resources.unique().tolist()
    mags= df.Magnitude.unique().tolist()
    decision_type = df.decision_type.unique().tolist()
    auction_type = df.auction_type.unique().tolist()
    valuation_type = df.valuation_type.unique().tolist()
    if listHDadd:
        listHD = pd.read_csv(listHDadd)   
    
    cols=['t','resource','decision_type','auction_type','valuation_type','sample','Magnitude','layer','no_resources','PoA','distance_to_optimal']
    df_res = pd.DataFrame(columns=cols)
    optimal_method = ['tdindp','indp','sample_indp_12Node']
    print '\nResource allocation|',
    for m in mags:
        for dt,nr,sr in itertools.product(decision_type,no_resources,sample_range):
            if listHDadd==None or len(listHD.loc[(listHD.set == sr) & (listHD.sce == m)].index):                        
                if dt in optimal_method:
                    compare_to_dir= '../results/'+dt+'_results_L'+`L`+'_m'+str(m)+'_v'+str(nr)
                    for t in range(T):
                        for P in range(len(layers)):
                            df_res=df_res.append({'t':t+1,'resource':0.0,'decision_type':dt,
                                'auction_type':'','valuation_type':'','sample':sr,
                                'Magnitude':m,'layer':`P+1`,'no_resources':nr,
                                'PoA':1,'distance_to_optimal':0.0}, ignore_index=True)
                    # Read optimal resource allocation based on the actions
                    action_file=compare_to_dir+"/actions_"+str(sr)+"_"+suffix+".csv" 
                    if os.path.isfile(action_file):
                        with open(action_file) as f:
                            lines=f.readlines()[1:]
                            for line in lines:
                                data=string.split(str.strip(line),",")
                                t=int(data[0])
                                action=str.strip(data[1])
                                P = int(action[-1])
                                if '/' in action:
                                    addition = 0.5
                                else:
                                    addition = 1.0
                                df_res.loc[(df_res['t']==t)&(df_res['decision_type']==dt)&
                                           (df_res['sample']==sr)&(df_res['Magnitude']==m)&
                                           (df_res['layer']==`P`)&(df_res['no_resources']==nr),'resource']+=addition
                else: 
                    # Read  resource allocation based on auction results
                    for at,vt in itertools.product(auction_type,valuation_type):
                        outdir= '../results/'+dt+'_results_L'+`L`+'_m'+str(m)+'_v'+str(nr)+'_auction_'+at+'_'+vt+'/auctions'
                        auction_file=outdir+"/auctions_"+str(sr)+"_"+suffix+".csv"
                        if os.path.isfile(auction_file):
                            with open(auction_file) as f:
                                lines=f.readlines()[1:]
                                for line in lines:
                                    data=string.split(str.strip(line),",")
                                    t=int(data[0])
                                    for P in range(len(layers)):
                                        poa = float(data[len(layers)+1])
                                        df_res=df_res.append({'t':t,'resource':float(data[P+1]),
                                            'decision_type':dt,'auction_type':at,
                                            'valuation_type':vt,'sample':sr,
                                            'Magnitude':m,'layer':`P+1`,'no_resources':nr,
                                            'PoA':poa}, ignore_index=True)
        print 'm%d'%(m),

    print '\nRelative allocation|',
    for m in mags:
        for nr,sr in itertools.product(no_resources,sample_range):
            if listHDadd==None or len(listHD.loc[(listHD.set == sr) & (listHD.sce == m)].index): 
                # Construct vector of resource allocation of reference method                       
                vector_res_ref = {P:np.zeros(T) for P in layers}
                for P in layers:
                    for t in range(T):
                        vector_res_ref[P][t]= df_res.loc[(df_res['t']==t+1)&
                                (df_res['decision_type']==ref_method)&
                                (df_res['sample']==sr)&(df_res['Magnitude']==m)&
                                (df_res['layer']==`P`)&(df_res['no_resources']==nr),'resource']
                # Compute distance of resource allocation vectors
                for dt,at,vt in itertools.product(decision_type,auction_type,valuation_type):
                    if dt!=ref_method and vt!=''and at!='':
                        vector_res = {P:np.zeros(T) for P in layers}
                        for P in layers:
                            row = (df_res['decision_type']==dt)&(df_res['sample']==sr)&(df_res['Magnitude']==m)&(df_res['layer']==`P`)&(df_res['no_resources']==nr)&(df_res['auction_type']==at)&(df_res['valuation_type']==vt)
                            for t in range(T):
                                vector_res[P][t] = df_res.loc[(df_res['t']==t+1)&row,'resource']
                            distance = np.linalg.norm(vector_res[P]-vector_res_ref[P]) #L2 norm
#                            distance = sum(abs(vector_res[P]-vector_res_ref[P])) #L1 norm
#                            distance = 1-scipy.stats.pearsonr(vector_res[P],vector_res_ref[P])[0] # correlation distance
                            df_res.loc[row,'distance_to_optimal']=distance/float(vector_res[P].shape[0])
        print 'm%d'%(m),
        
    return df_res

def write_judgments_csv(N,outdir,functionality,realizations,sample_num=1,agent=1,time=0,suffix=""):
    if not os.path.exists(outdir):
        os.makedirs(outdir)  
        
    interdep_nodes_src={}
    for u,v,a in N.G.edges_iter(data=True):
        if a['data']['inf_data'].is_interdep and N.G.node[v]['data']['inf_data'].net_id==agent:
            if u not in interdep_nodes_src:
                interdep_nodes_src[u]=[]
            interdep_nodes_src[u].append(v)
            
    if interdep_nodes_src:
        judge_file=outdir+'/judge_'+`sample_num`+"_agent"+`agent`+'_time'+`time`+'_'+suffix+".csv"
        
        header = "no.,src node,src layer,src actual func,src judge func,dest Name,dest layer,dest uncorrected func, dest repair,if dest corrected"
        with open(judge_file,'w') as f:
            f.write(header+"\n")
            for t,timeValue in realizations.items():
                for c,Value in timeValue.items():
                    row = `c`+','+`Value['uName']`+','\
                        +`Value['uValue']`+','+`functionality[t][Value['uName']]`+','\
                        +`Value['vName']`+','+`Value['vValue']`+','\
                        +`Value['vRepaired']`+','\
                        +`Value['vCorrected']`
                        
                    f.write(row+'\n')
    else:
        print 'No judgment by agent '+`agent`+'.'

def read_and_aggregate_results(mags,method_name,auction_type,valuation_type,suffixes,L,sample_range,no_resources=[3],listHDadd=None):
    columns = ['t','Magnitude','cost_type','decision_type','auction_type','valuation_type','no_resources','sample','cost']
    optimal_method = ['tdindp','indp','sample_indp_12Node']
    agg_results = pd.DataFrame(columns=columns)
    if listHDadd:
        listHD = pd.read_csv(listHDadd) 

    auction_type.append('')
    valuation_type.append('')        
    print "\nAggregating Results"
    for m in mags:
        for rc in no_resources:
            print 'm %d|v=%d|' %(m,rc),  
            for dt,at,vt in itertools.product(method_name,auction_type,valuation_type):
                if (dt in optimal_method) and ((at!='') or (vt!='')):
                    continue
                # Constructing the directory
                if dt in optimal_method:
                    full_suffix = '_L'+`L`+'_m'+`m`+'_v'+`rc`
                else:
                    full_suffix = '_L'+`L`+'_m'+`m`+'_v'+`rc`+'_auction_'+at+'_'+vt
                result_dir = '../results/'+dt+'_results'+full_suffix
                if os.path.exists(result_dir):
                    print '.',
#                    # Save average values to file #!!!!!!!!!!!
#                    results_average = INDPResults()
#                    results_average = results_average.from_results_dir(outdir=result_dir,
#                                        sample_range=sample_range,suffix=suffixes[i])
#                    
#                    outdir = '../results/average_cost_all/'
#                    if not os.path.exists(outdir):
#                        os.makedirs(outdir) 
#                    costs_file =outdir+method_name[i]+full_suffix+"_average_costs.csv"
#                    with open(costs_file,'w') as f:
#                        f.write("t,Space Prep,Arc,Node,Over Supply,Under Supply,Flow,Total\n")
#                        for t in results_average.results:
#                            costs=results_average.results[t]['costs']
#                            f.write(`t`+","+`costs["Space Prep"]`+","+`costs["Arc"]`+","
#                                    +`costs["Node"]`+","+`costs["Over Supply"]`+","+
#                                    `costs["Under Supply"]`+","+`costs["Flow"]`+","+
#                                    `costs["Total"]`+"\n")            
                            
                    # Save all results to Pandas dataframe
                    sample_result = INDPResults()
                    for s in sample_range:
                        if listHDadd==None or len(listHD.loc[(listHD.set == s) & (listHD.sce == m)].index):
                            for suf in suffixes:
                                if os.path.exists(result_dir+"/costs_"  +`s`+"_"+suf+".csv"):
                                    sample_result=sample_result.from_csv(result_dir,s,suffix=suf)
                            for t in sample_result.results:
                                for c in sample_result.cost_types:
                                    values = [t,m,c,dt,at,vt,rc,s,
                                            float(sample_result[t]['costs'][c])]
                                    agg_results = agg_results.append(dict(zip(columns,values)), ignore_index=True)
#                else:
#                    print "\nWARNING: Unable to find folder " + result_dir
            print 'Aggregated'
            
    return agg_results

def correct_tdindp_results(df,mags,method_name,sample_range):    
    # correct total cost of td-indp
    resource_cap = ['_fixed_layer_cap', '']
    tVector = df['t'].unique().tolist()
    for t in tVector:
        for m in mags:
            for rc in resource_cap:
                for sr in sample_range:
                    rows = df[(df['t']==t)&(df['Magnitude']==m)&
                             (df['method']=='tdindp_results')&(df['resource_cap']==rc)&
                             (df['sample']==sr)]
                    
                    if t!=int(tVector[-1]) and t!=0:
                        rowsNext = df[(df['t']==t+1)&(df['Magnitude']==m)&
                         (df['method']=='tdindp_results')&(df['resource_cap']==rc)&
                         (df['sample']==sr)]
                        
                        nodeCost=rows[rows['cost_type']=='Node']['cost'].values
                        arcCost=rows[rows['cost_type']=='Arc']['cost'].values
                        flowCost=rowsNext[rowsNext['cost_type']=='Flow']['cost'].values
                        overSuppCost=rowsNext[rowsNext['cost_type']=='Over Supply']['cost'].values
                        underSuppCost=rowsNext[rowsNext['cost_type']=='Under Supply']['cost'].values
                        spacePrepCost=rows[rows['cost_type']=='Space Prep']['cost'].values
                        
                        totalCost = flowCost+arcCost+nodeCost+overSuppCost+underSuppCost+spacePrepCost
                        
                        df.loc[(df['t']==t)&(df['Magnitude']==m)&(df['method']=='tdindp_results')&
                            (df['resource_cap']==rc)&(df['sample']==sr)&
                            (df['cost_type']=='Total'),'cost'] = totalCost
    return df
               
def relative_performance(df,sample_range,cost_type='Total',listHDadd=None,ref_method='indp'):    
    sns.set()
    auction_type = df.auction_type.unique().tolist()
    valuation_type = df.valuation_type.unique().tolist()
    no_resources = df.no_resources.unique().tolist()
    mags=df.Magnitude.unique().tolist()
    decision_type = df.decision_type.unique().tolist()
    columns = ['Magnitude','cost_type','decision_type','auction_type','valuation_type','no_resources','sample','Area','lambda_TC']
    lambda_df = pd.DataFrame(columns=columns)
    if listHDadd:
        listHD = pd.read_csv(listHDadd)    

    # Computing reference area for lambda
    ref_at=''
    ref_vt=''
    print 'Ref area calculation|',
    for m in mags:
        for dt,at,vt,nr,sr in itertools.product([ref_method],[ref_at],[ref_vt],no_resources,sample_range):
            if listHDadd==None or len(listHD.loc[(listHD.set == sr) & (listHD.sce == m)].index):
                rows = df[(df['Magnitude']==m)&(df['decision_type']==dt)&
                         (df['sample']==sr)&(df['cost_type']==cost_type)&
                         (df['auction_type']==at)&(df['valuation_type']==vt)&
                         (df['no_resources']==nr)]
                                  
                if not rows.empty:
                    area = np.trapz(rows.cost[:20],dx=1)
                        
                    tempdf = pd.Series()
                    tempdf['Magnitude'] = m
                    tempdf['cost_type'] = cost_type
                    tempdf['decision_type'] = dt
                    tempdf['auction_type'] = at  
                    tempdf['valuation_type'] = vt   
                    tempdf['no_resources'] = nr 
                    tempdf['sample'] = sr  
                    tempdf['Area'] = area  
                    lambda_df=lambda_df.append(tempdf,ignore_index=True)
        print 'm'+`int(m)`,
    # Computing areaa and lambda
    print '\nLambda calculation|',
    for m in mags:
        for nr,sr in itertools.product(no_resources,sample_range):        
            if (listHDadd==None or len(listHD.loc[(listHD.set == sr) & (listHD.sce == m)].index)):
                # Check if reference area exists
                cond = ((lambda_df['Magnitude']==m)&(lambda_df['decision_type']==ref_method)&
                    (lambda_df['auction_type']==ref_at)&
                    (lambda_df['valuation_type']==ref_vt)&
                    (lambda_df['cost_type']==cost_type)&
                    (lambda_df['sample']==sr)&
                    (lambda_df['no_resources']==nr))
                if not cond.any():
                    import sys
                    sys.exit('Reference type is not here! for m %d|resource %d' %(m,nr))
                    
                ref_area=float(lambda_df.loc[cond==True,'Area'])
                for dt,at,vt in itertools.product(decision_type,auction_type,valuation_type):
                     if dt!=ref_method:
                        rows = df[(df['Magnitude']==m)&(df['decision_type']==dt)&
                                 (df['sample']==sr)&(df['cost_type']==cost_type)&
                                 (df['auction_type']==at)&(df['valuation_type']==vt)&
                                 (df['no_resources']==nr)]
                                          
                        if not rows.empty:
                            area = np.trapz(rows.cost[:20],dx=1)
                            lambda_TC = 'nan'
                            if ref_area != 0.0 and area != 'nan':
                                lambda_TC = (ref_area-float(area))/ref_area
                            elif area == 0.0:
                                lambda_TC = 0.0
                            else:
                                pass  
                              
                            tempdf = pd.Series()
                            tempdf['Magnitude'] = m
                            tempdf['cost_type'] = cost_type
                            tempdf['decision_type'] = dt
                            tempdf['auction_type'] = at  
                            tempdf['valuation_type'] = vt   
                            tempdf['no_resources'] = nr 
                            tempdf['sample'] = sr  
                            tempdf['Area'] = area  
                            tempdf['lambda_TC'] = lambda_TC
                            lambda_df=lambda_df.append(tempdf,ignore_index=True)
        print 'm'+`int(m)`, 
    return lambda_df

def plot_performance_curves(df,x='t',y='cost',cost_type='Total',
                            decision_names=['tdindp_results'],
                            auction_type=None,valuation_type=None,
                            ci=None):
    sns.set(context='notebook',style='darkgrid')
#    plt.rc('text', usetex=True)
#    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    no_resources = df.no_resources.unique().tolist()
    if not auction_type:
        auction_type = df.auction_type.unique().tolist()
    auction_type.remove('')
    if not valuation_type:
        valuation_type = df.valuation_type.unique().tolist()
    valuation_type.remove('')
    T = len(df[x].unique().tolist())
    
    fig, axs = plt.subplots(len(valuation_type), len(no_resources), sharex=True, sharey=True, tight_layout=False)
    for idxnr,nr in enumerate(no_resources):
        for idxvt,vt in enumerate(valuation_type):
            if len(valuation_type)==1 and len(no_resources)==1:
                ax=axs
            elif len(valuation_type)==1:
                ax = axs[idxnr]
            elif len(no_resources)==1:
                ax = axs[idxvt]
            else:
                ax = axs[idxvt,idxnr]
                
            with sns.xkcd_palette(['black',"windows blue",'red',"green"]): #sns.color_palette("muted"):
                ax = sns.lineplot(x=x, y=y, hue="auction_type", style='decision_type',
                    markers=False, ci=ci, ax=ax,legend='full',
                    data=df[(df['cost_type']==cost_type)&
                            (df['decision_type'].isin(decision_names))&
                            (df['no_resources']==nr)&
                            ((df['valuation_type']==vt)|(df['valuation_type']==''))]) 
                ax.set(xlabel=r'time step $t$', ylabel=cost_type+' Cost')
                ax.get_legend().set_visible(False)
                ax.xaxis.set_ticks(np.arange(0,11 , 1.0))   #ax.get_xlim()                          
    handles, labels = ax.get_legend_handles_labels()
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5)
    
    if len(valuation_type)==1 and len(no_resources)==1:
        axx=[axs]
        axy=[axs]
    elif len(valuation_type)==1:
        axy = [axs[0]]
        axx = axs
    elif len(no_resources)==1:
        axy = axs
        axx = [axs[0]]
    else:
        axx = axs[0,:]
        axy = axs[:,0]  
    for idx, ax in enumerate(axx):
        ax.set_title(r'Total resources=%d'%(no_resources[idx]))
    for idx, ax in enumerate(axy):
        ax.annotate('Valuation = '+valuation_type[idx], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label, textcoords='offset points', ha='right', va='center', rotation=90) 
        
    plt.savefig('Performance_curves.pdf',dpi=600)  
    
def plot_relative_performance(lambda_df,cost_type='Total'):   
#    sns.set(context='notebook',style='darkgrid')
#    plt.rc('text', usetex=True)
#    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    no_resources = lambda_df.no_resources.unique().tolist()
    auction_type = lambda_df.auction_type.unique().tolist()
    auction_type.remove('')
    valuation_type = lambda_df.valuation_type.unique().tolist()
    valuation_type.remove('')
    
    fig, axs = plt.subplots(len(valuation_type), len(auction_type),sharex=True, sharey='row',tight_layout=False)
    for idxnr,nr in enumerate(auction_type):   
        for idxvt,vt in enumerate(valuation_type): 
            if len(valuation_type)==1 and len(auction_type)==1:
                ax=axs
            elif len(valuation_type)==1:
                ax = axs[idxnr]
            elif len(auction_type)==1:
                ax = axs[idxvt]
            else:
                ax = axs[idxvt,idxnr]
                                           
            with sns.color_palette("RdYlGn", 8):  #sns.color_palette("YlOrRd", 7)
                ax=sns.barplot(x='no_resources',y='lambda_TC',hue="decision_type",
                            data=lambda_df[(lambda_df['cost_type']==cost_type)&
                                                (lambda_df['lambda_TC']!='nan')&
                                                ((lambda_df['auction_type']==nr)|(lambda_df['auction_type']==''))&
                                                ((lambda_df['valuation_type']==vt)|(lambda_df['valuation_type']==''))], 
                                linewidth=0.5,edgecolor=[.25,.25,.25],
                                capsize=.05,errcolor=[.25,.25,.25],errwidth=1,ax=ax) 
                ax.get_legend().set_visible(False)
                ax.grid(which='major', axis='y', color=[.75,.75,.75], linewidth=.75)
                ax.set_xlabel(r'No. resources')
                if idxvt!=len(valuation_type)-1:
                    ax.set_xlabel('')
                ax.set_ylabel(r'E[$\lambda_{%s}$]'%('TC'))
                if idxnr!=0:
                    ax.set_ylabel('')
                ax.xaxis.set_label_position('bottom')  
#                ax.xaxis.tick_top()
                ax.set_facecolor('w')
                
    handles, labels = ax.get_legend_handles_labels()   
    labels = correct_legend_labels(labels)
    fig.legend(handles, labels,loc='upper right',frameon =True,framealpha=0.5, ncol=1) 
     
    if len(auction_type)==1 and len(valuation_type)==1:
        axx=[axs]
        axy=[axs]
    elif len(auction_type)==1:
        axx = [axs[0]]
        axy = axs
    elif len(valuation_type)==1:
        axx = axs
        axy = [axs[0]]
    else:
        axx = axs[0,:]
        axy = axs[:,0]  
    for idx, ax in enumerate(axx):
        ax.set_title(r'Auction Type = %s'%(auction_type[idx]))
    for idx, ax in enumerate(axy):
        ax.annotate('Valuation:'+valuation_type[idx],xy=(0.1, 0.5),xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center',rotation=90) 

    plt.savefig('Relative_perforamnce.pdf',dpi=600)
    
def plot_auction_allocation(df_res,ci=None):  
#    sns.set(context='notebook',style='darkgrid')
#    plt.rc('text', usetex=True)
#    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    no_resources = df_res.no_resources.unique().tolist()
    layer = df_res.layer.unique().tolist()
    auction_type = df_res.auction_type.unique().tolist()
    auction_type.remove('')
    valuation_type = df_res.valuation_type.unique().tolist()
    valuation_type.remove('')
    T = len(df_res.t.unique().tolist())

    for idxat,at in enumerate(valuation_type):
        fig, axs = plt.subplots(len(layer), len(no_resources), sharex=True, sharey='col', tight_layout=False)
        for idxnr,nr in enumerate(no_resources):
            for idxvt,vt in enumerate(layer):
                if len(layer)==1 and len(no_resources)==1:
                    ax=axs
                elif len(layer)==1:
                    ax = axs[idxnr]
                elif len(no_resources)==1:
                    ax = axs[idxvt]
                else:
                    ax = axs[idxvt,idxnr]
                    
                with sns.xkcd_palette(['black',"windows blue",'red',"green"]): #sns.color_palette("muted"):
                    ax = sns.lineplot(x='t', y='resource', hue="auction_type", style='decision_type',
                        markers=True, ci=ci, ax=ax,legend='full', 
                        data=df_res[(df_res['layer']==vt)&
                                (df_res['no_resources']==nr)&
                                ((df_res['valuation_type']==at)|(df_res['valuation_type']==''))]) 
                    ax.get_legend().set_visible(False)
                    ax.set(xlabel=r'time step $t$', ylabel='No. resources')
                    ax.xaxis.set_ticks(np.arange(1, 11, 1.0))   #ax.get_xlim()       
#                    ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 1.0), minor=True)
                    ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1], 1.0))  
                    ax.grid(b=True, which='major', color='w', linewidth=1.0)
#                    ax.grid(b=True, which='minor', color='w', linewidth=0.5)    
                       
        handles, labels = ax.get_legend_handles_labels()
        labels = correct_legend_labels(labels)
        fig.legend(handles, labels, loc='upper right', ncol=1, framealpha=0.5, labelspacing=0.2) #(0.75,0.6)
        
        if len(no_resources)==1 and len(layer)==1:
            axx=[axs]
            axy=[axs]
        elif len(no_resources)==1:
            axx = [axs[0]]
            axy = axs
        elif len(layer)==1:
            axx = axs
            axy = [axs[0]]
        else:
            axx = axs[0,:]
            axy = axs[:,0]  
        fig.suptitle('Valuation Type = '+valuation_type[idxat])
        for idx, ax in enumerate(axx):
            ax.set_title(r'Total resources = %d'%(no_resources[idx]))
        for idx, ax in enumerate(axy):
            ax.annotate('Layer '+layer[idx],xy=(0.1, 0.5),xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',ha='right',va='center',rotation=90)  
  
        plt.savefig('Allocations_'+at+'.pdf',dpi=600)

def plot_relative_allocation(df_res):   
    sns.set(context='notebook',style='darkgrid')
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    no_resources = df_res.no_resources.unique().tolist()
    layer = df_res.layer.unique().tolist()
    decision_type = df_res.decision_type.unique().tolist()
    auction_type = df_res.auction_type.unique().tolist()
    auction_type.remove('')
    valuation_type = df_res.valuation_type.unique().tolist()
    valuation_type.remove('')
    
#    pals = [sns.color_palette("Blues"),sns.color_palette("Reds"),sns.color_palette("Greens")]
    clrs = [['strawberry','salmon pink'],['azure','light blue'],['green','light green'],['purple','orchid']]
    fig, axs = plt.subplots(len(valuation_type), len(auction_type),sharex=True,
                            sharey=True,tight_layout=False, figsize=(10,7))
    for idxat,at in enumerate(auction_type):   
        for idxvt,vt in enumerate(valuation_type): 
            if len(auction_type)==1 and len(valuation_type)==1:
                ax=axs
            elif len(valuation_type)==1:
                ax = axs[idxat]
            elif len(auction_type)==1:
                ax = axs[idxvt]
            else:
                ax = axs[idxvt,idxat]
            data_ftp = df_res[(df_res['auction_type']==at)&(df_res['valuation_type']==vt)]
            
            for dt in decision_type:
                for nr in no_resources:
                    bottom = 0
                    for P in layer:
                        data_ftp.loc[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['no_resources']==nr),'distance_to_optimal']+=bottom
                        bottom=data_ftp[(data_ftp['layer']==P)&(data_ftp['decision_type']==dt)&(data_ftp['no_resources']==nr)]['distance_to_optimal'].mean()
            for P in reversed(layer):    
                with sns.xkcd_palette(clrs[int(P)-1]): #pals[int(P)-1]:
                    ax=sns.barplot(x='no_resources',y='distance_to_optimal',hue="decision_type",
                                data=data_ftp[(data_ftp['layer']==P)], 
                                linewidth=0.5,edgecolor=[.25,.25,.25],
                                capsize=.05,errcolor=[.25,.25,.25],errwidth=.75,ax=ax)
               
            ax.get_legend().set_visible(False)
            ax.grid(which='major', axis='y', color=[.75,.75,.75], linewidth=.75)
            ax.set_xlabel(r'No. Resources')
            if idxvt!=len(valuation_type)-1:
                ax.set_xlabel('')
            ax.set_ylabel(r'$E[\omega^k(r^k_d,r^k_c)]$')
            if idxat!=0:
                ax.set_ylabel('')
            ax.xaxis.set_label_position('bottom')  
            ax.set_facecolor('w')
                
    handles, labels = ax.get_legend_handles_labels()   
    labels = correct_legend_labels(labels)
    for idx,lab in enumerate(labels):
        layer_num = len(layer) - idx//(len(decision_type)-1)
        labels[idx] = lab[:7] + '. (Layer ' + `layer_num` + ')'
    lgd = fig.legend(handles, labels,loc='center', bbox_to_anchor=(0.5, 0.95),
               frameon =True,framealpha=0.5, ncol=4)     #, fontsize='small'
    if len(auction_type)==1 and len(valuation_type)==1:
        axx=[axs]
        axy=[axs]
    elif len(auction_type)==1:
        axx = [axs[0]]
        axy = axs
    elif len(valuation_type)==1:
        axx = axs
        axy = [axs[0]]
    else:
        axx = axs[0,:]
        axy = axs[:,0]
    for idx, ax in enumerate(axx):
        ax.set_title(r'Auction Type: %s'%(auction_type[idx]))
    for idx, ax in enumerate(axy):
        rowtitle = ax.annotate('Valuation: '+valuation_type[idx],xy=(0.1, 0.5),xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center',rotation=90)
    plt.savefig('Allocation_Difference.pdf', bbox_extra_artists=(rowtitle,lgd,), dpi=600)    #, bbox_inches='tight'
    
def correct_legend_labels(labels):
    labels = ['iINDP' if x=='sample_indp_12Node' else x for x in labels]
    labels = ['JC Optimistic' if x=='sample_judgeCall_12Node_OPTIMISTIC' else x for x in labels]
    labels = ['JC Pessimistic' if x=='sample_judgeCall_12Node_PESSIMISTIC' else x for x in labels]
    labels = ['Auction Type' if x=='auction_type' else x for x in labels]
    labels = ['Decision Type' if x=='decision_type' else x for x in labels]
    labels = ['iINDP' if x=='indp' else x for x in labels]
    labels = ['JC Optimistic' if x=='judgeCall_OPTIMISTIC' else x for x in labels]
    labels = ['JC Pessimistic' if x=='judgeCall_PESSIMISTIC' else x for x in labels]
    labels = ['Auction Type' if x=='auction_type' else x for x in labels]
    labels = ['Decision Type' if x=='decision_type' else x for x in labels]
    labels = ['MCA' if x=='EC' else x for x in labels]
    return labels