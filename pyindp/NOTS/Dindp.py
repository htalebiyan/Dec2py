import pandas as pd
#import seaborn as sns
import numpy as np
from indp import *
import os.path
import operator
import networkx as nx
from infrastructure import *
from indputils import *
import copy
from gurobipy import *
import itertools 
import time
import sys
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
 
    num_iterations = params["NUM_ITERATIONS"]
    v_r=params["V"]
    if isinstance(v_r, (int, long)):
        v_r = [v_r]
    if auction_type:
        output_dir = params["OUTPUT_DIR"]+'_L'+`len(layers)`+'_m'+`params["MAGNITUDE"]`+"_v"+`sum(v_r)`+'_auction_'+auction_type+'_'+valuation_type
    else:
        output_dir = params["OUTPUT_DIR"]+'_L'+`len(layers)`+'_m'+`params["MAGNITUDE"]`+"_v"+`sum(v_r)`+'_uniform_alloc'
    
    Dindp_results={P:INDPResults() for P in layers}   
    Dindp_results_Real={P:INDPResults() for P in layers} 
    currentTotalCost = {}
    if T == 1: 
        if auction_type:
            print "\n--Running Judgment Call with type "+judgment_type +" with auction "+auction_type+ ' & valuation '+ valuation_type
        else:
            print "\n--Running Judgment Call with type "+judgment_type +" with uniform allocation "
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
        res_alloc_time={}
        for i in range(num_iterations):
            if print_cmd:
                print "\n-Iteration "+`i`+"/"+`num_iterations-1`
            
            res_alloc_time_start = time.time()
            v_r_applied = []
            if auction_type:
                res_allocate[i],PoA[i],valuations[i],auction_time,valuation_time=auction_resources(sum(v_r),params,
                    layers=layers,T=1,print_cmd=print_cmd,judgment_type=judgment_type,
                    auction_type=auction_type,valuation_type=valuation_type)
               
                for key, value in res_allocate[i].items():
                    v_r_applied.append(len(value))
            elif len(v_r)!=len(layers):
                v_r_applied =  [v_r[0]/len(layers) for x in layers]
                for x in range(v_r[0]%len(layers)):
                    v_r_applied[x]+=1
                res_allocate[i] = {P:[] for P in layers}
                for P in layers:
                    res_allocate[i][P]=range(1,1+v_r_applied[P-1])
            else:
                v_r_applied = v_r
                res_allocate[i] = {P:[] for P in layers}
                for P in layers:
                    res_allocate[i][P]=range(1,1+v_r_applied[P-1])
            if auction_type:
                res_alloc_time[i]=[time.time()-res_alloc_time_start,auction_time,valuation_time]
            else:
                res_alloc_time[i]=[time.time()-res_alloc_time_start,0,0]
            
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
                                print_cmd=print_cmd,time_limit=10*60)
                        
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
                    write_judgments_csv(output_dir_judgments,realizations,
                                        sample_num=params["SIM_NUMBER"],
                                        agent=P,time=i+1,suffix="")       
        # Calculate sum of costs    
        Dindp_results_sum = INDPResults()
        Dindp_results_Real_sum = INDPResults()
        cost_types = Dindp_results[1][0]['costs'].keys()
        for i in range(num_iterations+1):   
            sum_run_time = 0.0
            sum_run_time_Real = 0.0             
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
                if Dindp_results[P][i]['run_time']>sum_run_time:
                    sum_run_time = Dindp_results[P][i]['run_time']
                if Dindp_results_Real[P][i]['run_time']>sum_run_time_Real:
                    sum_run_time_Real = Dindp_results_Real[P][i]['run_time']
                for a in Dindp_results[P][i]['actions']:
                    Dindp_results_sum.add_action(i,a) 
            Dindp_results_sum.add_run_time(i,sum_run_time)
            Dindp_results_Real_sum.add_run_time(i,sum_run_time_Real)
                    
        output_dir_auction = output_dir + '/auctions'        
        if auction_type:
            write_auction_csv(output_dir_auction,res_allocate,res_alloc_time,PoA,valuations,sample_num=params["SIM_NUMBER"],suffix="") 
        else:
            write_auction_csv(output_dir_auction,res_allocate,res_alloc_time,sample_num=params["SIM_NUMBER"],suffix="")
        # Save results of D-iINDP run to file.
        if saveJC:   
            output_dir_agents = output_dir + '/agents'
            make_dir(output_dir)
            make_dir(output_dir_agents)
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
            if u not in interdep_nodes:
                interdep_nodes[u]=[]
            interdep_nodes[u].append(v)
    list_interdep_nodes = interdep_nodes.keys()
    
    functionality_realized = copy.deepcopy(functionality) 
    realizations = {t:{} for t in range(T)}       
    for t in range(T):     
        realCount = 0 
        for u, value in functionality[t].iteritems(): 
            if u in list_interdep_nodes:
                realCount += 1 
                vValues = [G_prime.node[x]['data']['inf_data'].functionality for x in interdep_nodes[u]]
                realizations[t][realCount]={'vNames':interdep_nodes[u],'uName':u,
                            'uJudge':functionality[t][u],'uCorrected':False,'vValues':vValues} 
                
                if functionality[t][u]==1.0 and G_prime.node[u]['data']['inf_data'].functionality==0.0:
                    functionality_realized[t][u] = 0.0
                    realizations[t][realCount]['uCorrected'] = True
                    if print_cmd:
                        print 'Correct '+`u`+' to 0 (affect. '+`vValues`+')'  
                     
    indp_results_Real = indp(N,v_r=0,T=1,layers=layers,controlled_layers=controlled_layers,
                             functionality=functionality_realized,
                                print_cmd=print_cmd,time_limit=10*60)  
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
    valuation, optimal_valuation, valuation_time = compute_valuations(v_r,InterdepNet,layers=layers,
                                T=1,print_cmd=print_cmd,judgment_type=judgment_type,
                                valuation_type=valuation_type)
    
    #Auctioning
    if print_cmd:
        print "Auction (" + auction_type + ")" 
    start_time_auction = time.time()
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
#            if cur_valuation[v+1][winner]==0:
#                for x in layers:
#                    if len(resource_allocation[x])==0:
#                        winner = x
#                        break
##                if print_cmd:
#                print "Player %d wins (generously)!" % winner
#                sum_valuation += cur_valuation[v+1][winner]
#                resource_allocation[winner].append(v+1)             
            if cur_valuation[v+1][winner]>0:
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
    auction_time = time.time()-start_time_auction
    if sum_valuation!=0:
        PoA['poa'] = optimal_valuation/sum_valuation
    else:
        PoA['poa'] = -10
    
    return resource_allocation,PoA,valuation,auction_time,valuation_time

def compute_valuations(v_r,InterdepNet,layers,T=1,print_cmd=True,judgment_type="OPTIMISTIC",valuation_type='DTC_uniform',compute_optimal_valuation=False):
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
    optimal_valuation = 1.0
    if compute_optimal_valuation:      
        indp_results = indp(InterdepNet,v_r=0,T=1,layers=layers,
                                    controlled_layers=layers)
        optimal_total_cost_current = indp_results[1][0]['costs']['Total']
        indp_results = indp(InterdepNet,v_r=v_r,T=1,layers=layers,
                                    controlled_layers=layers)
        optimal_total_cost = indp_results[1][0]['costs']['Total']
        optimal_valuation = optimal_total_cost_current - optimal_total_cost
    
    valuation={P:[] for P in layers} 
    valuation_time=[] 
    if T == 1: # For iterative INDP formulation
        for P in layers:
            start_time_val = time.time()
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
                                print_cmd=print_cmd,time_limit=2*60)
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
                                    print_cmd=print_cmd,time_limit=2*60)
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
                sys.exit( "Wrong valuation type!!!")
            valuation_time.append(time.time()-start_time_val)
    return valuation, optimal_valuation, valuation_time
                
def write_auction_csv(outdir,res_allocate,res_alloc_time,PoA=None,valuations=None,sample_num=1,suffix=""):
    make_dir(outdir)
    auction_file=outdir+"/auctions_"+`sample_num`+"_"+suffix+".csv"
    # Making header
    header = "t,"
    for key,value in res_allocate[0].items():
        header += "P"+`key`+","
    if valuations:
        header += "PoA,optimal_val,winner_val,"
        for p,value in valuations[0].items(): 
            header +=  "bidder_"+`p`+"_valuation,"
    header += "Res Alloc Time,Auction Time,"
    if valuations:
        for key,value in res_allocate[0].items():
            header += "Val. Time P"+`key`+","
    # Write to file
    with open(auction_file,'w') as f:
        f.write(header+"\n")
        for t,value in res_allocate.items():
            row = `t+1`+","
            for p,pvalue in value.items():
                row += `len(pvalue)`+','
            if valuations:
                row += `PoA[t]['poa']`+','+`PoA[t]['optimal']`+','
                for pitem in PoA[t]['winner']:
                    row += `pitem`+"|"  
                row += ','
                for p,pvalue in valuations[t].items():
                    for pitem in pvalue:
                        row += `pitem`+"|"
                    row += ','
            row+=`res_alloc_time[t][0]`+','+`res_alloc_time[t][1]`+','
            if valuations:
                for titem in res_alloc_time[t][2]:
                    row += `titem`+','
                    
            f.write(row+"\n")
            
def read_resourcec_allocation(df,combinations,optimal_combinations,ref_method='indp',suffix="",root_result_dir='../results/'):  
    cols=['t','resource','decision_type','auction_type','valuation_type','sample','Magnitude','layer','no_resources','normalized_resources','PoA']
    T = int(max(df.t.unique().tolist()))
    df_res = pd.DataFrame(columns=cols, dtype=int)
    print '\nResource allocation\n',
    for idx,x in enumerate(optimal_combinations):                       
        compare_to_dir= root_result_dir+x[4]+'_results_L'+`x[2]`+'_m'+`x[0]`+'_v'+`x[3]`
        for t in range(T):
            for P in range(1,x[2]+1):
                df_res=df_res.append({'t':t+1,'resource':0.0,'normalized_resource':0.0,
                    'decision_type':x[4],'auction_type':'','valuation_type':'','sample':x[1],
                    'Magnitude':x[0],'layer':P,'no_resources':x[3],'PoA':1}, ignore_index=True)
        # Read optimal resource allocation based on the actions
        action_file=compare_to_dir+"/actions_"+`x[1]`+"_"+suffix+".csv" 
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
                    row = (df_res['t']==t)&(df_res['decision_type']==x[4])&(df_res['sample']==x[1])&(df_res['Magnitude']==x[0])&(df_res['layer']==P)&(df_res['no_resources']==x[3])
                    df_res.loc[row,'resource']+=addition
                    df_res.loc[row,'normalized_resource']+=addition/float(x[3])
                    
        if idx%(len(combinations+optimal_combinations)/100+1)==0:           
            update_progress(idx+1,len(optimal_combinations)+len(combinations))
    # Read  resource allocation based on auction results
    for idx,x in enumerate(combinations): 
        if x[5] in ['Uniform']:
            outdir= root_result_dir+x[4]+'_results_L'+`x[2]`+'_m'+`x[0]`+'_v'+`x[3]`+'_uniform_alloc/auctions'
        else:
            outdir= root_result_dir+x[4]+'_results_L'+`x[2]`+'_m'+`x[0]`+'_v'+`x[3]`+'_auction_'+x[5]+'_'+x[6]+'/auctions'
        auction_file=outdir+"/auctions_"+`x[1]`+"_"+suffix+".csv"
        if os.path.isfile(auction_file):
            with open(auction_file) as f:
                lines=f.readlines()[1:]
                for line in lines:
                    data=string.split(str.strip(line),",")
                    t=int(data[0])
                    for P in range(1,x[2]+1):
                        if x[5] in ['Uniform']:
                            poa = 0.0
                        else:
                            poa = float(data[x[2]+1])
                        df_res=df_res.append({'t':t,'resource':float(data[P]),
                            'normalized_resource':float(data[P])/float(x[3]),
                            'decision_type':x[4],'auction_type':x[5],'valuation_type':x[6],'sample':x[1],
                            'Magnitude':x[0],'layer':P,'no_resources':x[3],'PoA':poa}, ignore_index=True)
                
        if idx%(len(combinations+optimal_combinations)/100+1)==0:   
            update_progress(len(optimal_combinations)+idx+1,len(optimal_combinations)+len(combinations))
    update_progress(len(optimal_combinations)+idx+1,len(optimal_combinations)+len(combinations)) 
    
    cols=['decision_type','auction_type','valuation_type','sample','Magnitude','layer','no_resources','distance_to_optimal','norm_distance_to_optimal']
    T = int(max(df.t.unique().tolist()))
    df_res_rel = pd.DataFrame(columns=cols, dtype=int)
    print '\nRelative allocation\n',
    for idx,x in enumerate(combinations+optimal_combinations): 
        # Construct vector of resource allocation of reference method 
        if x[4]!=ref_method:                      
            vector_res_ref = {P:np.zeros(T) for P in range(1,x[2]+1)}
            for P in range(1,x[2]+1):
                for t in range(T):
                    vector_res_ref[P][t]= df_res.loc[(df_res['t']==t+1)&
                            (df_res['decision_type']==ref_method)&
                            (df_res['sample']==x[1])&(df_res['Magnitude']==x[0])&
                            (df_res['layer']==P)&(df_res['no_resources']==x[3]),'resource']
            # Compute distance of resource allocation vectors
            vector_res = {P:np.zeros(T) for P in range(1,x[2]+1)}
            for P in range(1,x[2]+1):
                row = (df_res['decision_type']==x[4])&(df_res['sample']==x[1])&(df_res['Magnitude']==x[0])&(df_res['layer']==P)&(df_res['no_resources']==x[3])&(df_res['auction_type']==x[5])&(df_res['valuation_type']==x[6])
                for t in range(T):
                    vector_res[P][t] = df_res.loc[(df_res['t']==t+1)&row,'resource']
                distance = np.linalg.norm(vector_res[P]-vector_res_ref[P]) #L2 norm
                norm_distance = np.linalg.norm(vector_res[P]/float(x[3])-vector_res_ref[P]/float(x[3]))
    #            distance = sum(abs(vector_res[P]-vector_res_ref[P])) #L1 norm
    #            distance = 1-scipy.stats.pearsonr(vector_res[P],vector_res_ref[P])[0] # correlation distance
                df_res_rel=df_res_rel.append({'decision_type':x[4],'auction_type':x[5],
                    'valuation_type':x[6],'sample':x[1],'Magnitude':x[0],'layer':P,'no_resources':x[3],
                    'distance_to_optimal':distance/float(vector_res[P].shape[0]),
                    'norm_distance_to_optimal':norm_distance/float(vector_res[P].shape[0])}, ignore_index=True)
                
            if idx%(len(combinations+optimal_combinations)/100+1)==0:
                update_progress(idx+1,len(combinations+optimal_combinations))   
    update_progress(idx+1,len(combinations+optimal_combinations))
    
    return df_res,df_res_rel

def write_judgments_csv(outdir,realizations,sample_num=1,agent=1,time=0,suffix=""):
    make_dir(outdir)
    judge_file=outdir+'/judge_'+`sample_num`+"_agent"+`agent`+'_time'+`time`+'_'+suffix+".csv"
    header = "no.,src node,src layer,src judge,if src corr.,dest Names,dest init. funcs"
    with open(judge_file,'w') as f:
        f.write(header+"\n")
        for t,timeValue in realizations.iteritems():
            if timeValue:
                for c,Value in timeValue.iteritems():
                    row = `c`+','+`Value['uName']`+','+`Value['uJudge']`+','+`Value['uCorrected']`+','\
                        +`Value['vNames']`+','+`Value['vValues']`
                    f.write(row+'\n')
            else:
                print '<><><> No judgment by agent '+`agent`+' t:'+`time`+' step:'+`t`

def read_and_aggregate_results(combinations,optimal_combinations,suffixes,root_result_dir='../results/'):
    columns = ['t','Magnitude','cost_type','decision_type','auction_type','valuation_type','no_resources','sample','cost','normalized_cost']
    optimal_method = ['tdindp','indp','sample_indp_12Node']
    agg_results = pd.DataFrame(columns=columns, dtype=int)

    print "\nAggregating Results"
    joinedlist = combinations + optimal_combinations
    for idx,x in enumerate(joinedlist):
        if x[4] in optimal_method:
            full_suffix = '_L'+`x[2]`+'_m'+`x[0]`+'_v'+`x[3]`
        elif x[5]=='Uniform':
            full_suffix = '_L'+`x[2]`+'_m'+`x[0]`+'_v'+`x[3]`+'_uniform_alloc'
        else:
            full_suffix = '_L'+`x[2]`+'_m'+`x[0]`+'_v'+`x[3]`+'_auction_'+x[5]+'_'+x[6] 
        
        result_dir = root_result_dir+x[4]+'_results'+full_suffix
        if os.path.exists(result_dir):    
            # Save all results to Pandas dataframe
            sample_result = INDPResults()
            for suf in suffixes:
                if os.path.exists(result_dir+"/costs_"  +`x[1]`+"_"+suf+".csv"):
                    sample_result=sample_result.from_csv(result_dir,x[1],suffix=suf)
            intitial_cost = {}
            norm_cost = 0
            for t in sample_result.results:
                for c in sample_result.cost_types:
                    if t==0:
                        norm_cost = 1.0
                        intitial_cost[c] = sample_result[t]['costs'][c]
                    elif intitial_cost[c]!=0.0:
                        norm_cost = sample_result[t]['costs'][c]/intitial_cost[c]
                    else:
                        norm_cost = -1.0
                    values = [t,x[0],c,x[4],x[5],x[6],x[3],x[1],
                            float(sample_result[t]['costs'][c]),norm_cost]
                    agg_results = agg_results.append(dict(zip(columns,values)), ignore_index=True)
            if idx%(len(joinedlist)/100+1)==0:
                update_progress(idx+1,len(joinedlist))
        else:
            sys.exit('Error: The combination or folder does not exist'+`x`) 
    update_progress(idx+1,len(joinedlist))
    return agg_results

def read_run_time(combinations,optimal_combinations,suffixes,root_result_dir='../results/'):
    columns = ['t','Magnitude','decision_type','auction_type','valuation_type','no_resources','sample','decision_time','auction_time','valuation_time']
    optimal_method = ['tdindp','indp','sample_indp_12Node']
    run_time_results = pd.DataFrame(columns=columns, dtype=int)

    print "\nReading run times"
    joinedlist = combinations + optimal_combinations
    for idx,x in enumerate(joinedlist):
        if x[4] in optimal_method:
            full_suffix = '_L'+`x[2]`+'_m'+`x[0]`+'_v'+`x[3]`
        elif x[5]=='Uniform':
            full_suffix = '_L'+`x[2]`+'_m'+`x[0]`+'_v'+`x[3]`+'_uniform_alloc'
        else:
            full_suffix = '_L'+`x[2]`+'_m'+`x[0]`+'_v'+`x[3]`+'_auction_'+x[5]+'_'+x[6] 
        
        result_dir = root_result_dir+x[4]+'_results'+full_suffix
        run_time_all = {}
        if os.path.exists(result_dir): 
            for suf in suffixes:
                run_time_file = result_dir+"/run_time_"  +`x[1]`+"_"+suf+".csv"  
                if os.path.exists(run_time_file):
                # Save all results to Pandas dataframe                
                    with open(run_time_file) as f:
                        lines=f.readlines()[1:]
                        for line in lines:
                            data=string.split(str.strip(line),",")
                            t=int(data[0])
                            run_time_all[t]=[float(data[1]),0,0]
                if x[4] not in optimal_method and x[5]!='Uniform': 
                    auction_file = result_dir+"/auctions/auctions_"  +`x[1]`+"_"+suf+".csv"  
                    if os.path.exists(auction_file):
                    # Save all results to Pandas dataframe                
                        with open(auction_file) as f:
                            lines=f.readlines()[1:]
                            for line in lines:
                                data=string.split(str.strip(line),",")
                                t=int(data[0])
                                auction_time=float(data[2*x[2]+5])
                                decision_time = run_time_all[t][0]
                                valuation_time_max = 0.0
                                for vtm in range(x[2]):
                                    if float(data[2*x[2]+5+vtm+1])>valuation_time_max:
                                        valuation_time_max= float(data[2*x[2]+5+vtm+1])                                   
                                run_time_all[t]=[decision_time,auction_time,valuation_time_max]
            for t,value in run_time_all.items():
                values = [t,x[0],x[4],x[5],x[6],x[3],x[1],value[0],value[1],value[2]]
                run_time_results = run_time_results.append(dict(zip(columns,values)), ignore_index=True)
            if idx%(len(joinedlist)/100+1)==0:
                update_progress(idx+1,len(joinedlist))
        else:
            sys.exit('Error: The combination or folder does not exist') 
    update_progress(idx+1,len(joinedlist))
    return run_time_results

def correct_tdindp_results(df,optimal_combinations):    
    # correct total cost of td-indp
    print '\nCorrecting td-INDP Results\n',
    tVector = df['t'].unique().tolist()
    for t in tVector:
        for idx,x in enumerate(optimal_combinations):
            if x[4]=='tdindp':
                rows = df[(df['t']==t)&(df['Magnitude']==x[0])&
                         (df['decision_type']=='tdindp')&(df['no_resources']==x[3])&
                         (df['sample']==x[1])]
                
                if t!=int(tVector[-1]) and t!=0:
                    rowsNext = df[(df['t']==t+1)&(df['Magnitude']==x[0])&
                     (df['decision_type']=='tdindp')&(df['no_resources']==x[3])&
                     (df['sample']==x[1])]
                    
                    nodeCost=rows[rows['cost_type']=='Node']['cost'].values
                    arcCost=rows[rows['cost_type']=='Arc']['cost'].values
                    flowCost=rowsNext[rowsNext['cost_type']=='Flow']['cost'].values
                    overSuppCost=rowsNext[rowsNext['cost_type']=='Over Supply']['cost'].values
                    underSuppCost=rowsNext[rowsNext['cost_type']=='Under Supply']['cost'].values
                    spacePrepCost=rows[rows['cost_type']=='Space Prep']['cost'].values
                    
                    totalCost = flowCost+arcCost+nodeCost+overSuppCost+underSuppCost+spacePrepCost
                    
                    df.loc[(df['t']==t)&(df['Magnitude']==x[0])&(df['decision_type']=='tdindp')&
                        (df['no_resources']==x[3])&(df['sample']==x[1])&
                        (df['cost_type']=='Total'),'cost'] = totalCost
                    
                    initial_cost=df[(df['t']==0)&(df['Magnitude']==x[0])&(df['decision_type']=='tdindp')&
                        (df['no_resources']==x[3])&(df['sample']==x[1])&
                        (df['cost_type']=='Total')]['cost'].values
                    df.loc[(df['t']==t)&(df['Magnitude']==x[0])&(df['decision_type']=='tdindp')&
                        (df['no_resources']==x[3])&(df['sample']==x[1])&
                        (df['cost_type']=='Total'),'normalized_cost'] = totalCost/initial_cost
        update_progress(t+1,len(tVector))
    return df
               
def relative_performance(df,combinations,optimal_combinations,ref_method='indp',ref_at='',ref_vt='',cost_type='Total'):    
    columns = ['Magnitude','cost_type','decision_type','auction_type','valuation_type','no_resources','sample',
               'Area_TC','Area_P','lambda_TC','lambda_P','lambda_U']
    lambda_df = pd.DataFrame(columns=columns, dtype=int)
    # Computing reference area for lambda
    # Check if the method in optimal combination is the reference method #!!!
    print '\nRef area calculation\n',
    for idx,x in enumerate(optimal_combinations):
        if x[4]==ref_method:
            rows = df[(df['Magnitude']==x[0])&(df['decision_type']==ref_method)&
                     (df['sample']==x[1])&(df['auction_type']==ref_at)&
                     (df['valuation_type']==ref_vt)&(df['no_resources']==x[3])]
                              
            if not rows.empty:
                area_TC = np.trapz(rows[rows['cost_type']==cost_type].cost[:20],dx=1)
                area_P = np.trapz(rows[rows['cost_type']=='Under Supply Perc'].cost[:20],dx=1)
                values = [x[0],cost_type,x[4],ref_at,ref_vt,x[3],x[1],area_TC,area_P,'nan','nan','nan']
                lambda_df = lambda_df.append(dict(zip(columns,values)), ignore_index=True)
                
            if idx%(len(optimal_combinations)/100+1)==0:
                update_progress(idx+1,len(optimal_combinations))
    update_progress(idx+1,len(optimal_combinations))
    
    # Computing areaa and lambda
    print '\nLambda calculation\n',
    for idx,x in enumerate(combinations+optimal_combinations):
        if x[4]!=ref_method:
            # Check if reference area exists
            cond = ((lambda_df['Magnitude']==x[0])&(lambda_df['decision_type']==ref_method)&
                (lambda_df['auction_type']==ref_at)&(lambda_df['valuation_type']==ref_vt)&
                (lambda_df['cost_type']==cost_type)&(lambda_df['sample']==x[1])&
                (lambda_df['no_resources']==x[3]))
            if not cond.any():
                sys.exit('Error:Reference type is not here! for %s m %d|resource %d' %(x[4],x[0],x[3]))    
            ref_area_TC=float(lambda_df.loc[cond==True,'Area_TC'])
            ref_area_P=float(lambda_df.loc[cond==True,'Area_P'])
            
            rows = df[(df['Magnitude']==x[0])&(df['decision_type']==x[4])&(df['sample']==x[1])&
                      (df['auction_type']==x[5])&(df['valuation_type']==x[6])&(df['no_resources']==x[3])]
            if not rows.empty:
                area_TC = np.trapz(rows[rows['cost_type']==cost_type].cost[:20],dx=1)
                area_P = np.trapz(rows[rows['cost_type']=='Under Supply Perc'].cost[:20],dx=1)
                lambda_TC = 'nan'
                lambda_P = 'nan'
                if ref_area_TC != 0.0 and area_TC != 'nan':
                    lambda_TC = (ref_area_TC-float(area_TC))/ref_area_TC
                elif area_TC == 0.0:
                    lambda_TC = 0.0
                    
                if ref_area_P != 0.0 and area_P != 'nan':
                    lambda_P = (ref_area_P-float(area_P))/ref_area_P
                elif area_P == 0.0:
                    lambda_P = 0.0
                else:
                    pass  
                values = [x[0],cost_type,x[4],x[5],x[6],x[3],x[1],area_TC,area_P,lambda_TC,lambda_P,(lambda_TC+lambda_P)/2]
                lambda_df = lambda_df.append(dict(zip(columns,values)), ignore_index=True)
            else:
                sys.exit('Error: No entry for %s %s %s m %d|resource %d,...' %(x[4],x[5],x[6],x[0],x[3]))  
                
        if idx%(len(combinations+optimal_combinations)/100+1)==0:
            update_progress(idx+1,len(combinations+optimal_combinations))
    update_progress(idx+1,len(combinations+optimal_combinations))
    
    return lambda_df

def generate_combinations(database,mags,sample,layers,no_resources,decision_type,auction_type,valuation_type,listHDadd=None,synthetic_dir=None):
    combinations = []
    optimal_combinations = []
    optimal_method = ['tdindp','indp','sample_indp_12Node']
    print '\nCombination Generation\n',
    idx=0
    no_total = len(mags)*len(sample)  
    if database=='shelby':
        if listHDadd:
            listHD = pd.read_csv(listHDadd)     
        L = len(layers)
              
        for m,s in itertools.product(mags,sample):
            if listHDadd==None or len(listHD.loc[(listHD.set == s) & (listHD.sce == m)].index):
                for rc in no_resources:
                    for dt,at,vt in itertools.product(decision_type,auction_type,valuation_type):
                        if (dt in optimal_method) and not [m,s,L,rc,dt,'',''] in optimal_combinations:
                            optimal_combinations.append([m,s,L,rc,dt,'',''])
                        elif (dt not in optimal_method) and (at not in ['Uniform']):
                            combinations.append([m,s,L,rc,dt,at,vt])
                        elif (dt not in optimal_method) and (at in ['Uniform']):
                            combinations.append([m,s,L,rc,dt,at,''])
            idx+=1
            update_progress(idx,no_total)
    elif database=='synthetic':
        # Read net configurations
        if synthetic_dir==None:
            sys.exit('Error: Provide the address of the synthetic databse')
        with open(synthetic_dir+'List_of_Configurations.txt') as f:
            config_data = pd.read_csv(f, delimiter='\t')  
        for m,s in itertools.product(mags,sample):
            config_param = config_data.iloc[m]
            L = int(config_param.loc[' No. Layers'])   
            no_resources =  int(config_param.loc[' Resource Cap'])            
            for rc in [no_resources]:
                for dt,at,vt in itertools.product(decision_type,auction_type,valuation_type):
                    if (dt in optimal_method) and not [m,s,L,rc,dt,'',''] in optimal_combinations:
                        optimal_combinations.append([m,s,L,rc,dt,'',''])
                    elif (dt not in optimal_method) and (at not in ['Uniform']):
                        combinations.append([m,s,L,rc,dt,at,vt]) 
                    elif (dt not in optimal_method) and (at in ['Uniform']):
                        combinations.append([m,s,L,rc,dt,at,''])
            idx+=1
            update_progress(idx,no_total)
    else:
        sys.exit('Error: Wrong database type')
    
    return combinations,optimal_combinations

def update_progress(progress,total):
    print '\r[%s] %1.1f%%' % ('#'*int(progress/float(total)*20), (progress/float(total)*100)),
    sys.stdout.flush()

def make_dir(dir):
    try:
        os.makedirs(dir)
    except OSError, e:
        if e.errno != os.errno.EEXIST:
            raise  
        pass