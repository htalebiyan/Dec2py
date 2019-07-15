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

def run_judgment_call(params,layers=[1,2,3],T=1,saveJC=True,print_cmd=True,saveJCModel=False,validate=False):
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
    if auction_type=="second_price":
        output_dir = params["OUTPUT_DIR"]+'_m'+`params["MAGNITUDE"]`+"_v"+`sum(v_r)`+'_auction_layer_cap'
    elif auction_type=="second_price_uniform":
        output_dir = params["OUTPUT_DIR"]+'_m'+`params["MAGNITUDE"]`+"_v"+`sum(v_r)`+'_auction_layer_cap_uniform'
    else:
        output_dir = params["OUTPUT_DIR"]+'_m'+`params["MAGNITUDE"]`+"_v"+`sum(v_r)`+'_fixed_layer_cap'
    
    Dindp_results={P:INDPResults() for P in layers}   
    Dindp_results_Real={P:INDPResults() for P in layers} 
    currentTotalCost = {}
    if T == 1: 
        print "Running Judgment Call with type "+judgment_type #+" and "+auction_type #!!!
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
        for i in range(num_iterations):
            print "\n--Iteration "+`i`+"/"+`num_iterations-1`
            
            v_r_applied = []
            if auction_type:
                res_allocate[i],PoA[i]=auction_resources(sum(v_r),params,currentTotalCost,
                    layers=layers,T=1,print_cmd=print_cmd,judgment_type=judgment_type,
                    auction_type=auction_type)
               
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
                                controlled_layers=[P],functionality= functionality[P])
                        
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
                
                if auction_type:
                    currentTotalCost[P]=indp_results_Real[1][0]['costs']['Total']
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
            write_auction_csv(output_dir_auction,res_allocate,PoA,sample_num=params["SIM_NUMBER"],suffix="")    
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
                                            T=1,layers=[1,2,3],controlled_layers=[1],
                                            print_cmd=False,saveJCModel=False):
    
    G_prime_nodes = [n[0] for n in N.G.nodes_iter(data=True) if n[1]['data']['inf_data'].net_id in layers]
    G_prime = N.G.subgraph(G_prime_nodes)
    # Nodes in controlled network.
    N_hat_nodes   = [n[0] for n in G_prime.nodes_iter(data=True) if n[1]['data']['inf_data'].net_id in controlled_layers]
#    N_hat = G_prime.subgraph(N_hat_nodes)
    
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
                     
    indp_results_Real = indp(N,0,1,layers=layers,controlled_layers=controlled_layers,functionality=functionality_realized)  
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

def auction_resources(v_r,params,currentTotalCost={},layers=[1,2,3],T=1,print_cmd=True,judgment_type="OPTIMISTIC",auction_type="second_price"):
    # Initialize failure scenario.
    InterdepNet=None
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=params['SIM_NUMBER'],magnitude=params["MAGNITUDE"])
    else:
        InterdepNet=params["N"]
        
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"]=1
 
    resource_allocation = {P:[] for P in layers}
    num_iterations = params["NUM_ITERATIONS"]
    
    if not currentTotalCost:
        if print_cmd:
            print "Calculating current total cost..."
        for P in layers:     
            '''!!! check what v_r must be for det demand JC'''
            indp_results = indp(InterdepNet,v_r=0,T=1,layers=layers,
                                controlled_layers=[P])
            currentTotalCost[P] = indp_results[1][0]['costs']['Total']
            
           
    indp_results = indp(InterdepNet,v_r=0,T=1,layers=layers,
                                controlled_layers=layers)
    optimal_total_cost_current = indp_results[1][0]['costs']['Total']
    indp_results = indp(InterdepNet,v_r=v_r,T=1,layers=layers,
                                controlled_layers=layers)
    optimal_total_cost = indp_results[1][0]['costs']['Total']
    optimal_valuation = optimal_total_cost_current - optimal_total_cost

    PoA = {}
    PoA['optimal'] = optimal_valuation  
    PoA['winner'] = []          
    if T == 1: 
        if print_cmd:
            print "Auction: "
        sum_valuation = 0
        for v in range(v_r):
            newTotalCost={}
            valuation={}
            indp_results={}
            if print_cmd:
                print "Resource-%d"%(v+1)
                
            if auction_type=='second_price_uniform':
                for P in layers:
                    totalCostBounds = []
                    for jt in ["PESSIMISTIC","OPTIMISTIC"]:
                        negP=[x for x in layers if x != P]
                        functionality = create_judgment_matrix(InterdepNet,T,negP,v_r,
                                                actions=None,judgment_type=jt) 
                        '''!!! check what v_r must be for det demand JC'''
                        indp_results = indp(InterdepNet,v_r=len(resource_allocation[P])+1,
                                    T=1,layers=layers,controlled_layers=[P],functionality=functionality)
                        totalCostBounds.append(indp_results[1][0]['costs']['Total'])
                    newTotalCost[P] = np.random.uniform(min(totalCostBounds),
                                                    max(totalCostBounds),1)[0]
                    if indp_results[1][0]['actions']!=[]:
                        valuation[P] = currentTotalCost[P]-newTotalCost[P]
                    else:
                        valuation[P] = 0.0
            else:
                for P in layers:
                    negP=[x for x in layers if x != P]
                    functionality = create_judgment_matrix(InterdepNet,T,negP,v_r,
                                            actions=None,judgment_type=judgment_type) 
                    '''!!! check what v_r must be for det demand JC'''
                    indp_results = indp(InterdepNet,v_r=len(resource_allocation[P])+1,
                                T=1,layers=layers,controlled_layers=[P],functionality=functionality)
                    newTotalCost[P] = indp_results[1][0]['costs']['Total']
                    if indp_results[1][0]['actions']!=[]:
                        valuation[P] = currentTotalCost[P]-newTotalCost[P]
                    else:
                        valuation[P] = 0.0
            
            winner = max(valuation.iteritems(), key=operator.itemgetter(1))[0]
            PoA['winner'].append(valuation[winner])
            if valuation[winner]!=0:
                if print_cmd:
                    print "Player %d wins!" % winner
                sum_valuation += valuation[winner]
                resource_allocation[winner].append(v+1)            
                currentTotalCost[winner] = newTotalCost[winner]  
            else:
                if print_cmd:
                    print "No auction winner!"
        if sum_valuation!=0:
            PoA['poa'] = optimal_valuation/sum_valuation
        else:
            PoA['poa'] = -10
    else:
        print 'hahahaha'
        
    return resource_allocation,PoA

def write_auction_csv(outdir,res_allocate,PoA,sample_num=1,suffix=""):
    if not os.path.exists(outdir):
        os.makedirs(outdir)        
    auction_file=outdir+"/auctions_"+`sample_num`+"_"+suffix+".csv"
    header = "t,"
    for key,value in res_allocate[0].items():
        header += "P"+`key`+","
    header += "PoA,optimal_val,winner_val"    
    with open(auction_file,'w') as f:
        f.write(header+"\n")
        for t,value in res_allocate.items():
            row = `t+1`+","
            for p,value2 in value.items():
                row += `len(value2)`+','
            row += `PoA[t]['poa']`+','+`PoA[t]['optimal']`+','+`PoA[t]['winner']`
            f.write(row+"\n")
            
def resourcec_allocation(df,sample_range,T=1,L=3,layers=[1,3],suffix="",ci=None,listHDadd=None):
    no_resources = df.no_resources.unique().tolist()
    sce_range= df.Magnitude.unique().tolist()
    method_name = df.method.unique().tolist()
    auction_types = df.resource_cap.unique().tolist()
    if listHDadd:
        listHD = pd.read_csv(listHDadd)   
    
    cols=['t','resource','method','sample','sce','layer','no_res','PoA']
    df_res = pd.DataFrame(columns=cols)
    optimal_method = ['tdindp_results','indp_results','sample_indp_2-1_samp2']
    for nr in no_resources:
        for mn in method_name:
            for sce in sce_range:
                for i in sample_range:
                    if listHDadd==None or len(listHD.loc[(listHD.set == i) & (listHD.sce == sce)].index):                        
                        if mn in optimal_method:
                            compare_to_dir= '../results/'+mn+'_L'+`L`+'_m'+str(sce)+'_v'+str(nr)
                            for t in range(T):
                                for P in range(len(layers)):
                                    df_res=df_res.append({'t':t+1,'resource':0.0,'method':mn,'sample':i,'sce':sce,'layer':`P+1`,'no_res':nr,'PoA':1}, ignore_index=True)
                            action_file=compare_to_dir+"/actions_"+str(i)+"_"+suffix+".csv"                
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
                                        df_res.loc[(df_res['t']==t)&(df_res['method']==mn)&(df_res['sample']==i)&(df_res['sce']==sce)&(df_res['layer']==`P`)&(df_res['no_res']==nr),'resource']+=addition        
                        else:   
                            for at in auction_types:
                                outdir= '../results/'+mn+'_L'+`L`+'_m'+str(sce)+'_v'+str(nr)+at+'/auctions/'
                                auction_file=outdir+"/auctions_"+str(i)+"_"+suffix+".csv"
                                if os.path.isfile(auction_file):
                                    with open(auction_file) as f:
                                        lines=f.readlines()[1:]
                                        for line in lines:
                                            data=string.split(str.strip(line),",")
                                            t=int(data[0])
                                            for P in range(len(layers)):
                                                poa = float(data[len(layers)+1])
                                                df_res=df_res.append({'t':t,'resource':float(data[P+1]),'method':mn+'_'+at,'sample':i,'sce':sce,'layer':`P+1`,'no_res':nr,'PoA':poa}, ignore_index=True)
            print 'Resource allocation|%s|no_res:%d' %(mn,nr)
    return df_res

def plot_auction_allocation(df_res,ci=None):
    no_resources = df_res.no_res.unique().tolist()
    no_methods = len(df_res.method.unique().tolist())
    T = len(df_res.t.unique().tolist())
    for nr in no_resources:        
        plt.figure()           
        ax = sns.lineplot(x='t',y='resource',style='layer',hue='method',
                          data=df_res[(df_res['no_res']==nr)],
                          ci=ci, palette=sns.color_palette("muted",no_methods))                        
        ax.set(xticks=np.arange(0,T+1,1))
        ax.set_title(r'Total resources = %d'%(nr))
        
        plt.figure()         
        ax = sns.barplot(x="t", y="PoA", hue='method', data=df_res[(df_res['no_res']==nr)&(df_res['PoA']!=0.0)&(df_res['PoA']!=-10)])
        ax.set_title(r'Total resources = %d'%(nr))
        ax.set_ylim(0,min(10,ax.get_ylim()[1]))

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

def read_and_aggregate_results(mags,method_name,resource_cap,suffixes,L,sample_range,no_resources=[3],listHDadd=None):
    columns = ['t','Magnitude','cost_type','method','resource_cap','no_resources','sample','cost']
    agg_results = pd.DataFrame(columns=columns)
    if listHDadd:
        listHD = pd.read_csv(listHDadd)        
    for m in mags:
        for i in range(len(method_name)):
            for rc in no_resources:
                full_suffix = '_L'+`L`+'_m'+`m`+'_v'+`rc`+resource_cap[i]
                result_dir = '../results/'+method_name[i]+full_suffix
                
#                # Save average values to file #!!!!!!!!!!!
#                results_average = INDPResults()
#                results_average = results_average.from_results_dir(outdir=result_dir,
#                                    sample_range=sample_range,suffix=suffixes[i])
#                
#                outdir = '../results/average_cost_all/'
#                if not os.path.exists(outdir):
#                    os.makedirs(outdir) 
#                costs_file =outdir+method_name[i]+full_suffix+"_average_costs.csv"
#                with open(costs_file,'w') as f:
#                    f.write("t,Space Prep,Arc,Node,Over Supply,Under Supply,Flow,Total\n")
#                    for t in results_average.results:
#                        costs=results_average.results[t]['costs']
#                        f.write(`t`+","+`costs["Space Prep"]`+","+`costs["Arc"]`+","
#                                +`costs["Node"]`+","+`costs["Over Supply"]`+","+
#                                `costs["Under Supply"]`+","+`costs["Flow"]`+","+
#                                `costs["Total"]`+"\n")            
                        
                # Save all results to Pandas dataframe
                sample_result = INDPResults()
                for s in sample_range:
                    if listHDadd==None or len(listHD.loc[(listHD.set == s) & (listHD.sce == m)].index):
                        sample_result=sample_result.from_csv(result_dir,s,suffix=suffixes[i])
                        for t in sample_result.results:
                            for c in sample_result.cost_types:
                                values = [t,m,c,method_name[i],resource_cap[i],rc,s,
                                        float(sample_result[t]['costs'][c])]
                                agg_results = agg_results.append(dict(zip(columns,values)), ignore_index=True)
            print 'm %d|%s|Aggregated' %(m,method_name[i])  
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
    
def plot_performance_curves(df,x='t',y='cost',cost_type='Total',method_name=['tdindp_results'],ci=None):
    sns.set()
    no_resources = df.no_resources.unique().tolist()
    T = len(df[x].unique().tolist())
    for nr in no_resources:
        plt.figure()
        with sns.color_palette("muted"):
            ax = sns.lineplot(x=x, y=y, hue="method", style='resource_cap',
                markers=False, ci=ci,
                data=df[(df['cost_type']==cost_type)&
                        (df['method'].isin(method_name))&
                        (df['no_resources']==nr)])              
            ax.set_title(r'Total resources = %d'%(nr))
            ax.set(xticks=np.arange(0,T+1,1))
            
def relative_performance(df,sample_range,cost_type='Total',listHDadd=None):    
    sns.set()
    resource_cap = df.resource_cap.unique().tolist()
    no_resources = df.no_resources.unique().tolist()
    mags=df.Magnitude.unique().tolist()
    method_name = df.method.unique().tolist()
    columns = ['Magnitude','cost_type','method','resource_cap','no_resources','sample','Area','lambda_TC']
    lambda_df = pd.DataFrame(columns=columns)
    if listHDadd:
        listHD = pd.read_csv(listHDadd)    
    
    for m in mags:
        for jc in method_name:
            for rc in resource_cap:
                for sr in sample_range:
                    for nr in no_resources:
                        if listHDadd==None or len(listHD.loc[(listHD.set == sr) & (listHD.sce == m)].index):
                            rows = df[(df['Magnitude']==m)&(df['method']==jc)&
                                     (df['sample']==sr)&(df['cost_type']==cost_type)&
                                     (df['resource_cap']==rc)&(df['no_resources']==nr)]
                              
                            if not rows.empty:
                                area = np.trapz(rows.cost[:20],dx=1)
                            else:
                                area = 'nan'
                            
                            tempdf = pd.Series()
                            tempdf['Magnitude'] = m
                            tempdf['cost_type'] = cost_type
                            tempdf['method'] = jc
                            tempdf['resource_cap'] = rc   
                            tempdf['no_resources'] = nr 
                            tempdf['sample'] = sr  
                            tempdf['Area'] = area  
                            lambda_df=lambda_df.append(tempdf,ignore_index=True)
        print 'm %d|lambda_TC calculated' %(m)  
          
    ref_method = 'sample_indp_2-1_samp2' #!!! 'tdindp_results'
    ref_rc = '' #!!! 'Network Cap'
    for m in mags:
        for nr in no_resources:
            cond = ((lambda_df['Magnitude']==m)&(lambda_df['method']==ref_method)&
                            (lambda_df['resource_cap']==ref_rc)&
                            (lambda_df['cost_type']==cost_type)&
                            (lambda_df['no_resources']==nr)).any()
            if not cond:
                print 'Reference type is not here! for m %d|resource %d' %(m,nr)
                break
            for sr in sample_range: 
                if listHDadd==None or len(listHD.loc[(listHD.set == sr) & (listHD.sce == m)].index):
                    ref_area=float(lambda_df.loc[(lambda_df['Magnitude']==m)&
                                    (lambda_df['method']==ref_method)&
                                    (lambda_df['resource_cap']==ref_rc)&
                                    (lambda_df['sample']==sr)&
                                    (lambda_df['cost_type']==cost_type)&
                                    (lambda_df['no_resources']==nr),'Area'].values)
                    for jc in method_name:
                        for rc in resource_cap:
                            area = lambda_df.loc[(lambda_df['Magnitude']==m)&
                                    (lambda_df['method']==jc)&
                                    (lambda_df['resource_cap']==rc)&
                                    (lambda_df['sample']==sr)&
                                    (lambda_df['cost_type']==cost_type)&
                                    (lambda_df['no_resources']==nr),'Area'].values
                                
                            lambda_TC = 'nan'
                            if ref_area != 0.0 and area != 'nan':
                                lambda_TC = (ref_area-float(area))/ref_area
                            elif area == 0.0:
                                lambda_TC = 0.0
                            else:
                                pass
                            
                            lambda_df.loc[(lambda_df['Magnitude']==m)&
                                    (lambda_df['method']==jc)&
                                    (lambda_df['resource_cap']==rc)&
                                    (lambda_df['sample']==sr)&
                                    (lambda_df['cost_type']==cost_type)&
                                    (lambda_df['no_resources']==nr),'lambda_TC']=lambda_TC
    return lambda_df

def plot_relative_performance(lambda_df,cost_type='Total'):    
    sns.set()
    no_resources = lambda_df.no_resources.unique().tolist()    

    for nr in no_resources:                                                  
        plt.figure()
#            plt.rc('text', usetex=True)
#            plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        color = sns.color_palette("RdYlGn", 7) #sns.color_palette("YlOrRd", 7)
        ax = sns.barplot(x='resource_cap',y='lambda_TC',hue="method",
                         data=lambda_df[(lambda_df['cost_type']==cost_type)&
                                        (lambda_df['lambda_TC']!='nan')&
                                        (lambda_df['no_resources']==nr)], 
                         palette=color, linewidth=0.5,edgecolor=[.25,.25,.25],
                        capsize=.05,errcolor=[.25,.25,.25],errwidth=1)  
             
        ax.grid(which='major', axis='y', color=[.75,.75,.75], linewidth=.75)
        ax.set_xlabel(r'Resource Distribution Method')
        ax.set_ylabel(r'Mean Relative Measure, E[$\lambda_{%s}$]'%('TC'))
        ax.xaxis.set_label_position('bottom')  
        ax.set_title(r'Total resources = %d'%(nr))
#        ax.xaxis.tick_top()
        ax.set_facecolor('w')   
        plt.legend(handles=ax.get_legend_handles_labels()[0][:7],loc=0,frameon =True,
                   framealpha=0.0, ncol=1,bbox_to_anchor=(1.1, 0.1))   