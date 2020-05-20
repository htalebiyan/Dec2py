''' Decentralized restoration for interdepndent networks'''
import os.path
import operator
import copy
import itertools
import time
import sys
import pandas as pd
import numpy as np
import gurobipy
import indp
import indputils

class JcModel:
    '''
    Description
    '''
    def __init__(self, ide, params):
        #: Basic attributes
        self.id = ide
        self.algo = self.set_algo(params['ALGORITHM'])
        self.layers = params['L']
        self.judge_type = params['JUDGMENT_TYPE']
        self.net = self.set_network(params)
        self.time_steps = self.set_time_steps(params['T'], params['NUM_ITERATIONS'])
        self.judgments = JudgmentModel(params, self.time_steps)
        #: Resource allocation attributes
        self.resource = ResourceModel(params, self.time_steps)
        self.res_alloc_type = self.resource.type
        self.v_r = self.resource.v_r
        #: Results attributes
        self.ouptut_dir = self.set_out_dir(params['OUTPUT_DIR'], params['MAGNITUDE'])
        self.results = indputils.INDPResults(self.layers)
        self.results_real = indputils.INDPResults(self.layers)
    def set_out_dir(self, root, mag):
        '''
        
        Parameters
        ----------
        root : TYPE
            DESCRIPTION.
        mag : TYPE
            DESCRIPTION.

        Returns
        -------
        ouput_dir : TYPE
            DESCRIPTION.

        '''
        ouput_dir = root+'_L'+str(len(self.layers))+'_m'+str(mag)+"_v"+\
            str(self.resource.sum_resource)+'_'+self.res_alloc_type
        if self.res_alloc_type == 'auction':
            ouput_dir += '_'+self.resource.auction_model.auction_type+\
                '_'+self.resource.auction_model.valuation_type
        return ouput_dir
    @staticmethod
    def set_algo(algorithm):
        '''

        Parameters
        ----------
        algorithm : TYPE
            DESCRIPTION.

        Returns
        -------
        str
            DESCRIPTION.

        '''
        if algorithm == 'JC':
            return 'JC'
        else:
            sys.exit('Wrong Algorithm. It should be JC.')
    def set_network(self, params):
        '''

        Parameters
        ----------
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if "N" in params:
            return copy.deepcopy(params["N"]) #!!! deepcopy
        else:
            sys.exit('No initial network object for: '+self.judge_type+', '+\
                     self.res_alloc_type)
    @staticmethod
    def set_time_steps(T, num_iter):
        '''

        Parameters
        ----------
        T : TYPE
            DESCRIPTION.
        num_iter : TYPE
            DESCRIPTION.

        Returns
        -------
        num_iter : TYPE
            DESCRIPTION.

        '''
        if T == 1:
            return num_iter
        else: #!!! to be modified in futher expansions
            sys.exit('JC currently only supports iINDP, not td_INDP.')

class JudgmentModel:
    '''
    Description
    '''
    def __init__(self, params, t_steps):
        self.judgment_type = params['JUDGMENT_TYPE']
        self.judgment = {t+1:{l:{} for l in params['L']} for t in range(t_steps)}
        self.realized = {t+1:{l:{} for l in params['L']} for t in range(t_steps)}

    def save_judgments(self, obj, func_dict, lyr, t_step):
        interdep_src = []
        for u, _, a in obj.net.G.edges(data=True):
            if a['data']['inf_data'].is_interdep and u[1] != lyr:
                interdep_src.append(u)
        for n in interdep_src:
            if (n in func_dict.keys()) and (obj.net.G.nodes[n]['data']['inf_data'].functionality == 0.0):
                self.judgment[t_step][lyr][n] = func_dict[n]
                
    def create_judgment_dict(self, obj, layers_tbj, T=1, judge_type_forced=None):
        '''
        Creates a functionality map for input into the functionality parameter in the indp function.
    
        Parameters
        ----------
        obj : JcModel instance
            DESCRIPTION.
        layers_tbj : list
            List of layers to be judged, which can be different
            from all layers of the network or controlled layers.
        T : int, optional
            DESCRIPTION. The default is 1
    
        Returns
        -------
        functionality : TYPE
            DESCRIPTION.
    
        '''
        judge_type = self.judgment_type
        if judge_type_forced:
            judge_type = judge_type_forced
        functionality = {}
        g_prime_nodes = [n[0] for n in obj.net.G.nodes(data=True) if n[1]['data']['inf_data'].net_id in layers_tbj]
        g_prime = obj.net.G.subgraph(g_prime_nodes)
        N_prime = [n for n in g_prime.nodes(data=True) if n[1]['data']['inf_data'].functionality == 0.0]
        N_prime_nodes = [n[0] for n in g_prime.nodes(data=True) if n[1]['data']['inf_data'].functionality == 0.0]
        for t in range(T):
            functionality[t] = {}
            functional_nodes = []
            # Generate resoration probabilities and corresponding bernoulli experiments
            # for demand-based and deterministic demand-based judgments
            # Updates the bernoulli experiments (not the resoration probabilities) for each t
            interdep_src = []
            det_priority = []
            if judge_type in ['DEMAND', 'DET-DEMAND']:
                priority_list = self.demand_based_priority_list(obj.net, layers_tbj)
                if judge_type == 'DET-DEMAND':
                    sorted_priority_list = sorted(priority_list.items(), key=operator.itemgetter(1),
                                                  reverse=True)
                    num_layers = 1 #len(obj.layers)+1
                    res_cap = obj.resource.sum_resource//num_layers
                    for u, _, a in obj.net.G.edges(data=True):
                        if a['data']['inf_data'].is_interdep and u[1] in layers_tbj:
                            interdep_src.append(u)
                    for i in sorted_priority_list:
                        if (i[0] in N_prime_nodes) and (len(det_priority) < (t+1)*res_cap) and\
                            (i[0] in interdep_src):
                            det_priority.append(i[0])
            # Nodes that are judged/known to be functional for t_p<t
            for t_p in range(t):
                for key in functionality[t_p]:
                    if functionality[t_p][key] == 1.0:
                        functional_nodes.append(key)
            for n, d in g_prime.nodes(data=True):
                #print "layers=", layers, "n=", n
                if d['data']['inf_data'].net_id in layers_tbj:
                    # Undamged Nodes don't get judged
                    if g_prime.has_node(n) and (n, d) not in N_prime:
                        functionality[t][n] = 1.0
                    # Nodes in functional nodes don't get judged
                    elif n in functional_nodes:
                        functionality[t][n] = 1.0
                    # Judgments
                    else:
                        if judge_type == "OPTIMISTIC":
                            functionality[t][n] = 1.0
                        elif judge_type == "PESSIMISTIC":
                            functionality[t][n] = 0.0
                        elif judge_type == "DEMAND":
                            functionality[t][n] = priority_list[n][1]
                        elif judge_type == "DET-DEMAND":
                            if n in det_priority:
                                functionality[t][n] = 1
                            else:
                                functionality[t][n] = 0.0
                        elif judge_type == "RANDOM":
                            functionality[t][n] = np.random.choice([0, 1], p=[0.5, 0.5])
                        elif judge_type == "REALISTIC":
                            functionality[t][n] = d['data']['inf_data'].functionality
                        elif judge_type == "BHM":
                            pass #!!! Add BHM Judgment
                        else:
                            if not n in functionality[t]:
                                functionality[t][n] = 0.0
        return functionality
    @staticmethod
    def demand_based_priority_list(N, layers_tbj):
        '''
        This function generates the prioirt list for the demand-based judgment
        Here, an agent judges that the dependee node in the dependee network is repaired
        until the next time step with the probability that is equal to the demand/supply
        value of the dependee node divided by the maximum demand/supply value in the dependee
        network. Also, based on the probability, a judgment is generated for the dependee node.
    
        Parameters
        ----------
        N : InfrastructureNetwork instance
            DESCRIPTION.
        layers_tbj : list
            List of layers to be judged, which can be different
            from all layers of the network or controlled layers.
    
        Returns
        -------
        prob : TYPE
            DESCRIPTION.
    
        '''
        max_values = {}
        prob = {}
        for l in layers_tbj:
            g_lyr_nodes = [n[0] for n in N.G.nodes(data=True) if n[1]['data']['inf_data'].net_id == l]
            g_lyr = N.G.subgraph(g_lyr_nodes)
            max_values[l, 'Demand'] = min([n[1]['data']['inf_data'].demand for n in g_lyr.nodes(data=True)])
            max_values[l, 'Supply'] = max([n[1]['data']['inf_data'].demand for n in g_lyr.nodes(data=True)])
            for n in g_lyr.nodes(data=True):
                prob_node = 0.5
                if not n[0] in prob.keys():
                    value = n[1]['data']['inf_data'].demand
                    if value > 0:
                        prob_node = value/max_values[l, 'Supply']
                    elif value <= 0:
                        prob_node = value/max_values[l, 'Demand']
                prob[n[0]] = [prob_node, np.random.choice([0, 1], p=[1-prob_node, prob_node])]
        return prob

class ResourceModel:
    '''
    DESCRIPTION
    '''
    def __init__(self, params, time_steps):
        self.t_steps = time_steps
        # self.time = {t+1:0.0 for t in range(self.t_steps)}
        self.v_r = {t+1:{l:0.0 for l in params['L']} for t in range(self.t_steps)}
        if params['RES_ALLOC_TYPE'] in ["MDA", "MAA", "MCA"]:
            self.type = 'auction'
            self.auction_model = AuctionModel(params, self.t_steps)
            self.sum_resource = params['V']
        elif params['RES_ALLOC_TYPE'] == "UNIFORM":
            self.type = 'uniform'
            self.set_uniform_res(params)
        elif params['RES_ALLOC_TYPE'] == "LAYER_FIXED":
            self.type = 'uniform_lf'
            self.set_lf_res(params)
        else:
            sys.exit('Unsupported resource allocation type: '+params['RES_ALLOC_TYPE'])
    def set_uniform_res(self, params):
        '''

        Parameters
        ----------
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if isinstance(params['V'], (int)):
            self.sum_resource = params['V']
            for t in range(self.t_steps):
                v_r_uni = {x:self.sum_resource//len(params['L']) for x in params['L']}
                rnd_idx = np.random.choice(params['L'], self.sum_resource%len(params['L']),
                                           replace=False)
                for x in rnd_idx:
                    v_r_uni[x] += 1
                self.v_r[t+1] = v_r_uni
        else:
            sys.exit('Number of resources is an integer for the resource allocation type UNIFORM.')
    def set_lf_res(self, params):
        '''

        Parameters
        ----------
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        try:
            len(params['V']) == len(params['L'])
        except:
            sys.exit('Length of resource vector and layer vector should be the same for '+\
                     'the resource allocation type FIXED_LAYER.')
        self.sum_resource = sum(params['V'])
        for t in range(self.t_steps):
            for l in params['L']:
                self.v_r[t+1][l] = params['V'][params['L'].index(l)]

class AuctionModel():
    '''
    DESCRIPTION
    '''
    def __init__(self, params, time_steps):
        self.auction_type = params['RES_ALLOC_TYPE']
        self.valuation_type = params['VALUATION_TYPE']
        self.bidder_type = 'truthful'
        self.winners = {t+1:{} for t in range(time_steps)}
        self.win_bids = {t+1:{} for t in range(time_steps)}
        self.win_prices = {t+1:{} for t in range(time_steps)}
        self.bids = {t+1:{l:{} for l in params['L']} for t in range(time_steps)}
        self.valuations = {t+1:{l:{} for l in params['L']} for t in range(time_steps)}
        self.valuation_time = {t+1:{l:0.0 for l in params['L']} for t in range(time_steps)}
        self.auction_time = {t+1:0.0 for t in range(time_steps)}
        self.poa = {t+1:0.0 for t in range(time_steps)}
    def bidding(self, time_step):
        '''

        Parameters
        ----------
        time_step : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        for l, lval in self.valuations[time_step].items():
            for v, vval in lval.items():
                if self.bidder_type == 'truthful':
                    self.bids[time_step][l][v] = vval
    def auction_resources(self, obj, time_step, print_cmd=True, compute_poa=False):
        '''
        allocate resources based on given types of auction and valuatoin.

        Parameters
        ----------
        obj : JcModel
            DESCRIPTION.
        time_step : int
            DESCRIPTION.
        print_cmd : bool, optional
            DESCRIPTION. The default is True.
        compute_poa : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        #: Compute Valuations
        if print_cmd:
            print('Compute Valuations: '+self.valuation_type)
        self.compute_valuations(obj, time_step, print_cmd=print_cmd,
                                compute_optimal_valuation=compute_poa)
        #: Bidding
        self.bidding(time_step)
        #: Auctioning
        if print_cmd:
            print("Auction: " +self.auction_type)
        start_time_auction = time.time()
        if self.auction_type == "MDA":
            for v in range(obj.resource.sum_resource):
                if print_cmd:
                    print('Resource '+str(v+1)+': ', end='')
                cur_bid = {l:self.bids[time_step][l][obj.resource.v_r[time_step][l]+1]\
                           for l in obj.layers}
                winner = max(cur_bid.items(), key=operator.itemgetter(1))[0]
                self.winners[time_step][v+1] = winner
                self.win_bids[time_step][v+1] = cur_bid[winner]
                self.win_prices[time_step][v+1] = cur_bid[winner]
                if cur_bid[winner] > 0:
                    if print_cmd:
                        print("Player %d wins!" % winner)
                    obj.resource.v_r[time_step][winner] += 1
                else:
                    if print_cmd:
                        print("No auction winner!")
        if self.auction_type == "MAA":
            all_bids = []
            for l, lval in self.bids[time_step].items():
                for v, vval in lval.items():
                    all_bids.append(vval)
            all_bids.sort()
            Q = obj.resource.sum_resource*len(obj.layers)
            q = {l:obj.resource.sum_resource for l in obj.layers}
            price = 0.0
            while Q > obj.resource.sum_resource:
                price = all_bids[0]
                Q = 0
                for l in obj.layers:
                    q[l] = 0
                    for x, val in self.bids[time_step][l].items():
                        if val > price:
                            q[l] += 1
                        else:
                            break
                    Q += q[l]
                all_bids = [x for x in all_bids if x != price]
            num_assigned_res = 0
            for l in obj.layers:
                obj.resource.v_r[time_step][l] = q[l]
                for v in range(q[l]):
                    if print_cmd:
                        print('Resource '+str(num_assigned_res+1)+': Player '+\
                              str(l)+' wins!')
                    self.winners[time_step][num_assigned_res+1] = l
                    self.win_bids[time_step][num_assigned_res+1] = self.bids[time_step][l][v+1]
                    self.win_prices[time_step][num_assigned_res+1] = price
                    num_assigned_res += 1
            if Q < obj.resource.sum_resource:
                for v in range(int(obj.resource.sum_resource-Q)):
                    if print_cmd:
                        print('Resource '+str(Q+v+1)+': No auction winner!')
        if self.auction_type == "MCA":
            m = gurobipy.Model('auction')
            m.setParam('OutputFlag', False)
            # Add allocation variables and populate objective function.
            for l in obj.layers:
                for v in range(obj.resource.sum_resource):
                    m.addVar(name='y_'+str(v+1)+", "+str(l), vtype=gurobipy.GRB.BINARY,
                             obj=sum([-self.bids[time_step][l][vv] for vv in range(1, v+2)]))
            m.update()
            # Add constraints
            num_alloc_res = gurobipy.LinExpr()
            for l in obj.layers:
                each_bidder_alloc = gurobipy.LinExpr()
                for v in range(obj.resource.sum_resource):
                    num_alloc_res += m.getVarByName('y_'+str(v+1)+", "+str(l))*(v+1)
                    each_bidder_alloc += m.getVarByName('y_'+str(v+1)+", "+str(l))
                m.addConstr(each_bidder_alloc, gurobipy.GRB.LESS_EQUAL, 1.0,
                            "Bidder "+str(l)+" allocation")
            m.addConstr(num_alloc_res, gurobipy.GRB.LESS_EQUAL, obj.resource.sum_resource,
                        "Total number of resources")
            m.update()
            m.optimize()
            # m.write('model.lp')
            # m.write('model.sol')
            num_assigned_res = 0
            for l in obj.layers:
                for v in range(obj.resource.sum_resource):
                    if m.getVarByName('y_'+str(v+1)+", "+str(l)).x == 1:
                        obj.resource.v_r[time_step][l] = v+1
                        for vv in range(v+1):
                            if print_cmd:
                                print('Resource '+str(num_assigned_res+1)+': Player '+\
                                      str(l)+' wins!')
                            self.winners[time_step][num_assigned_res+1] = l
                            self.win_bids[time_step][num_assigned_res+1] = \
                                self.bids[time_step][l][vv+1]
                            self.win_prices[time_step][num_assigned_res+1] = \
                                self.bids[time_step][l][vv+1]
                            num_assigned_res += 1
        self.auction_time[time_step] = time.time()-start_time_auction
        if compute_poa:
            winners_valuations = []
            for v, winner in self.winners[time_step].items():
                winners_valuations.append(self.valuations[time_step][winner][v])
            if sum(winners_valuations) != 0:
                self.poa[time_step] = self.poa[time_step]/sum(winners_valuations)
            else:
                self.poa[time_step] = 'nan'
    def compute_valuations(self, obj, t_step, print_cmd=True, compute_optimal_valuation=False):
        '''
        Computes bidders' valuations and bids for different number of resources

        Parameters
        ----------
        obj : JcModel instance
            DESCRIPTION.
        t_step : int
            DESCRIPTION.
        print_cmd : bool, optional
            DESCRIPTION. The default is True.
        compute_optimal_valuation : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        #: Calculating current total cost ###
        time_limit = 2*60 #!!! Maybe adjusted
        current_total_cost = {}
        for l in obj.layers:
            current_total_cost[l] = obj.results_real.results_layer[l][t_step-1]['costs']['Total']
        #: Optimal Valuation, which the optimal walfare value. Used to compute POA ###
        if compute_optimal_valuation:
            current_optimal_tc = obj.results_real.results[t_step-1]['costs']['Total']
            indp_results = indp.indp(obj.net, v_r=obj.resource.sum_resource, T=1,
                                     layers=obj.layers, controlled_layers=obj.layers)
            optimal_tc = indp_results[1][0]['costs']['Total']
            self.poa[t_step] = current_optimal_tc - optimal_tc
        #: Compute valuations
        for l in obj.layers:
            start_time_val = time.time()
            if print_cmd:
                print("Bidder-%d"%(l))
            if self.valuation_type == 'DTC':
                for v in range(obj.resource.sum_resource):
                    neg_layer = [x for x in obj.layers if x != l]
                    functionality = obj.judgments.create_judgment_dict(obj, neg_layer)
                    indp_results = indp.indp(obj.net, v_r=v+1, T=1, layers=obj.layers,
                                             controlled_layers=[l], functionality=functionality,
                                             print_cmd=print_cmd, time_limit=time_limit)
                    new_total_cost = indp_results[1][0]['costs']['Total']
                    if indp_results[1][0]['actions'] != []:
                        self.valuations[t_step][l][v+1] = current_total_cost[l]-new_total_cost
                        current_total_cost[l] = new_total_cost
                    else:
                        self.valuations[t_step][l][v+1] = 0.0
            elif self.valuation_type == 'DTC_uniform':
                for v in range(obj.resource.sum_resource):
                    total_cost_bounds = []
                    for jt in ["PESSIMISTIC", "OPTIMISTIC"]:
                        neg_layer = [x for x in obj.layers if x != l]
                        functionality = obj.judgments.create_judgment_dict(obj, neg_layer,
                                                                           judge_type_forced=jt)
                        indp_results = indp.indp(obj.net, v_r=v+1, T=1, layers=obj.layers,
                                                 controlled_layers=[l], functionality=functionality,
                                                 print_cmd=print_cmd, time_limit=time_limit)
                        total_cost_bounds.append(indp_results[1][0]['costs']['Total'])
                    new_total_cost = np.random.uniform(min(total_cost_bounds),
                                                       max(total_cost_bounds), 1)[0]
                    if current_total_cost[l]-new_total_cost > 0:
                        self.valuations[t_step+1][l][v+1] = current_total_cost[l]-new_total_cost
                        current_total_cost[l] = new_total_cost
                    else:
                        self.valuations[t_step][l][v+1] = 0.0
            elif self.valuation_type == 'MDDN':
                g_prime_nodes = [n[0] for n in obj.net.G.nodes(data=True)\
                                 if n[1]['data']['inf_data'].net_id == l]
                g_prime = obj.net.G.subgraph(g_prime_nodes)
                penalty_dmgd_nodes = []
                for n in g_prime.nodes(data=True):
                    if n[1]['data']['inf_data'].functionality == 0.0:
                        if n[1]['data']['inf_data'].demand > 0:
                            penalty_dmgd_nodes.append(abs(n[1]['data']['inf_data'].demand*\
                                                          n[1]['data']['inf_data'].oversupply_penalty))
                        else:
                            penalty_dmgd_nodes.append(abs(n[1]['data']['inf_data'].demand*\
                                                          n[1]['data']['inf_data'].undersupply_penalty))
                penalty_rsorted = np.sort(penalty_dmgd_nodes)[::-1]
                for v in range(obj.resource.sum_resource):
                    if v >= len(penalty_rsorted):
                        self.valuations[t_step][l][v+1] = 0.0
                    else:
                        self.valuations[t_step][l][v+1] = penalty_rsorted[v]
            self.valuation_time[t_step][l] = time.time()-start_time_val