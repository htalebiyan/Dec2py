# %%
''' 
Classes for modeling Judgment Call method and Auction-based resource 
:cite:`Talebiyan2019c,Talebiyan2019`.
'''
import os.path
import operator
import copy
import time
import sys
import pickle
import numpy as np
import gurobipy
import indp
import indputils
import stm
import indpalt


# %%
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
        self.output_dir = self.set_out_dir(params['OUTPUT_DIR'], params['MAGNITUDE'])
        self.results_judge = indputils.INDPResults(self.layers)
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
        output_dir : TYPE
            DESCRIPTION.

        '''
        outDirSuffixRes = ''
        for rc, val in self.resource.sum_resource.items():
            if isinstance(val, (int)):
                outDirSuffixRes += rc[0] + str(val)
            else:
                outDirSuffixRes += rc[0] + str(sum([lval for _, lval in val.items()])) + '_fixed_layer_Cap'

        output_dir = root + '_L' + str(len(self.layers)) + '_m' + str(mag) + "_v" + \
                     outDirSuffixRes + '_' + self.judge_type + '_' + self.res_alloc_type
        if self.res_alloc_type == 'AUCTION':
            a_model = next(iter(self.resource.auction_model.values()))
            output_dir += '_' + a_model.auction_type + '_' + a_model.valuation_type
        return output_dir

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
        if algorithm != 'JC':
            sys.exit('Wrong Algorithm. It should be JC.')
        else:
            return 'JC'

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
        assert "N" in params, 'No initial network object for: ' + self.judge_type + \
                              ', ' + params['RES_ALLOC_TYPE']
        return copy.deepcopy(params["N"])  # !!! deepcopy

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
        if T != 1:  # !!! to be modified in futher expansions
            sys.exit('ERROR: T!=1, JC currently only supports iINDP, not td_INDP.')
        else:
            return num_iter

    def recal_result_sum(self, t_step):
        '''

        Parameters
        ----------
        t_step : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # compute total demand of all layers and each layer
        total_demand = 0.0
        total_demand_layer = {l: 0.0 for l in self.layers}
        for n, d in self.net.G.nodes(data=True):
            demand_value = d['data']['inf_data'].demand
            if demand_value < 0 and n[1] in self.layers:
                total_demand += demand_value
                total_demand_layer[n[1]] += demand_value

        max_run_time = 0.0
        max_rtime_r = 0.0
        for cost_type in self.results_judge.cost_types:
            sum_temp = 0.0
            sum_temp_r = 0.0
            for l in self.layers:
                if cost_type != 'Under Supply Perc':
                    sum_temp += self.results_judge.results_layer[l][t_step]['costs'][cost_type]
                    sum_temp_r += self.results_real.results_layer[l][t_step]['costs'][cost_type]
                else:
                    factor = total_demand_layer[l] / total_demand
                    sum_temp += self.results_judge.results_layer[l][t_step]['costs'][cost_type] * factor
                    sum_temp_r += self.results_real.results_layer[l][t_step]['costs'][cost_type] * factor
            self.results_judge.add_cost(t_step, cost_type, sum_temp)
            self.results_real.add_cost(t_step, cost_type, sum_temp_r)
        for l in self.layers:
            if self.results_judge.results_layer[l][t_step]['run_time'] > max_run_time:
                max_run_time = self.results_judge.results_layer[l][t_step]['run_time']
            if self.results_real.results_layer[l][t_step]['run_time'] > max_rtime_r:
                max_rtime_r = self.results_real.results_layer[l][t_step]['run_time']
            for a in self.results_judge.results_layer[l][t_step]['actions']:
                self.results_judge.add_action(t_step, a, save_layer=False)
                self.results_real.add_action(t_step, a, save_layer=False)
        self.results_judge.add_run_time(t_step, max_run_time, save_layer=False)
        self.results_real.add_run_time(t_step, max_rtime_r, save_layer=False)

    def save_results_to_file(self, sample):
        '''

        Parameters
        ----------
        sample : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        output_dir_agents = self.output_dir + '/agents'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(output_dir_agents):
            os.makedirs(output_dir_agents)
        self.results_judge.to_csv(self.output_dir, sample)
        self.results_real.to_csv(self.output_dir, sample, suffix='real')
        self.results_judge.to_csv_layer(output_dir_agents, sample)
        self.results_real.to_csv_layer(output_dir_agents, sample, suffix='real')

    def save_object_to_file(self, sample):
        '''

        Parameters
        ----------
        sample : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self.output_dir + '/objs_' + str(sample) + '.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def correct_results_real(self, lyr, t_step):
        '''

        Parameters
        ----------
        lyr : TYPE
            DESCRIPTION.
        t_step : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        rslts = self.results_judge.results_layer[lyr][t_step]
        rslts_real = self.results_real.results_layer[lyr][t_step]
        rslts_real['costs']["Node"] = rslts['costs']["Node"]
        rslts_real['costs']["Arc"] = rslts['costs']["Arc"]
        rslts_real['costs']["Space Prep"] = rslts['costs']["Space Prep"]
        rslts_real['actions'] = [x for x in rslts['actions']]


class JudgmentModel:
    '''
    Description
    '''

    def __init__(self, params, t_steps):
        self.judgment_type = params['JUDGMENT_TYPE']
        self.judged_nodes = {t + 1: {l: {} for l in params['L']} for t in range(t_steps)}
        self.dest_nodes = {t + 1: {l: {} for l in params['L']} for t in range(t_steps)}

    def save_judgments(self, obj, judge_dict, lyr, t_step):
        '''
        only for damage dependee nodes and their depndents
        Parameters
        ----------
        obj : TYPE
            DESCRIPTION.
        judge_dict : TYPE
            DESCRIPTION.
        lyr : TYPE
            DESCRIPTION.
        t_step : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        for u, v, a in obj.net.G.edges(data=True):
            if a['data']['inf_data'].is_interdep and u[1] != lyr and v[1] == lyr and u[1] in obj.layers:
                if obj.net.G.nodes[u]['data']['inf_data'].functionality == 0.0:
                    self.judged_nodes[t_step][lyr][u] = [judge_dict[0][u]]
                    # !!! td-INDP: 0 in judge_dict[0] should be replaced
                    self.dest_nodes[t_step][lyr][v] = [u]

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
        g_prime_nodes = [n[0] for n in obj.net.G.nodes(data=True) \
                         if n[1]['data']['inf_data'].net_id in layers_tbj]
        g_prime = obj.net.G.subgraph(g_prime_nodes)
        N_prime = [n for n in g_prime.nodes(data=True) \
                   if n[1]['data']['inf_data'].functionality == 0.0]
        N_prime_nodes = [n[0] for n in g_prime.nodes(data=True) \
                         if n[1]['data']['inf_data'].functionality == 0.0]
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
                    num_layers = 1  # len(obj.layers)+1
                    res_cap = obj.resource.sum_resource // num_layers
                    for u, _, a in obj.net.G.edges(data=True):
                        if a['data']['inf_data'].is_interdep and u[1] in layers_tbj:
                            interdep_src.append(u)
                    for i in sorted_priority_list:
                        if (i[0] in N_prime_nodes) and (len(det_priority) < (t + 1) * res_cap) and \
                                (i[0] in interdep_src):
                            det_priority.append(i[0])
            # Nodes that are judged/known to be functional for t_p<t
            for t_p in range(t):
                for key in functionality[t_p]:
                    if functionality[t_p][key] == 1.0:
                        functional_nodes.append(key)
            for n, d in g_prime.nodes(data=True):
                # print "layers=", layers, "n=", n
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
                            pass  # !!! Add BHM Judgment
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
            g_lyr_nodes = [n[0] for n in N.G.nodes(data=True) \
                           if n[1]['data']['inf_data'].net_id == l]
            g_lyr = N.G.subgraph(g_lyr_nodes)
            max_values[l, 'Demand'] = min([n[1]['data']['inf_data'].demand \
                                           for n in g_lyr.nodes(data=True)])
            max_values[l, 'Supply'] = max([n[1]['data']['inf_data'].demand \
                                           for n in g_lyr.nodes(data=True)])
            for n in g_lyr.nodes(data=True):
                prob_node = 0.5
                if not n[0] in prob.keys():
                    value = n[1]['data']['inf_data'].demand
                    if value > 0:
                        prob_node = value / max_values[l, 'Supply']
                    elif value <= 0:
                        prob_node = value / max_values[l, 'Demand']
                prob[n[0]] = [prob_node, np.random.choice([0, 1], p=[1 - prob_node, prob_node])]
        return prob


# %%
class ResourceModel:
    '''
    DESCRIPTION
    '''

    def __init__(self, params, time_steps):
        self.t_steps = time_steps
        self.time = {t + 1: 0.0 for t in range(self.t_steps)}
        self.v_r = {t + 1: {l: {} for l in params['L']} for t in range(self.t_steps)}
        if params['RES_ALLOC_TYPE'] in ["MDA", "MAA", "MCA"]:
            self.type = 'AUCTION'
            self.auction_model = {}
            for key in params['V'].keys():
                self.auction_model[key] = AuctionModel(params, self.t_steps, key)
            self.sum_resource = params['V']
        elif params['RES_ALLOC_TYPE'] == "OPTIMAL":
            self.type = 'OPTIMAL'
            self.set_optimal_res(params)
            self.sum_resource = params['V']
        elif params['RES_ALLOC_TYPE'] == "UNIFORM":
            self.type = 'UNIFORM'
            self.set_uniform_res(params)
        elif params['RES_ALLOC_TYPE'] == "LAYER_FIXED":
            self.type = 'LAYER_FIXED'
            self.set_lf_res(params)
        else:
            sys.exit('Unsupported resource allocation type: ' + str(params['RES_ALLOC_TYPE']))

    def set_optimal_res(self, params):
        '''
        Parameters
        ----------
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        results_folder = params['OUTPUT_DIR'].replace(params['OUTPUT_DIR'].split('/')[-1], "")
        action_file = results_folder + 'indp_results' + '_L' + str(len(params["L"])) + '_m' + \
                      str(params["MAGNITUDE"]) + "_v" + str(params["V"]) + '/actions_' + str(
            params["SIM_NUMBER"]) + '_.csv'
        if os.path.isfile(action_file):
            with open(action_file) as f:
                lines = f.readlines()[1:]
                for line in lines:
                    data = line.strip().split(',')
                    t = int(data[0])
                    action = str.strip(data[1])
                    l = int(action[-1])
                    if '/' in action:
                        addition = 0.5
                    else:
                        addition = 1.0
                    self.v_r[t][l] += addition
            for t, val in self.v_r.items():
                for l, val_l in val.items():
                    self.v_r[t][l] = int(val_l)
        else:
            sys.exit('No optimal action file for the resource allocation type OPTIMAL.' + \
                     ' Mag ' + str(params["MAGNITUDE"]) + ' Sample ' + str(params["SIM_NUMBER"]) + \
                     ' Rc ' + str(params["V"]))

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
        self.sum_resource = params['V']
        for key, val in params['V'].items():
            assert isinstance(val, (int)), 'Number of resources should be an integer\
                    for the resource allocation type UNIFORM and resource ' + str(key) + ':' + str(val)
            for t in range(self.t_steps):
                v_r_uni = {x: val // len(params['L']) for x in params['L']}
                rnd_idx = np.random.choice(params['L'], val % len(params['L']),
                                           replace=False)
                for x in rnd_idx:
                    v_r_uni[x] += 1
                for l in params['L']:
                    self.v_r[t + 1][l][key] = v_r_uni[l]

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
            sys.exit('Length of resource vector and layer vector should be the same for ' + \
                     'the resource allocation type FIXED_LAYER.')
        self.sum_resource = sum(params['V'])
        for t in range(self.t_steps):
            for l in params['L']:
                self.v_r[t + 1][l] = int(params['V'][params['L'].index(l)])


# %%
class AuctionModel():
    '''
    DESCRIPTION
    '''

    def __init__(self, params, time_steps, resource_name):
        self.auction_type = params['RES_ALLOC_TYPE']
        self.resource_name = resource_name
        self.resource_unit = self.set_resource_unit(resource_name)
        self.valuation_type = params['VALUATION_TYPE']
        self.bidder_type = 'truthful'  # !!! refine truthfulness assumption
        self.winners = {t + 1: {} for t in range(time_steps)}
        self.win_bids = {t + 1: {} for t in range(time_steps)}
        self.win_prices = {t + 1: {} for t in range(time_steps)}
        self.bids = {t + 1: {l: {} for l in params['L']} for t in range(time_steps)}
        self.valuations = {t + 1: {l: {} for l in params['L']} for t in range(time_steps)}
        self.valuation_time = {t + 1: {l: 0.0 for l in params['L']} for t in range(time_steps)}
        self.auction_time = {t + 1: 0.0 for t in range(time_steps)}
        self.poa = {t + 1: 0.0 for t in range(time_steps)}
        if self.valuation_type == 'STM':
            self.stm_pred_dict = params['STM_MODEL_DICT']

    def set_resource_unit(self, resource_name):
        '''

        Parameters
        ----------
        time_step : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if resource_name == 'budget':  # !!! Just for NIST
            return 20000
        elif resource_name == 'time':
            return 7
        else:
            return 1

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
        obj : :class:`~.JcModel` or :class:`~gameclasses.InfrastructureGame`
            The object that stores the overall decentralized method for which the
            resource allocation is computed using :class:`~dindpclasses.AuctionModel`.
            The object type that can be passed here should have five attributes: 
            net, layers, results (or results_real for :class:`~.JcModel`), and
            judgments (for computing the valuations).
        time_step : int
            Time step for which the auction is computed and resoures are allocated.
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
            print('Compute Valuations: ', self.resource_name, self.valuation_type)
        self.compute_valuations(obj, time_step, print_cmd=print_cmd,
                                compute_optimal_valuation=compute_poa)
        #: Bidding
        self.bidding(time_step)
        #: Auctioning
        if print_cmd:
            print("Auction: ", self.resource_name, self.auction_type)
        start_time_auction = time.time()
        num_units = obj.resource.sum_resource[self.resource_name] // self.resource_unit
        if self.auction_type == "MDA":  # !!! adapt for multi resource
            for v in range(obj.resource.sum_resource):
                if print_cmd:
                    print('Resource ' + str(v + 1) + ': ', end='')
                cur_bid = {l: self.bids[time_step][l][obj.resource.v_r[time_step][l] + 1] \
                           for l in obj.layers}
                winner = max(cur_bid.items(), key=operator.itemgetter(1))[0]
                self.winners[time_step][v + 1] = winner
                self.win_bids[time_step][v + 1] = cur_bid[winner]
                self.win_prices[time_step][v + 1] = cur_bid[winner]
                if cur_bid[winner] > 0:
                    if print_cmd:
                        print("Player %d wins!" % winner)
                    obj.resource.v_r[time_step][winner] += 1
                else:
                    if print_cmd:
                        print("No auction winner!")
        if self.auction_type == "MAA":  # !!! adapt for multi resource
            all_bids = []
            for l, lval in self.bids[time_step].items():
                for v, vval in lval.items():
                    all_bids.append(vval)
            all_bids.sort()
            Q = obj.resource.sum_resource * len(obj.layers)
            q = {l: obj.resource.sum_resource for l in obj.layers}
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
                obj.resource.v_r[time_step][l] = int(q[l])
                for v in range(q[l]):
                    if print_cmd:
                        print('Resource ' + str(num_assigned_res + 1) + ': Player ' + \
                              str(l) + ' wins!')
                    self.winners[time_step][num_assigned_res + 1] = l
                    self.win_bids[time_step][num_assigned_res + 1] = self.bids[time_step][l][v + 1]
                    self.win_prices[time_step][num_assigned_res + 1] = price
                    num_assigned_res += 1
            if Q < obj.resource.sum_resource:
                for v in range(int(obj.resource.sum_resource - Q)):
                    if print_cmd:
                        print('Resource ' + str(Q + v + 1) + ': No auction winner!')
        if self.auction_type == "MCA":
            m = gurobipy.Model('auction')
            m.setParam('OutputFlag', False)
            # Add allocation variables and populate objective function.
            for l in obj.layers:
                for v in range(num_units):
                    m.addVar(name='y_' + str((v + 1) * self.resource_unit) + ", " + str(l),
                             vtype=gurobipy.GRB.BINARY,
                             obj=sum([-self.bids[time_step][l][vv * self.resource_unit] for vv in range(1, v + 2)]))
            m.update()
            # Add constraints
            num_alloc_res = gurobipy.LinExpr()
            for l in obj.layers:
                each_bidder_alloc = gurobipy.LinExpr()
                for v in range(num_units):
                    num_alloc_res += m.getVarByName('y_' + str((v + 1) * self.resource_unit) + ", " + str(l)) * (
                                v + 1) * self.resource_unit
                    each_bidder_alloc += m.getVarByName('y_' + str((v + 1) * self.resource_unit) + ", " + str(l))
                m.addConstr(each_bidder_alloc, gurobipy.GRB.LESS_EQUAL, 1.0,
                            "Bidder " + str(l) + " allocation")
            m.addConstr(num_alloc_res, gurobipy.GRB.LESS_EQUAL,
                        obj.resource.sum_resource[self.resource_name],
                        "Total number of resources")
            m.update()
            m.optimize()
            # m.write('model.lp')
            # m.write('model.sol')
            num_assigned_res = 0
            for l in obj.layers:
                obj.resource.v_r[time_step][l][self.resource_name] = 0
                for v in range(num_units):
                    if m.getVarByName('y_' + str((v + 1) * self.resource_unit) + ", " + str(l)).x == 1:
                        obj.resource.v_r[time_step][l][self.resource_name] = int((v + 1) * self.resource_unit)
                        for vv in range(v + 1):
                            if print_cmd:
                                print('Resource ' + str((num_assigned_res + 1) * self.resource_unit) + \
                                      ': Player ' + str(l) + ' wins!')
                            self.winners[time_step][(num_assigned_res + 1) * self.resource_unit] = l
                            self.win_bids[time_step][(num_assigned_res + 1) * self.resource_unit] = \
                                self.bids[time_step][l][(vv + 1) * self.resource_unit]
                            self.win_prices[time_step][(num_assigned_res + 1) * self.resource_unit] = \
                                self.bids[time_step][l][(vv + 1) * self.resource_unit]
                            num_assigned_res += 1
        self.auction_time[time_step] = time.time() - start_time_auction
        if compute_poa:
            winners_valuations = []
            for v, winner in self.winners[time_step].items():
                winners_valuations.append(self.valuations[time_step][winner][v])
            if sum(winners_valuations) != 0:
                self.poa[time_step] = self.poa[time_step] / sum(winners_valuations)
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
        time_limit = 2 * 60  # !!! Maybe adjusted
        current_total_cost = {}
        for l in obj.layers:
            if type(obj) is JcModel:
                current_total_cost[l] = obj.results_real.results_layer[l][t_step - 1]['costs']['Total']
            else:
                current_total_cost[l] = obj.results.results_layer[l][t_step - 1]['costs']['Total']
        #: Optimal Valuation, which the optimal walfare value. Used to compute POA ###
        if compute_optimal_valuation:
            if type(obj) is JcModel:
                current_optimal_tc = obj.results_real.results[t_step - 1]['costs']['Total']
            else:
                current_optimal_tc = obj.results.results[t_step - 1]['costs']['Total']
            rc = {self.resource_name: obj.resource.sum_resource[self.resource_name]}
            indp_results = indp.indp(obj.net, v_r=rc, T=1, layers=obj.layers,
                                     controlled_layers=obj.layers)
            optimal_tc = indp_results[1][0]['costs']['Total']
            self.poa[t_step] = current_optimal_tc - optimal_tc
        #: Compute valuations
        num_units = obj.resource.sum_resource[self.resource_name] // self.resource_unit
        for l in obj.layers:
            max_val_time = 0.0
            if print_cmd:
                print("Bidder-%d" % (l))
            if self.valuation_type == 'DTC':
                for v in range(num_units):
                    start_time_val = time.time()
                    neg_layer = [x for x in obj.layers if x != l]
                    functionality = obj.judgments.create_judgment_dict(obj, neg_layer)
                    rc = {self.resource_name: (v + 1) * self.resource_unit}
                    indp_results = indp.indp(obj.net, v_r=rc, T=1, layers=obj.layers,
                                             controlled_layers=[l], functionality=functionality,
                                             print_cmd=print_cmd, time_limit=time_limit)
                    new_total_cost = indp_results[1][0]['costs']['Total']
                    if indp_results[1][0]['actions'] != []:
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = current_total_cost[
                                                                                       l] - new_total_cost
                        current_total_cost[l] = new_total_cost
                    else:
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = 0.0
                    if time.time() - start_time_val > max_val_time:
                        max_val_time = time.time() - start_time_val
            elif self.valuation_type == 'DTC_uniform':
                for v in range(num_units):
                    start_time_val = time.time()
                    total_cost_bounds = []
                    for jt in ["PESSIMISTIC", "OPTIMISTIC"]:
                        neg_layer = [x for x in obj.layers if x != l]
                        functionality = obj.judgments.create_judgment_dict(obj, neg_layer,
                                                                           judge_type_forced=jt)
                        rc = {self.resource_name: (v + 1) * self.resource_unit}
                        indp_results = indp.indp(obj.net, v_r=rc, T=1, layers=obj.layers,
                                                 controlled_layers=[l], functionality=functionality,
                                                 print_cmd=print_cmd, time_limit=time_limit)
                        total_cost_bounds.append(indp_results[1][0]['costs']['Total'])
                    new_total_cost = np.random.uniform(min(total_cost_bounds),
                                                       max(total_cost_bounds), 1)[0]
                    if current_total_cost[l] - new_total_cost > 0:
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = current_total_cost[
                                                                                       l] - new_total_cost
                        current_total_cost[l] = new_total_cost
                    else:
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = 0.0
                    if time.time() - start_time_val > max_val_time:
                        max_val_time = time.time() - start_time_val
            elif self.valuation_type == 'MDDN':
                start_time_val = time.time()
                g_prime_nodes = [n[0] for n in obj.net.G.nodes(data=True) \
                                 if n[1]['data']['inf_data'].net_id == l]
                g_prime = obj.net.G.subgraph(g_prime_nodes)
                penalty_dmgd_nodes = []
                for n in g_prime.nodes(data=True):
                    if n[1]['data']['inf_data'].functionality == 0.0:
                        if n[1]['data']['inf_data'].demand > 0:
                            penalty_dmgd_nodes.append(abs(n[1]['data']['inf_data'].demand * \
                                                          n[1]['data']['inf_data'].oversupply_penalty))
                        else:
                            penalty_dmgd_nodes.append(abs(n[1]['data']['inf_data'].demand * \
                                                          n[1]['data']['inf_data'].undersupply_penalty))
                penalty_rsorted = np.sort(penalty_dmgd_nodes)[::-1]
                for v in range(num_units):
                    if v >= len(penalty_rsorted):
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = 0.0
                    else:
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = penalty_rsorted[v]
                max_val_time = time.time() - start_time_val
            elif self.valuation_type == 'STM':
                pred_dict = obj.resource.auction_model.stm_pred_dict
                max_val_time = 0.0
                for v in range(num_units):
                    pred_dict['V'] = v + 1
                    pred_results = stm.predict_resotration(obj, l, t_step, pred_dict)
                    new_total_cost = 0.0
                    equiv_run_time = 0.0
                    for pred_s, val in pred_results.items():
                        new_total_cost += val.results_layer[l][1]['costs']['Total']
                        equiv_run_time += pred_results[pred_s].results[1]['run_time']
                    new_total_cost /= pred_dict['num_pred']
                    if current_total_cost[l] - new_total_cost > 0:
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = current_total_cost[
                                                                                       l] - new_total_cost
                        current_total_cost[l] = new_total_cost
                    else:
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = 0.0

                    if equiv_run_time / pred_dict['num_pred'] > max_val_time:
                        max_val_time = equiv_run_time / pred_dict['num_pred']
            elif self.valuation_type == 'DTC-LP':
                start_time_val = time.time()
                for v in range(num_units):
                    neg_layer = [x for x in obj.layers if x != l]
                    functionality = obj.judgments.create_judgment_dict(obj, neg_layer)
                    rc = {self.resource_name: (v + 1) * self.resource_unit}
                    indp_results = indpalt.indp_relax(obj.net, v_r=rc, T=1, layers=obj.layers,
                                                      controlled_layers=[l], functionality=functionality,
                                                      print_cmd=print_cmd, time_limit=time_limit)
                    new_total_cost = indp_results[1][0]['costs']['Total']
                    if indp_results[1][0]['actions'] != []:
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = current_total_cost[
                                                                                       l] - new_total_cost
                        current_total_cost[l] = new_total_cost
                    else:
                        self.valuations[t_step][l][(v + 1) * self.resource_unit] = 0.0
                    if time.time() - start_time_val > max_val_time:
                        max_val_time = time.time() - start_time_val
            self.valuation_time[t_step][l] = max_val_time
