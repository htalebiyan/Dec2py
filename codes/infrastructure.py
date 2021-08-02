import networkx as nx
import csv
import re
import string
import numpy as np
import os
import sys
import pandas as pd
import warnings


class InfrastructureNode(object):
    """
    This class models a node in an infrastructure network

    Attributes
    ----------
    id : int
        Node id
    net_id : int
        The id of the layer of the network to which the node belong
    local_id : int
        Local node id
    failure_probability : float
        Failure probability of the node
    functionality : bool
        Functionality state of the node
    repaired : bool
        If the node is repaired or not
    reconstruction_cost : float
        Reconstruction cost of the node
    oversupply_penalty : float
        Penalty per supply unit  of the main commodity that is not used for the the node
    undersupply_penalty : float
        Penalty per demand unit  of the main commodity that is not satisfied for the the node
    demand : float
        Demand or supply value of the main commodity assigned to the node
    space : int
        The id of the geographical space where the node is
    resource_usage : dict
        The dictionary that shows how many resource (of each resource type) is employed to repair the node
    extra_com : dict
        The dictionary that shows demand, oversupply_penalty, and undersupply_penalty corresponding to commodities
        other than the main commodity
    """
    def __init__(self, id, net_id, local_id=""):
        self.id = id
        self.net_id = net_id
        if local_id == "":
            self.local_id = id
        self.local_id = local_id
        self.failure_probability = 0.0
        self.functionality = 1.0
        self.repaired = 1.0
        self.reconstruction_cost = 0.0
        self.oversupply_penalty = 0.0
        self.undersupply_penalty = 0.0
        self.demand = 0.0
        self.space = 0
        self.resource_usage = {}
        self.extra_com = {}

    def set_failure_probability(self, failure_probability):
        """
        This function sets the failure probability of the node

        Parameters
        ----------
        failure_probability : float
            Assigned failure probability of node

        Returns
        -------
        None.

        """
        self.failure_probability = failure_probability

    def set_extra_commodity(self, extra_commodity):
        """
        This function initialize the dictionary for the extra commodities

        Parameters
        ----------
        extra_commodity : list
            List of extra commodities

        Returns
        -------
        None.

        """
        for l in extra_commodity:
            self.extra_com[l] = {'demand': 0, 'oversupply_penalty': 0, 'undersupply_penalty': 0}

    def set_resource_usage(self, resource_names):
        """
        This function initialize the dictionary for resource usage per all resource types in the analysis

        Parameters
        ----------
        resource_names : list
            List of resource types

        Returns
        -------
        None.

        """
        for rc in resource_names:
            self.resource_usage[rc] = 0

    def in_space(self, space_id):
        """
        This function checks if the node is in a given space or not

        Parameters
        ----------
        space_id :
            The id of the space that is checked

        Returns
        -------
            : bool
            Returns 1 if the node is in the space, and 0 otherwise.
        """
        if self.space == space_id:
            return 1
        else:
            return 0


class InfrastructureArc(object):
    """
    This class models an arc in an infrastructure network

    Attributes
    ----------
    source : int
        Start (or head) node id
    dest : int
        End (or tail) node id
    layer : int
        The id of the layer of the network to which the arc belong
    failure_probability : float
        Failure probability of the arc
    functionality : bool
        Functionality state of the node
    repaired : bool
        If the arc is repaired or not
    flow_cost : float
        Unit cost of sending the main commodity through the arc
    reconstruction_cost : float
        Reconstruction cost of the arc
    capacity : float
        Maximum volume of the commodities that the arc can carry
    space : int
        The id of the geographical space where the arc is
    resource_usage : dict
        The dictionary that shows how many resource (of each resource type) is employed to repair the arc
    extra_com : dict
        The dictionary that shows flow_cost corresponding to commodities other than the main commodity
    is_interdep : bool
        If arc represent a normal arc (that carry commodity within a single layer) or physical interdependency between
        nodes from different layers
    """
    def __init__(self, source, dest, layer, is_interdep=False):
        self.source = source
        self.dest = dest
        self.layer = layer
        self.failure_probability = 0.0
        self.functionality = 1.0
        self.repaired = 1.0
        self.flow_cost = 0.0
        self.reconstruction_cost = 0.0
        self.resource_usage = {}
        self.capacity = 0.0
        self.space = 0
        self.extra_com = {}
        self.is_interdep = is_interdep

    def set_extra_commodity(self, extra_commodity):
        """
        This function initialize the dictionary for the extra commodities

        Parameters
        ----------
        extra_commodity : list
            List of extra commodities

        Returns
        -------
        None.

        """
        for l in extra_commodity:
            self.extra_com[l] = {'flow_cost': 0}

    def set_resource_usage(self, resource_names):
        """
        This function initialize the dictionary for resource usage per all resource types in the analysis

        Parameters
        ----------
        resource_names : list
            List of resource types

        Returns
        -------
        None.

        """
        for rc in resource_names:
            self.resource_usage[rc] = 0

    def in_space(self, space_id):
        """
        This function checks if the arc is in a given space or not

        Parameters
        ----------
        space_id :
            The id of the space that is checked

        Returns
        -------
            : bool
            Returns 1 if the arc is in the space, and 0 otherwise.
        """
        if self.space == space_id:
            return 1
        else:
            return 0


class InfrastructureInterdepArc(InfrastructureArc):
    """
    This class models a physical interdependency between nodes from two different layers. This class inherits from
    :class:`InfrastructureArc`, where `source` attributes corresponds to the dependee node, and `dest` corresponds to
    the depender node. The depender node is non-functional if the corresponding dependee node is non-functional.

    Attributes
    ----------
    source_layer : int
        The id of the layer where the dependee node is
    dest_layer : int
        The id of the layer where the depender node is
    gamma : float
        The strength of the dependency, which is a number between 0 and 1.
    """
    def __init__(self, source, dest, source_layer, dest_layer, gamma):
        super(InfrastructureInterdepArc, self).__init__(source, dest, source_layer, True)
        self.source_layer = source_layer
        self.dest_layer = dest_layer
        self.gamma = gamma


class InfrastructureSpace(object):
    """
    This class models a geographical space.

    Attributes
    ----------
    id : int
        The id of the space
    cost : float
        The cost of preparing the space for a repair action
    """
    def __init__(self, id, cost):
        self.id = id
        self.cost = cost


class InfrastructureNetwork(object):
    """
    Stores information of the infrastructure network

    Attributes
    ----------
    G : networkx.DiGraph
        The networkx graph object that stores node, arc, and interdependency information
    S : list
        List of geographical spaces on which the network lays
    id : int
        Id of the network
    """
    def __init__(self, id):
        self.G = nx.DiGraph()
        self.S = []
        self.id = id

    def copy(self):
        """
        This function copies the current :class:`InfrastructureNetwork` object

        Returns
        -------
        new_net: :class:`InfrastructureNetwork`
            Copy of the current infrastructure network object
        """
        new_net = InfrastructureNetwork(self.id)
        new_net.G = self.G.copy()
        new_net.S = [s for s in self.S]
        return new_net

    def update_with_strategy(self, player_strategy):
        """
        This function modify the functionality of node and arc per a given strategy

        Parameters
        ----------
        player_strategy : list
            Given strategy, where the first list item shows the functionality of nodes, and the second one is for arcs

        Returns
        -------
        None.

        """
        for q in player_strategy[0]:
            node = q
            strat = player_strategy[0][q]
            self.G.node[q]['data']['inf_data'].repaired = round(strat['repair'])
            self.G.node[q]['data']['inf_data'].functionality = round(strat['w'])
        for q in player_strategy[1]:
            src = q[0]
            dst = q[1]
            strat = player_strategy[1][q]
            self.G[src][dst]['data']['inf_data'].repaired = round(strat['repair'])
            self.G[src][dst]['data']['inf_data'].functionality = round(strat['y'])

    def get_clusters(self, layer):
        """
        This function find the clusters in a layer of the network

        Parameters
        ----------
        layer : int
            The id of the desired layer

        Returns
        -------
            : list
            List of layer components

        """
        g_prime_nodes = [n[0] for n in self.G.nodes(data=True) if
                         n[1]['data']['inf_data'].net_id == layer and n[1]['data']['inf_data'].functionality == 1.0]
        g_prime = nx.DiGraph(self.G.subgraph(g_prime_nodes).copy())
        g_prime.remove_edges_from(
            [(u, v) for u, v, a in g_prime.edges(data=True) if a['data']['inf_data'].functionality == 0.0])
        # print nx.connected_components(g_prime.to_undirected())
        return list(nx.connected_components(g_prime.to_undirected()))

    def gc_size(self, layer):
        """
        This function finds the size of the largest component in a layer of the network

        Parameters
        ----------
        layer : int
            The id of the desired layer

        Returns
        -------
            : int
            Size of the largest component in the layer
        """
        g_prime_nodes = [n[0] for n in self.G.nodes(data=True) if
                         n[1]['data']['inf_data'].net_id == layer and n[1]['data']['inf_data'].functionality == 1.0]
        g_prime = nx.Graph(self.G.subgraph(g_prime_nodes))
        g_prime.remove_edges_from(
            [(u, v) for u, v, a in g_prime.edges(data=True) if a['data']['inf_data'].functionality == 0.0])
        cc = nx.connected_components(g_prime.to_undirected())
        if cc:
            # if len(list(cc)) == 1:
            #    print "I'm here"
            #    return len(list(cc)[0])
            # cc_list=list(cc)
            # print "Length",len(cc_list)
            # if len(cc_list) == 1:
            #    return len(cc_list[0])
            return len(max(cc, key=len))
        else:
            return 0

    def to_game_file(self, layers=None):
        """
        This function writes the multi-defender security games.

        Parameters
        ----------
        layers : list
            List of layers in the game.

        Returns
        -------
        None.

        """
        if layers is None:
            layers = [1, 3]
        with open("../results/indp_gamefile.game") as f:
            num_players = len(layers)
            num_targets = len([n for n in self.G.nodes(data=True) if n[1]['data']['inf_data'].net_id in layers])
            f.write(str(num_players) + "," + str(num_targets) + ",2\n")
            for layer in range(len(layers)):
                layer_nodes = [n for n in self.G.nodes(data=True) if n[1]['data']['inf_data'].net_id == layers[layer]]
                for node in layer_nodes:
                    def_values = [0.0] * len(layers)
                    def_values[layer] = abs(node[1]['data']['inf_data'].demand)
                    atk_values = sum(def_values)
                    f.write(str(layer) + "\n")
                    # f.write("0.0,1.0")
                    for v in def_values:
                        f.write(str(v) + "\n")
                    f.write(str(atk_values) + "\n")

    def to_csv(self, filename="infrastructure_adj.csv"):
        """
        This function writes the object to a csv file

        Parameters
        ----------
        filename : str
            Name of the file to which the network should be written

        Returns
        -------
        None.

        """
        with open(filename, 'w') as f:
            for u, v, a in self.G.edges(data=True):
                f.write(str(u[0]) + "." + str(u[1]) + "," + str(v[0]) + "." + str(v[1]) + "\n")


def load_infrastructure_data(base_dir, ext_interdependency_dir=None,  magnitude=6, sim_number=1, cost_scale=1.0,
                             data_format='shelby_extended', extra_commodity=None):
    """
    This function reads the infrastructure network from file

    Parameters
    ----------
    base_dir : str
        The address of the folder where the basic network information (topology, parameters, etc.) are stored
    ext_interdependency_dir : str
        (only for old format of input data) The address of the file where the interdependency data are stored. The
        default in None.
    magnitude : int
        (only for old format of input data) Magnitude of the initial damage scenario. The default is 6.
    sim_number : int
        (only for old format of input data) The simulation number. The default is 1.
    cost_scale : float
        The factor by which all cost values has to multiplied. The default is 1.0.
    data_format : str
        The format of input data, which is either "shelby_old" (consistent with the output format of Xpress optimizing
        software employed by Andres Gonzalez) or "shelby_extended" (which is devised by Hesam Talebiyan). The default
        is "shelby_extended".
    extra_commodity :
        (only for extended format of input data) List of extra-commodities in the analysis. The default is None, which
        only considers a main commodity.

    Returns
    -------
    G : networkx.DiGraph
        The networkx graph object that stores node, arc, and interdependency information

    """
    if data_format == 'shelby_old':
        G = load_infrastructure_array_format(base_dir=base_dir, ext_interdependency_dir=ext_interdependency_dir,
                                             magnitude=magnitude, sim_number=sim_number, cost_scale=cost_scale)
        return G
    elif data_format == 'shelby_extended':
        G = load_infrastructure_array_format_extended(base_dir=base_dir, cost_scale=cost_scale,
                                                      extra_commodity=extra_commodity)
        return G
    else:
        sys.exit('Error: no interdependent network is found')


def load_interdependencies(net_number, src_net_number, base_dir="../data/INDP_4-12-2016"):
    """
    (only for old format of input data) This function reads interdependency data from file.

    Parameters
    ----------
    net_number : int
        ID of the depender layer.
    src_net_number : int
        ID of the dependee layer.
    base_dir : src
        The address of the folder where the basic network information (topology, parameters, etc.) are stored

    Returns
    -------
    arcs: list
        List of interdependencies as in arc format
    """
    # Variables in CSV format is broken into files:
    #     S<net_number>_adj : Adjacency matrix for <net_number>
    #     S<net_number>_bal : demand/supply of node.
    #     S<net_number>_cst : flow cost ?
    #     S<net_number>_dset : Set of distributors (but this can be calculated with *_bal)
    #     S<net_number>_gset : Set of generators (but this can be calculated with *_bal)
    #     S<net_number>_ub  : Upper bound on arc capacity.
    #     S<net_number>_lb  : Lower bound on arc capacity (typically 0).
    #     S<net_number>_imt : Interdependency matrix for network <net_number>. <=== This method will only load
    #     interdependencies.
    arcs = []
    with open(base_dir + "/csv/S" + str(net_number) + "_imt.csv") as f:
        rows = [[int(y) for y in string.split(x, ",")] for x in f.readlines()]
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                if rows[i][j] > 0:
                    arc = InfrastructureInterdepArc(i + 1, j + 1, src_net_number, net_number, rows[i][j])
                    arcs.append(arc)
    return arcs


def load_infrastructure_array_format(base_dir, ext_interdependency_dir=None, magnitude=6, sim_number=1, cost_scale=1.0):
    """
    This function reads the infrastructure network from file in the old format

    Parameters
    ----------
    base_dir : str
        The address of the folder where the basic network information (topology, parameters, etc.) are stored
    ext_interdependency_dir : str
        The address of the file where the interdependency data are stored. The default in None.
    magnitude : int
        Magnitude of the initial damage scenario. The default is 6.
    sim_number : int
        The simulation number. The default is 1.
    cost_scale : float
        The factor by which all cost values has to multiplied. The default is 1.0.

    Returns
    -------
    G : networkx.DiGraph
        The networkx graph object that stores node, arc, and interdependency information

    """
    variables = ['v', 'c', 'f', 'q', 'Mp', 'Mm', 'g', 'b', 'u', 'h', 'p', 'gamma', 'alpha', 'beta', 'a', 'n']
    entry_regex = r"\(\s*(?P<tuple>(\d+\s*,\s*)*\d+)\s*\)\s*(?P<val>-?\d+)"
    recovery_entry_regex = r"\(\s*(?P<tuple>(\d+\s+)*\d+)\s*\)\s*(?P<val>\d+)"
    var_regex = r"(?P<varname>\D+):\s*\[(?P<entries>[^\]]*)\]"
    pattern = re.compile(var_regex, flags=re.DOTALL)
    files = [base_dir + "/INDP_InputData/MURI_INDP_data.txt"]
    if sim_number > 0:
        files.append(base_dir + "/Failure and recovery scenarios/recoveryM" + str(magnitude) + "v3.txt")
    vars = {}
    for file in files:
        with open(file) as f:
            # print "Opened",file,"."
            lines = f.read()
            vars_strings = {}
            for match in pattern.finditer(lines):
                vars_strings[string.strip(match.group('varname'))] = string.strip(match.group('entries'))
                # print "Varstrings=",vars_strings.keys()
            for v in vars_strings:
                entry_string = vars_strings[v]
                entry_pattern = re.compile(entry_regex, flags=re.DOTALL)
                is_match = False
                for match in entry_pattern.finditer(entry_string):
                    is_match = True
                    tuple_string = match.group('tuple')
                    val_string = match.group('val')
                    tuple_string = string.split(tuple_string, ",")
                    tuple_string = [int(string.strip(x)) for x in tuple_string]
                    val_string = float(string.strip(val_string))
                    # print v,":",tuple_string,"=>",val_string
                    if v not in vars:
                        vars[v] = []
                    vars[v].append((tuple_string, val_string))
                if not is_match:
                    recovery_pattern = re.compile(recovery_entry_regex, flags=re.DOTALL)
                    for match in recovery_pattern.finditer(entry_string):
                        is_match = True
                        tuple_string = match.group('tuple')
                        val_string = match.group('val')
                        tuple_string = string.split(tuple_string, " ")
                        tuple_string = [int(string.strip(x)) for x in tuple_string]
                        val_string = float(string.strip(val_string))
                        # print v,":",tuple_string,"=>",val_string
                        if v not in vars:
                            vars[v] = []
                        vars[v].append((tuple_string, val_string))
    G = InfrastructureNetwork("Test")
    global_index = 1
    for node in vars['n']:
        n = InfrastructureNode(global_index, node[0][1], node[0][0])
        # print "Adding node",node[0][0],"in layer",node[0][1],"."
        G.G.add_node((n.local_id, n.net_id), data={'inf_data': n})
        global_index += 1
    for arc in vars['a']:
        a = InfrastructureArc(arc[0][0], arc[0][1], arc[0][2])
        G.G.add_edge((a.source, a.layer), (a.dest, a.layer), data={'inf_data': a})
    if ext_interdependency_dir:
        print("Loading external interdependencies...")
        interdep_arcs = load_interdependencies(1, 3, base_dir=ext_interdependency_dir)
        for a in interdep_arcs:
            G.G.add_edge((a.source, a.source_layer), (a.dest, a.dest_layer), data={'inf_data': a})
        interdep_arcs = load_interdependencies(3, 1, base_dir=ext_interdependency_dir)
        for a in interdep_arcs:
            G.G.add_edge((a.source, a.source_layer), (a.dest, a.dest_layer), data={'inf_data': a})
    else:
        for arc in vars['gamma']:
            gamma = float(arc[1])
            if gamma > 0.0:
                a = InfrastructureInterdepArc(arc[0][0], arc[0][1], arc[0][2], arc[0][3], gamma)
                G.G.add_edge((a.source, a.source_layer), (a.dest, a.dest_layer), data={'inf_data': a})
    # Arc Attributes.
    for flow_cost in vars['c']:
        G.G[(flow_cost[0][0], flow_cost[0][2])][(flow_cost[0][1], flow_cost[0][2])]['data'][
            'inf_data'].flow_cost = float(flow_cost[1]) * cost_scale
    for rec_cost in vars['f']:
        G.G[(rec_cost[0][0], rec_cost[0][2])][(rec_cost[0][1], rec_cost[0][2])]['data'][
            'inf_data'].reconstruction_cost = float(rec_cost[1]) * cost_scale
    for cap in vars['u']:
        G.G[(cap[0][0], cap[0][2])][(cap[0][1], cap[0][2])]['data']['inf_data'].capacity = cap[1]
    for res in vars['h']:
        # Assume only one kind of resource for now.
        G.G[(res[0][0], res[0][2])][(res[0][1], res[0][2])]['data']['inf_data'].resource_usage = res[1]
    # Space attributes.
    for space in vars['g']:
        G.S.append(InfrastructureSpace(int(space[0][0]), float(space[1])))
    for space in vars['beta']:
        G.G[(space[0][0], space[0][2])][(space[0][1], space[0][2])]['data']['inf_data'].space = int(space[0][3])
    for space in vars['alpha']:
        G.G.node[(space[0][0], space[0][1])]['data']['inf_data'].space = int(space[0][2])
    # Node Attributes.
    for rec_cost in vars['q']:
        G.G.node[(rec_cost[0][0], rec_cost[0][1])]['data']['inf_data'].reconstruction_cost = float(
            rec_cost[1]) * cost_scale
    for sup in vars['Mp']:
        G.G.node[(sup[0][0], sup[0][1])]['data']['inf_data'].oversupply_penalty = float(sup[1]) * cost_scale
    for sup in vars['Mm']:
        G.G.node[(sup[0][0], sup[0][1])]['data']['inf_data'].undersupply_penalty = float(sup[1]) * cost_scale
    for res in vars['p']:
        # Assume only one kind of resource for now.
        G.G.node[(res[0][0], res[0][1])]['data']['inf_data'].resource_usage = res[1]
    for dem in vars['b']:
        G.G.node[(dem[0][0], dem[0][1])]['data']['inf_data'].demand = dem[1]

    # Load failure scenarios.
    if sim_number > 0:
        for func in vars['nresults']:
            if func[0][0] == sim_number:
                G.G.node[(func[0][1], func[0][2])]['data']['inf_data'].functionality = float(func[1])
                G.G.node[(func[0][1], func[0][2])]['data']['inf_data'].repaired = float(func[1])
                # if float(func[1]) == 0.0:
                #    print "Node (",`func[0][1]`+","+`func[0][2]`+") broken."
        for func in vars['aresults']:
            if func[0][0] == sim_number:
                G.G[(func[0][1], func[0][3])][(func[0][2], func[0][3])]['data']['inf_data'].functionality = float(
                    func[1])
                G.G[(func[0][1], func[0][3])][(func[0][2], func[0][3])]['data']['inf_data'].repaired = float(func[1])
                # if float(func[1]) == 0.0:
                #    print "Arc ((",`func[0][1]`+","+`func[0][3]`+"),("+`func[0][2]`+","+`func[0][3]`+")) broken."
    return G


def load_infrastructure_array_format_extended(base_dir, cost_scale=1.0, extra_commodity=None):
    """
    This function reads the infrastructure network from file in the extended format

    Parameters
    ----------
    base_dir : str
        The address of the folder where the basic network information (topology, parameters, etc.) are stored
    cost_scale : float
        The factor by which all cost values has to multiplied. The default is 1.0.
    extra_commodity :
        (only for extended format of input data) List of extra-commodities in the analysis. The default is None, which
        only considers a main commodity.

    Returns
    -------
    G : networkx.DiGraph
        The networkx graph object that stores node, arc, and interdependency information

    """
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    net_names = {'Water': 1, 'Gas': 2, 'Power': 3, 'Telecommunication': 4}  # !!!
    G = InfrastructureNetwork("Test")
    global_index = 0
    for file in files:
        fname = file[0:-4]
        if fname[-5:] == 'Nodes':
            with open(base_dir + file) as f:
                data = pd.read_csv(f, delimiter=',')
                net = net_names[fname[:-5]]
                for v in data.iterrows():
                    try:
                        node_id = v[1]['ID']
                    except KeyError:
                        node_id = v[1]['nodenwid']
                    n = InfrastructureNode(global_index, net, int(node_id))
                    G.G.add_node((n.local_id, n.net_id), data={'inf_data': n})
                    global_index += 1
                    node_main_data = G.G.nodes[(n.local_id, n.net_id)]['data']['inf_data']
                    node_main_data.reconstruction_cost = float(v[1]['q (complete DS)']) * cost_scale
                    node_main_data.oversupply_penalty = float(v[1]['Mp']) * cost_scale
                    node_main_data.undersupply_penalty = float(v[1]['Mm']) * cost_scale
                    node_main_data.demand = float(v[1]['Demand'])
                    if 'guid' in v[1].index.values:
                        node_main_data.guid = v[1]['guid']
                    resource_names = [x for x in list(v[1].index.values) if x[:2] == 'p_']
                    if len(resource_names) > 0:
                        n.set_resource_usage(resource_names)
                        for rc in resource_names:
                            n.resource_usage[rc] = v[1][rc]
                    else:
                        n.resource_usage['p_'] = 1
                    if extra_commodity:
                        n.set_extra_commodity(extra_commodity[net])
                        for l in extra_commodity[net]:
                            ext_com_data = G.G.nodes[(n.local_id, n.net_id)]['data']['inf_data'].extra_com[l]
                            ext_com_data['oversupply_penalty'] = float(v[1]['Mp_' + l]) * cost_scale
                            ext_com_data['undersupply_penalty'] = float(v[1]['Mm_' + l]) * cost_scale
                            ext_com_data['demand'] = float(v[1]['Demand_' + l])
    for file in files:
        fname = file[0:-4]
        if fname[-4:] == 'Arcs':
            with open(base_dir + file) as f:
                data = pd.read_csv(f, delimiter=',')
                net = net_names[fname[:-4]]
                for v in data.iterrows():
                    try:
                        start_id = v[1]['Start Node']
                        end_id = v[1]['End Node']
                    except KeyError:
                        start_id = v[1]['fromnode']
                        end_id = v[1]['tonode']
                    for duplicate in range(2):
                        if duplicate == 0:
                            a = InfrastructureArc(int(start_id), int(end_id), net)
                        elif duplicate == 1:
                            a = InfrastructureArc(int(end_id), int(start_id), net)
                        G.G.add_edge((a.source, a.layer), (a.dest, a.layer), data={'inf_data': a})
                        arc_main_data = G.G[(a.source, a.layer)][(a.dest, a.layer)]['data']['inf_data']
                        arc_main_data.flow_cost = float(v[1]['c']) * cost_scale
                        arc_main_data.reconstruction_cost = float(v[1]['f']) * cost_scale
                        arc_main_data.capacity = float(v[1]['u'])
                        if 'guid' in v[1].index.values:
                            arc_main_data.guid = v[1]['guid']
                        resource_names = [x for x in list(v[1].index.values) if x[:2] == 'h_']
                        if len(resource_names) > 0:
                            a.set_resource_usage(resource_names)
                            for rc in resource_names:
                                a.resource_usage[rc] = v[1][rc]
                        else:
                            a.resource_usage['h_'] = 1
                        if extra_commodity:
                            a.set_extra_commodity(extra_commodity[net])
                            for l in extra_commodity[net]:
                                ext_com_data = \
                                    G.G[(a.source, a.layer)][(a.dest, a.layer)]['data']['inf_data'].extra_com[l]
                                ext_com_data['flow_cost'] = float(v[1]['c_' + l]) * cost_scale
    for file in files:
        fname = file[0:-4]
        if fname in ['beta', 'alpha', 'g', 'Interdep']:
            with open(base_dir + file) as f:
                data = pd.read_csv(f, delimiter=',')
                for v in data.iterrows():
                    if fname == 'beta':
                        net = net_names[v[1]['Network']]
                        G.G[(int(v[1]['Start Node']), net)][(int(v[1]['End Node']), net)]['data'][
                            'inf_data'].space = int(int(v[1]['Subspace']))
                        G.G[(int(v[1]['End Node']), net)][(int(v[1]['Start Node']), net)]['data'][
                            'inf_data'].space = int(int(v[1]['Subspace']))
                    if fname == 'alpha':
                        net = net_names[v[1]['Network']]
                        G.G.node[(int(v[1]['ID']), net)]['data']['inf_data'].space = int(int(v[1]['Subspace']))
                    if fname == 'g':
                        G.S.append(InfrastructureSpace(int(v[1]['Subspace_ID']), float(v[1]['g'])))
                    if fname == 'Interdep' and v[1]['Type'] == 'Physical':
                        i = int(v[1]['Dependee Node'])
                        net_i = net_names[v[1]['Dependee Network']]
                        j = int(v[1]['Depender Node'])
                        net_j = net_names[v[1]['Depender Network']]
                        a = InfrastructureInterdepArc(i, j, net_i, net_j, gamma=1.0)
                        G.G.add_edge((a.source, a.source_layer), (a.dest, a.dest_layer), data={'inf_data': a})
                        if extra_commodity:
                            a.set_extra_commodity(extra_commodity[net_i])
                            a.set_extra_commodity(extra_commodity[net_j])
    return G


def add_failure_scenario(G, dam_dir, magnitude=6, v=3, sim_number=1):
    """
    This function reads initial damage data from file in the ANDRES format, and apply it to the infrastructure network

    Parameters
    ----------
    G : networkx.DiGraph
        The networkx graph object that stores node, arc, and interdependency information
    dam_dir : str
        The address of the folder where the initial damage data are stored
    magnitude : int
        Magnitude of the initial damage scenario. The default is 6.
    v : int
        Number of the resources in the analysis. The default is 3.
    sim_number : int
        The simulation number. The default is 1.

    Returns
    -------
    None.

    """
    print("Initialize Damage...")
    if sim_number == "INF":
        # Destroy all nodes!!!!
        for n, d in G.G.nodes(data=True):
            G.G.node[n]['data']['inf_data'].functionality = 0.0
            G.G.node[n]['data']['inf_data'].repaired = 0.0
        for u, v, a in G.G.edges(data=True):
            if not a['data']['inf_data'].is_interdep:
                G.G[u][v]['data']['inf_data'].functionality = 0.0
                G.G[u][v]['data']['inf_data'].repaired = 0.0
    elif sim_number > 0:
        variables = ['v', 'c', 'f', 'q', 'Mp', 'Mm', 'g', 'b', 'u', 'h', 'p', 'gamma', 'alpha', 'beta', 'a', 'n']
        entry_regex = r"\(\s*(?P<tuple>(\d+\s*,\s*)*\d)\s*\)\s*(?P<val>-?\d+)"
        recovery_entry_regex = r"\(\s*(?P<tuple>(\d+\s+)*\d)\s*\)\s*(?P<val>\d+)"
        var_regex = r"(?P<varname>\D+):\s*\[(?P<entries>[^\]]*)\]"
        pattern = re.compile(var_regex, flags=re.DOTALL)
        file = dam_dir + "/Failure and recovery scenarios/recoveryM" + str(magnitude) + "v3.txt"
        vars = {}
        with open(file) as f:
            # print "Opened",file,"."
            lines = f.read()
            vars_strings = {}
            for match in pattern.finditer(lines):
                vars_strings[string.strip(match.group('varname'))] = string.strip(match.group('entries'))
            # print "Varstrings=",vars_strings.keys()
            for v in vars_strings:
                entry_string = vars_strings[v]
                entry_pattern = re.compile(entry_regex, flags=re.DOTALL)
                is_match = False
                for match in entry_pattern.finditer(entry_string):
                    is_match = True
                    tuple_string = match.group('tuple')
                    val_string = match.group('val')
                    tuple_string = string.split(tuple_string, ",")
                    tuple_string = [int(string.strip(x)) for x in tuple_string]
                    val_string = float(string.strip(val_string))
                    # print v,":",tuple_string,"=>",val_string
                    if v not in vars:
                        vars[v] = []
                    vars[v].append((tuple_string, val_string))
                if not is_match:
                    recovery_pattern = re.compile(recovery_entry_regex, flags=re.DOTALL)
                    for match in recovery_pattern.finditer(entry_string):
                        is_match = True
                        tuple_string = match.group('tuple')
                        val_string = match.group('val')
                        tuple_string = string.split(tuple_string, " ")
                        tuple_string = [int(string.strip(x)) for x in tuple_string]
                        val_string = float(string.strip(val_string))
                        # print v,":",tuple_string,"=>",val_string
                        if v not in vars:
                            vars[v] = []
                        vars[v].append((tuple_string, val_string))
        # Load failure scenarios.
        for func in vars['nresults']:
            if func[0][0] == sim_number:
                G.G.node[(func[0][1], func[0][2])]['data']['inf_data'].functionality = float(func[1])
                G.G.node[(func[0][1], func[0][2])]['data']['inf_data'].repaired = float(func[1])
                # if float(func[1]) == 0.0:
                #    print "Node (",`func[0][1]`+","+`func[0][2]`+") broken."
        for func in vars['aresults']:
            if func[0][0] == sim_number:
                G.G[(func[0][1], func[0][3])][(func[0][2], func[0][3])]['data']['inf_data'].functionality = float(
                    func[1])
                G.G[(func[0][1], func[0][3])][(func[0][2], func[0][3])]['data']['inf_data'].repaired = float(func[1])
                # if float(func[1]) == 0.0:
                #    print "Arc ((",`func[0][1]`+","+`func[0][3]`+"),("+`func[0][2]`+","+`func[0][3]`+")) broken."


def add_from_csv_failure_scenario(G, sample, dam_dir=""):
    """
    This function reads initial damage data from file in the from_csv format, and apply it to the infrastructure
    network. This format only consider one magnitude value (0), and there can be as many as samples form that magnitude.

    Parameters
    ----------
    G : networkx.DiGraph
        The networkx graph object that stores node, arc, and interdependency information
    sample : int
        Sample number of the initial damage scenario,
    dam_dir : str
        The address of the folder where the initial damage data are stored

    Returns
    -------
    None.

    """
    # print("Initialize Random Damage...")
    with open(dam_dir + 'Initial_node.csv') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        for row in data:
            raw_n = row[0]
            raw_n = raw_n.split(',')
            n = (int(raw_n[0].strip(' )(')), int(raw_n[1].strip(' )(')))
            state = float(row[sample + 1])
            G.G.nodes[n]['data']['inf_data'].functionality = state
            G.G.nodes[n]['data']['inf_data'].repaired = state

    with open(dam_dir + 'Initial_link.csv') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        for row in data:
            raw_uv = row[0]
            raw_uv = raw_uv.split(',')
            u = (int(raw_uv[0].strip(' )(')), int(raw_uv[1].strip(' )(')))
            v = (int(raw_uv[2].strip(' )(')), int(raw_uv[3].strip(' )(')))
            state = float(row[sample + 1])
            if state == 0.0:
                G.G[u][v]['data']['inf_data'].functionality = state
                G.G[u][v]['data']['inf_data'].repaired = state

                G.G[v][u]['data']['inf_data'].functionality = state
                G.G[v][u]['data']['inf_data'].repaired = state


def add_wu_failure_scenario(G, dam_dir, no_set=0, no_sce=0):
    """
    This function reads initial damage data from file in the WU format, and apply it to the infrastructure network. The
    damage data for this dataset comes in a format similar to the hazard maps from Jason Wu :cite:`Wu2017`, which
    consist of N sets (= samples) of M damage scenarios (= magnitudes). For shelby county, for example, N=50 and M=96.

    Parameters
    ----------
    G : networkx.DiGraph
        The networkx graph object that stores node, arc, and interdependency information
    dam_dir : str
        The address of the folder where the initial damage data are stored
    no_set : int
        The number of the set of the initial damage scenario. The default is 0.
    no_sce : int
        The number of the scenario of the initial damage scenario The default is 0.

    Returns
    -------
    None.

    """
    print("Initialize Wu failure scenarios...")
    dam_nodes = {}
    dam_arcs = {}
    folder_dir = dam_dir + 'Set%d/Sce%d/' % (no_set, no_sce)
    net_names = {1: 'Water', 2: 'Gas', 3: 'Power', 4: 'Telecommunication'}
    offset = 0  # set to 1 if node IDs start from 1
    # Load failure scenarios.
    if os.path.exists(folder_dir):
        for k in range(1, len(net_names.keys()) + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dam_nodes[k] = np.loadtxt(folder_dir + 'Net_%s_Damaged_Nodes.txt' \
                                          % net_names[k]).astype('int')
                dam_arcs[k] = np.loadtxt(folder_dir + 'Net_%s_Damaged_Arcs.txt' \
                                         % net_names[k]).astype('int')
        for k in range(1, len(net_names.keys()) + 1):
            if dam_nodes[k].size != 0:
                if dam_nodes[k].size == 1:
                    dam_nodes[k] = [dam_nodes[k]]
                for v in dam_nodes[k]:
                    G.G.nodes[(v + offset, k)]['data']['inf_data'].functionality = 0.0
                    G.G.nodes[(v + offset, k)]['data']['inf_data'].repaired = 0.0
                    # print("Node (",`v+offset`+","+`k`+") broken.")
            if dam_arcs[k].size != 0:
                if dam_arcs[k].size == 2:
                    dam_arcs[k] = [dam_arcs[k]]
                for a in dam_arcs[k]:
                    G.G[(a[0] + offset, k)][(a[1] + offset, k)]['data']['inf_data'].functionality = 0.0
                    G.G[(a[0] + offset, k)][(a[1] + offset, k)]['data']['inf_data'].repaired = 0.0
                    # print("Arc ((",`a[0]+offset`+","+`k`+"),("+`a[1]+offset`+","+`k`+")) broken.")

                    G.G[(a[1] + offset, k)][(a[0] + offset, k)]['data']['inf_data'].functionality = 0.0
                    G.G[(a[1] + offset, k)][(a[0] + offset, k)]['data']['inf_data'].repaired = 0.0
                    # print("Arc ((",`a[1]+offset`+","+`k`+"),("+`a[0]+offset`+","+`k`+")) broken.")
    else:
        pass  # Un-damaging scenarios are not presented with any file or folder in the datasets


def load_synthetic_network(base_dir, topology='Random', config=6, sample=0, cost_scale=1.0):
    """
    This function reads a synthetic network from file

    Parameters
    ----------
    base_dir : str
        The address of the folder where the basic network information (topology, parameters, etc.) are stored
    topology : str
        Topology of the layers of the synthetic network. The default in 'Random'.
    config : int
        Configuration number of the initial damage scenario. The default is 6.
    sample : int
        Sample number of the initial damage scenario. The default is 0.
    cost_scale : float
        The factor by which all cost values has to multiplied. The default is 1.0.

    Returns
    -------
    G : networkx.DiGraph
        The networkx graph object that stores node, arc, and interdependency information
    no_resource : int
        Number of resources available for restoration
    : list
        List of layers of the synthetic network
    """
    print("Initialize Damage...")
    net_dir = base_dir + '/' + topology + 'Networks/'
    topo_initial = {'Random': 'RN', 'ScaleFree': 'SFN', 'Grid': 'GN', 'General': 'GEN'}
    with open(net_dir + 'List_of_Configurations.txt') as f:
        config_data = pd.read_csv(f, delimiter='\t')
        config_data = config_data.rename(columns=lambda x: x.strip())
    config_param = config_data.iloc[config]
    no_layers = int(config_param.loc['No. Layers'])
    no_resource = int(config_param.loc['Resource Cap'])

    file_dir = net_dir + topo_initial[topology] + 'Config_' + str(config) + '/Sample_' + str(sample) + '/'
    G = InfrastructureNetwork("Test")
    global_index = 0
    files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
    for k in range(1, no_layers + 1):
        for file in files:
            if file == 'N' + str(k) + '_Nodes.txt':
                with open(file_dir + file) as f:
                    # print "Opened",file,"."
                    data = pd.read_csv(f, delimiter='\t', header=None)
                    for v in data.iterrows():
                        n = InfrastructureNode(global_index, k, int(v[1][0]))
                        # print "Adding node",node[0][0],"in layer",node[0][1],"."
                        G.G.add_node((n.local_id, n.net_id), data={'inf_data': n})
                        global_index += 1
                        G.G.nodes[(n.local_id, n.net_id)]['data']['inf_data'].reconstruction_cost = float(
                            v[1][7]) * cost_scale
                        G.G.nodes[(n.local_id, n.net_id)]['data']['inf_data'].oversupply_penalty = float(
                            v[1][5]) * cost_scale
                        G.G.nodes[(n.local_id, n.net_id)]['data']['inf_data'].undersupply_penalty = float(
                            v[1][6]) * cost_scale
                        # !!! Assume only one kind of resource for now and one resource for each repaired element.
                        G.G.nodes[(n.local_id, n.net_id)]['data']['inf_data'].resource_usage['p_'] = 1
                        G.G.nodes[(n.local_id, n.net_id)]['data']['inf_data'].demand = float(v[1][4])
        for file in files:
            if file == 'N' + str(k) + '_Arcs.txt':
                with open(file_dir + file) as f:
                    # print "Opened",file,"."
                    data = pd.read_csv(f, delimiter='\t', header=None)
                    for v in data.iterrows():
                        for duplicate in range(2):
                            if duplicate == 0:
                                a = InfrastructureArc(int(v[1][0]), int(v[1][1]), k)
                            elif duplicate == 1:
                                a = InfrastructureArc(int(v[1][1]), int(v[1][0]), k)
                            G.G.add_edge((a.source, a.layer), (a.dest, a.layer), data={'inf_data': a})
                            G.G[(a.source, a.layer)][(a.dest, a.layer)]['data']['inf_data'].flow_cost = float(
                                v[1][2]) * cost_scale
                            G.G[(a.source, a.layer)][(a.dest, a.layer)]['data']['inf_data'].reconstruction_cost = float(
                                v[1][4]) * cost_scale
                            G.G[(a.source, a.layer)][(a.dest, a.layer)]['data']['inf_data'].capacity = float(v[1][2])
                            # !!! Assume only one kind of resource for now and one resource for each repaired element.
                            G.G[(a.source, a.layer)][(a.dest, a.layer)]['data']['inf_data'].resource_usage['h_'] = 1
    for k in range(1, no_layers + 1):
        for kt in range(1, no_layers + 1):
            if k != kt:
                for file in files:
                    if file == 'Interdependent_Arcs_' + str(k) + '_' + str(kt) + '.txt':
                        with open(file_dir + file) as f:
                            #                print "Opened",file,"."
                            try:
                                data = pd.read_csv(f, delimiter='\t', header=None)
                                for v in data.iterrows():
                                    i = int(v[1][0])
                                    net_i = k
                                    j = int(v[1][1])
                                    net_j = kt
                                    a = InfrastructureInterdepArc(i, j, net_i, net_j, gamma=1.0)
                                    G.G.add_edge((a.source, a.source_layer), (a.dest, a.dest_layer),
                                                 data={'inf_data': a})
                                    # import random
                                    # if net_i == 1 and random.uniform(0, 1) > 0.5:
                                    #     pass
                                    # else:
                                    #     a = InfrastructureInterdepArc(i, j, net_i, net_j, gamma=1.0)
                                    #     G.G.add_edge((a.source, a.source_layer), (a.dest, a.dest_layer),
                                    #                  data={'inf_data': a})
                            except:
                                print('Empty file: ' + file)
    # add subspace data #.. todo: Add geographical spaces to synthetic networks
    # for v in data.iterrows():
    #    if fname=='beta':
    #        net = netNames[v[1]['Network']]
    #        G.G[(int(v[1]['Start Node']),net)][(int(v[1]['End Node']),net)]['data']['inf_data'].space=int(int(v[1]['Subspace']))
    #        G.G[(int(v[1]['End Node']),net)][(int(v[1]['Start Node']),net)]['data']['inf_data'].space=int(int(v[1]['Subspace']))
    #    if fname=='alpha':
    #        net = netNames[v[1]['Network']]
    #        G.G.node[(int(v[1]['ID']),net)]['data']['inf_data'].space=int(int(v[1]['Subspace']))
    #    if fname=='g':
    #        G.S.append(InfrastructureSpace(int(v[1]['Subspace_ID']),float(v[1]['g'])))

    return G, no_resource, range(1, no_layers + 1)


def add_synthetic_failure_scenario(G, dam_dir, topology='Random', config=0, sample=0):
    """
    This function reads initial damage data from file for a synthetic network.

    Parameters
    ----------
    G : networkx.DiGraph
        The networkx graph object that stores node, arc, and interdependency information.
    dam_dir : str
        The address of the folder where the initial damage data are stored.
    topology : str
        Topology of the layers of the synthetic network. The default in 'Random'.
    config : int
        Configuration number of the initial damage scenario. The default is 6.
    sample : int
        Sample number of the initial damage scenario. The default is 0.

    Returns
    -------
    None.

    """
    net_dir = dam_dir + '/' + topology + 'Networks/'
    topo_initial = {'Random': 'RN', 'ScaleFree': 'SFN', 'Grid': 'GN', 'General': 'GEN'}
    with open(net_dir + 'List_of_Configurations.txt') as f:
        config_data = pd.read_csv(f, delimiter='\t')
        config_data = config_data.rename(columns=lambda x: x.strip())
    config_param = config_data.iloc[config]
    no_layers = int(config_param.loc['No. Layers'])
    # noResource = int(config_param.loc['Resource Cap'])
    file_dir = net_dir + topo_initial[topology] + 'Config_' + str(config) + '/Sample_' + str(sample) + '/'
    files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
    for k in range(1, no_layers + 1):
        for file in files:
            if file == 'N' + str(k) + '_Damaged_Nodes.txt':
                with open(file_dir + file) as f:
                    try:
                        data = pd.read_csv(f, delimiter='\t', header=None)
                        for v in data.iterrows():
                            G.G.nodes[(v[1][0], k)]['data']['inf_data'].functionality = 0.0
                            G.G.nodes[(v[1][0], k)]['data']['inf_data'].repaired = 0.0
                            # print("Node (",`int(v[1][0])`+","+`k`+") broken.")
                    except:
                        print('Empty file: ' + file)

        for file in files:
            if file == 'N' + str(k) + '_Damaged_Arcs.txt':
                with open(file_dir + file) as f:
                    try:
                        data = pd.read_csv(f, delimiter='\t', header=None)
                        for a in data.iterrows():
                            G.G[(a[1][0], k)][(a[1][1], k)]['data']['inf_data'].functionality = 0.0
                            G.G[(a[1][0], k)][(a[1][1], k)]['data']['inf_data'].repaired = 0.0
                            # print "Arc ((",`int(a[1][0])`+","+`k`+"),("+`int(a[1][1])`+","+`k`+")) broken."

                            G.G[(a[1][1], k)][(a[1][0], k)]['data']['inf_data'].functionality = 0.0
                            G.G[(a[1][1], k)][(a[1][0], k)]['data']['inf_data'].repaired = 0.0
                            # print "Arc ((",`int(a[1][1])`+","+`k`+"),("+`int(a[1][0])`+","+`k`+")) broken."
                    except:
                        print('Empty file: ' + file)
