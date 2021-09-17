from infrastructure import *
from indputils import *
from gurobipy import GRB, Model, LinExpr
import string
import networkx as nx
import matplotlib.pyplot as plt
import copy
import time
import dislocationutils


def inmrp(N, v_r, T=1, layers=None, controlled_layers=None, functionality=None, forced_actions=False, fixed_nodes=None,
         print_cmd=True, time_limit=None, co_location=True, solution_pool=None):
    """
    INMRP optimization problem.

    Parameters
    ----------
    N : :class:`~infrastructure.InfrastructureNetwork`
        An InfrastructureNetwork instance.
    v_r : dict
        Dictionary of the number of resources of different types in the analysis.
        If the value is a scale for a type, it shows the total number of resources of that type for all layers .
        If the value is a list for a type, it shows the total number of resources of that type given to each layer.
    T : int, optional
        Number of time steps to optimize over. T=1 shows an iINDP analysis and T>1 shows a td-INDP. The default is 1.
    layers : list, optional
        Layer IDs in N included in the optimization.
    controlled_layers : list, optional
        Layer IDs that can be recovered in this optimization. Used for decentralized optimization. The default is None.
    functionality : dict, optional
        Dictionary of nodes to functionality values for non-controlled nodes.
        Used for decentralized optimization. The default is None.
    forced_actions : bool, optional
        If true, it forces the optimizer to repair at least one element. The default is False.
    fixed_nodes : dict, optional
        It fixes the functionality of given elements to a given value. The default is None.
    print_cmd : bool, optional
        If true, analysis information is written to the console. The default is True.
    time_limit : int, optional
        Time limit for the optimizer to stop. The default is None.
    co_location : bool, optional
        If false, exclude geographical interdependency from the optimization. The default is True.
    solution_pool : int, optional
        The number of solutions that should be retrieved from the optimizer in addition to
        the optimal one. The default is None.

    Returns
    -------
    : list
    A list of the form ``[m, results]`` for a successful optimization where m is the Gurobi  optimization model and
        results is a :class:`~indputils.INDPResults` object generated using  :func:`collect_results`.
        If :envvar:`solution_pool` is set to a number, the function returns ``[m, results,  sol_pool_results]`` where
        `sol_pool_results` is dictionary of solution that should be retrieved from the optimizer in addition to the
        optimal one collected using :func:`collect_solution_pool`.

    """
    if fixed_nodes is None:
        fixed_nodes = {}
    if functionality is None:
        functionality = {}
    if layers is None:
        layers = [1, 2, 3]
    if controlled_layers is None:
        controlled_layers = layers

    start_time = time.time()
    m = Model('indp')
    m.setParam('OutputFlag', False)
    if time_limit:
        m.setParam('TimeLimit', time_limit)
    g_prime_nodes = [n[0] for n in N.G.nodes(data=True) if n[1]['data']['inf_data'].net_id in layers]
    g_prime = N.G.subgraph(g_prime_nodes)
    # Damaged nodes in whole network
    n_prime = [n for n in g_prime.nodes(data=True) if n[1]['data']['inf_data'].repaired == 0.0]
    # Nodes in controlled network.
    n_hat_nodes = [n[0] for n in g_prime.nodes(data=True) if n[1]['data']['inf_data'].net_id in controlled_layers]
    n_hat = g_prime.subgraph(n_hat_nodes)
    # Damaged nodes in controlled network.
    n_hat_prime = [n for n in n_hat.nodes(data=True) if n[1]['data']['inf_data'].repaired == 0.0]
    # Damaged arcs in whole network
    a_prime = [(u, v, a) for u, v, a in g_prime.edges(data=True) if a['data']['inf_data'].functionality == 0.0]
    # Damaged arcs in controlled network.
    a_hat_prime = [(u, v, a) for u, v, a in a_prime if n_hat.has_node(u) and n_hat.has_node(v)]
    S = N.S
    # Populate interdependencies. Add nodes to N' if they currently rely on a non-functional node.
    interdep_nodes = {}
    for u, v, a in g_prime.edges(data=True):
        if not functionality:
            if a['data']['inf_data'].is_interdep and g_prime.nodes[u]['data']['inf_data'].functionality == 0.0:
                # print "Dependency edge goes from:",u,"to",v
                if v not in interdep_nodes:
                    interdep_nodes[v] = []
                interdep_nodes[v].append((u, a['data']['inf_data'].gamma))
        else:
            # Should populate n_hat with layers that are controlled. Then go through n_hat.edges(data=True)
            # to find interdependencies.
            for t in range(T):
                if t not in interdep_nodes:
                    interdep_nodes[t] = {}
                if n_hat.has_node(v) and a['data']['inf_data'].is_interdep:
                    if functionality[t][u] == 0.0:
                        if v not in interdep_nodes[t]:
                            interdep_nodes[t][v] = []
                        interdep_nodes[t][v].append((u, a['data']['inf_data'].gamma))

    for t in range(T):
        # Add geographical space variables.
        if co_location:
            for s in S:
                m.addVar(name='z_' + str(s.id) + "," + str(t), vtype=GRB.BINARY)
        # Add over/under-supply variables for each node.
        for n, d in n_hat.nodes(data=True):
            m.addVar(name='delta+_' + str(n) + "," + str(t), lb=0.0)
            m.addVar(name='delta-_' + str(n) + "," + str(t), lb=0.0)
            for l in d['data']['inf_data'].extra_com.keys():
                m.addVar(name='delta+_' + str(n) + "," + str(t) + "," + str(l), lb=0.0)
                m.addVar(name='delta-_' + str(n) + "," + str(t) + "," + str(l), lb=0.0)
        # Add functionality binary variables for each node in N'.
        for n, d in n_hat.nodes(data=True):
            m.addVar(name='w_' + str(n) + "," + str(t), vtype=GRB.BINARY)
            if T > 1:
                m.addVar(name='w_tilde_' + str(n) + "," + str(t), vtype=GRB.BINARY)
                # Fix node values (only for iINDP)
        m.update()
        for key, val in fixed_nodes.items():
            m.getVarByName('w_' + str(key) + "," + str(0)).lb = val
            m.getVarByName('w_' + str(key) + "," + str(0)).ub = val
        # Add flow variables for each arc. (main commodity)
        for u, v, a in n_hat.edges(data=True):
            m.addVar(name='x_' + str(u) + "," + str(v) + "," + str(t), lb=0.0)
            for l in a['data']['inf_data'].extra_com.keys():
                m.addVar(name='x_' + str(u) + "," + str(v) + "," + str(t) + "," + str(l), lb=0.0)
        # Add functionality binary variables for each arc in A'.
        for u, v, a in a_hat_prime:
            m.addVar(name='y_' + str(u) + "," + str(v) + "," + str(t), vtype=GRB.BINARY)
            if T > 1:
                m.addVar(name='y_tilde_' + str(u) + "," + str(v) + "," + str(t), vtype=GRB.BINARY)
    m.update()

    # Populate objective function.
    obj_func = LinExpr()
    for t in range(T):
        if co_location:
            for s in S:
                obj_func += s.cost * m.getVarByName('z_' + str(s.id) + "," + str(t))
        for u, v, a in a_hat_prime:
            if T == 1:
                obj_func += (float(a['data']['inf_data'].reconstruction_cost) / 2.0) * m.getVarByName(
                    'y_' + str(u) + "," + str(v) + "," + str(t))
            else:
                obj_func += (float(a['data']['inf_data'].reconstruction_cost) / 2.0) * m.getVarByName(
                    'y_tilde_' + str(u) + "," + str(v) + "," + str(t))
        for n, d in n_hat_prime:
            if T == 1:
                obj_func += d['data']['inf_data'].reconstruction_cost * m.getVarByName('w_' + str(n) + "," + str(t))
            else:
                obj_func += d['data']['inf_data'].reconstruction_cost * m.getVarByName(
                    'w_tilde_' + str(n) + "," + str(t))
        for n, d in n_hat.nodes(data=True):
            obj_func += d['data']['inf_data'].oversupply_penalty * m.getVarByName('delta+_' + str(n) + "," + str(t))
            obj_func += d['data']['inf_data'].undersupply_penalty * m.getVarByName('delta-_' + str(n) + "," + str(t))
            for l, val in d['data']['inf_data'].extra_com.items():
                obj_func += val['oversupply_penalty'] * m.getVarByName('delta+_' + str(n) + "," + str(t) + "," + str(l))
                obj_func += val['undersupply_penalty'] * m.getVarByName(
                    'delta-_' + str(n) + "," + str(t) + "," + str(l))

        for u, v, a in n_hat.edges(data=True):
            obj_func += a['data']['inf_data'].flow_cost * m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t))
            for l, val in a['data']['inf_data'].extra_com.items():
                obj_func += val['flow_cost'] * m.getVarByName(
                    'x_' + str(u) + "," + str(v) + "," + str(t) + "," + str(l))

    m.setObjective(obj_func, GRB.MINIMIZE)
    m.update()

    # Constraints.
    # Time-dependent constraints.
    if T > 1:
        for n, d in n_hat_prime:
            m.addConstr(m.getVarByName('w_' + str(n) + ",0"), GRB.EQUAL, 0,
                        "Initial state at node " + str(n) + "," + str(0))
        for u, v, a in a_hat_prime:
            m.addConstr(m.getVarByName('y_' + str(u) + "," + str(v) + ",0"), GRB.EQUAL, 0,
                        "Initial state at arc " + str(u) + "," + str(v) + "," + str(0))

    for t in range(T):
        # Time-dependent constraint.
        for n, d in n_hat_prime:
            if t > 0:
                w_tilde_sum = LinExpr()
                for t_prime in range(1, t + 1):
                    w_tilde_sum += m.getVarByName('w_tilde_' + str(n) + "," + str(t_prime))
                m.addConstr(m.getVarByName('w_' + str(n) + "," + str(t)), GRB.LESS_EQUAL, w_tilde_sum,
                            "Time dependent recovery constraint at node " + str(n) + "," + str(t))
        for u, v, a in a_hat_prime:
            if t > 0:
                y_tilde_sum = LinExpr()
                for t_prime in range(1, t + 1):
                    y_tilde_sum += m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t_prime))
                m.addConstr(m.getVarByName('y_' + str(u) + "," + str(v) + "," + str(t)), GRB.LESS_EQUAL, y_tilde_sum,
                            "Time dependent recovery constraint at arc " + str(u) + "," + str(v) + "," + str(t))
        # Enforce a_i,j to be fixed if a_j,i is fixed (and vice versa).
        for u, v, a in a_hat_prime:
            # print u,",",v
            m.addConstr(m.getVarByName('y_' + str(u) + "," + str(v) + "," + str(t)), GRB.EQUAL,
                        m.getVarByName('y_' + str(v) + "," + str(u) + "," + str(t)),
                        "Arc reconstruction equality (" + str(u) + "," + str(v) + "," + str(t) + ")")
            if T > 1:
                m.addConstr(m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t)), GRB.EQUAL,
                            m.getVarByName('y_tilde_' + str(v) + "," + str(u) + "," + str(t)),
                            "Arc reconstruction equality (" + str(u) + "," + str(v) + "," + str(t) + ")")
        # Conservation of flow constraint. (2) in INDP paper.
        for n, d in n_hat.nodes(data=True):
            out_flow_constr = LinExpr()
            in_flow_constr = LinExpr()
            demand_constr = LinExpr()
            for u, v, a in n_hat.out_edges(n, data=True):
                out_flow_constr += m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t))
            for u, v, a in n_hat.in_edges(n, data=True):
                in_flow_constr += m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t))
            demand_constr += d['data']['inf_data'].demand - m.getVarByName(
                'delta+_' + str(n) + "," + str(t)) + m.getVarByName('delta-_' + str(n) + "," + str(t))
            m.addConstr(out_flow_constr - in_flow_constr, GRB.EQUAL, demand_constr,
                        "Flow conservation constraint " + str(n) + "," + str(t))
            for l, val in d['data']['inf_data'].extra_com.items():
                out_flow_constr = LinExpr()
                in_flow_constr = LinExpr()
                demand_constr = LinExpr()
                for u, v, a in n_hat.out_edges(n, data=True):
                    out_flow_constr += m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t) + "," + str(l))
                for u, v, a in n_hat.in_edges(n, data=True):
                    in_flow_constr += m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t) + "," + str(l))
                demand_constr += val['demand'] - \
                                 m.getVarByName('delta+_' + str(n) + "," + str(t) + "," + str(l)) + \
                                 m.getVarByName('delta-_' + str(n) + "," + str(t) + "," + str(l))
                m.addConstr(out_flow_constr - in_flow_constr, GRB.EQUAL, demand_constr,
                            "Flow conservation constraint " + str(n) + "," + str(t) + "," + str(l))

        # Flow functionality constraints.
        if not functionality:
            interdep_nodes_list = interdep_nodes.keys()  # Interdependent nodes with a damaged dependee node
        else:
            interdep_nodes_list = interdep_nodes[t].keys()  # Interdependent nodes with a damaged dependee node
        for u, v, a in n_hat.edges(data=True):
            lhs = m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t)) + \
                  sum([m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t) + "," + str(l)) \
                       for l in a['data']['inf_data'].extra_com.keys()])
            if (u in [n for (n, d) in n_hat_prime]) | (u in interdep_nodes_list):
                m.addConstr(lhs, GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * m.getVarByName('w_' + str(u) + "," + str(t)),
                            "Flow in functionality constraint(" + str(u) + "," + str(v) + "," + str(t) + ")")
            else:
                m.addConstr(lhs, GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * N.G.nodes[u]['data']['inf_data'].functionality,
                            "Flow in functionality constraint (" + str(u) + "," + str(v) + "," + str(t) + ")")
            if (v in [n for (n, d) in n_hat_prime]) | (v in interdep_nodes_list):
                m.addConstr(lhs, GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * m.getVarByName('w_' + str(v) + "," + str(t)),
                            "Flow out functionality constraint(" + str(u) + "," + str(v) + "," + str(t) + ")")
            else:
                m.addConstr(lhs, GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * N.G.nodes[v]['data']['inf_data'].functionality,
                            "Flow out functionality constraint (" + str(u) + "," + str(v) + "," + str(t) + ")")
            if (u, v, a) in a_hat_prime:
                m.addConstr(lhs, GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * m.getVarByName(
                                'y_' + str(u) + "," + str(v) + "," + str(t)),
                            "Flow arc functionality constraint (" + str(u) + "," + str(v) + "," + str(t) + ")")
            else:
                m.addConstr(lhs, GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * N.G[u][v]['data']['inf_data'].functionality,
                            "Flow arc functionality constraint(" + str(u) + "," + str(v) + "," + str(t) + ")")

        # Resource availability constraints.
        for rc, val in v_r.items():
            is_sep_res = False
            if isinstance(val, int):
                total_resource = val
            else:
                is_sep_res = True
                total_resource = sum([lval for _, lval in val.items()])
                assert len(val.keys()) == len(layers), "The number of resource \
                    values does not match the number of layers."

            resource_left_constr = LinExpr()
            if is_sep_res:
                res_left_constr_sep = {key: LinExpr() for key in val.keys()}

            for u, v, a in a_hat_prime:
                idx_lyr = a['data']['inf_data'].layer
                res_use = 0.5 * a['data']['inf_data'].resource_usage['h_' + rc]
                if T == 1:
                    resource_left_constr += res_use * m.getVarByName('y_' + str(u) + "," + str(v) + "," + str(t))
                    if is_sep_res:
                        res_left_constr_sep[idx_lyr] += res_use * m.getVarByName(
                            'y_' + str(u) + "," + str(v) + "," + str(t))
                else:
                    resource_left_constr += res_use * m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t))
                    if is_sep_res:
                        res_left_constr_sep[idx_lyr] += res_use * m.getVarByName(
                            'y_tilde_' + str(u) + "," + str(v) + "," + str(t))

            for n, d in n_hat_prime:
                idx_lyr = n[1]
                res_use = d['data']['inf_data'].resource_usage['p_' + rc]
                if T == 1:
                    resource_left_constr += res_use * m.getVarByName('w_' + str(n) + "," + str(t))
                    if is_sep_res:
                        res_left_constr_sep[idx_lyr] += res_use * m.getVarByName('w_' + str(n) + "," + str(t))
                else:
                    resource_left_constr += res_use * m.getVarByName('w_tilde_' + str(n) + "," + str(t))
                    if is_sep_res:
                        res_left_constr_sep[idx_lyr] += res_use * m.getVarByName('w_tilde_' + str(n) + "," + str(t))

            m.addConstr(resource_left_constr, GRB.LESS_EQUAL, total_resource,
                        "Resource availability constraint for " + rc + " at " + str(t) + ".")
            if is_sep_res:
                for k, lval in val.items():
                    m.addConstr(res_left_constr_sep[k], GRB.LESS_EQUAL, lval,
                                "Resource availability constraint for " + rc + " at " + \
                                str(t) + " for layer " + str(k) + ".")

        # Interdependency constraints
        infeasible_actions = []
        for n, d in n_hat.nodes(data=True):
            if not functionality:
                if n in interdep_nodes:
                    interdep_l_constr = LinExpr()
                    interdep_r_constr = LinExpr()
                    for interdep in interdep_nodes[n]:
                        src = interdep[0]
                        gamma = interdep[1]
                        if not n_hat.has_node(src):
                            infeasible_actions.append(n)
                            interdep_l_constr += 0
                        else:
                            interdep_l_constr += m.getVarByName('w_' + str(src) + "," + str(t)) * gamma
                    interdep_r_constr += m.getVarByName('w_' + str(n) + "," + str(t))
                    m.addConstr(interdep_l_constr, GRB.GREATER_EQUAL, interdep_r_constr,
                                "Interdependency constraint for node " + str(n) + "," + str(t))
            else:
                if n in interdep_nodes[t]:
                    # print interdep_nodes[t]
                    interdep_l_constr = LinExpr()
                    interdep_r_constr = LinExpr()
                    for interdep in interdep_nodes[t][n]:
                        src = interdep[0]
                        gamma = interdep[1]
                        if not n_hat.has_node(src):
                            if print_cmd:
                                print("Forcing", str(n), "to be 0 (dep. on", str(src), ")")
                            infeasible_actions.append(n)
                            interdep_l_constr += 0
                        else:
                            interdep_l_constr += m.getVarByName('w_' + str(src) + "," + str(t)) * gamma
                    interdep_r_constr += m.getVarByName('w_' + str(n) + "," + str(t))
                    m.addConstr(interdep_l_constr, GRB.GREATER_EQUAL, interdep_r_constr,
                                "Interdependency constraint for node " + str(n) + "," + str(t))

        # Forced actions (if applicable)
        if forced_actions:
            recovery_sum = LinExpr()
            feasible_nodes = [(n, d) for n, d in n_hat_prime if n not in infeasible_actions]
            if len(feasible_nodes) + len(a_hat_prime) > 0:
                for n, d in feasible_nodes:
                    if T == 1:
                        recovery_sum += m.getVarByName('w_' + str(n) + "," + str(t))
                    else:
                        recovery_sum += m.getVarByName('w_tilde_' + str(n) + "," + str(t))
                for u, v, a in a_hat_prime:
                    if T == 1:
                        recovery_sum += m.getVarByName('y_' + str(u) + "," + str(v) + "," + str(t))
                    else:
                        recovery_sum += m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t))
                m.addConstr(recovery_sum, GRB.GREATER_EQUAL, 1, "Forced action constraint")

        # Geographic space constraints
        if co_location:
            for s in S:
                for n, d in n_hat_prime:
                    if d['data']['inf_data'].in_space(s.id):
                        if T == 1:
                            m.addConstr(
                                m.getVarByName('w_' + str(n) + "," + str(t)) * d['data']['inf_data'].in_space(s.id),
                                GRB.LESS_EQUAL, m.getVarByName('z_' + str(s.id) + "," + str(t)),
                                "Geographical space constraint for node " + str(n) + "," + str(t))
                        else:
                            m.addConstr(
                                m.getVarByName('w_tilde_' + str(n) + "," + str(t)) * d['data']['inf_data'].in_space(
                                    s.id), GRB.LESS_EQUAL, m.getVarByName('z_' + str(s.id) + "," + str(t)),
                                "Geographical space constraint for node " + str(n) + "," + str(t))
                for u, v, a in a_hat_prime:
                    if a['data']['inf_data'].in_space(s.id):
                        if T == 1:
                            m.addConstr(m.getVarByName('y_' + str(u) + "," + str(v) + "," + str(t)) * a['data'][
                                'inf_data'].in_space(s.id), GRB.LESS_EQUAL,
                                        m.getVarByName('z_' + str(s.id) + "," + str(t)),
                                        "Geographical space constraint for arc (" + str(u) + "," + str(v) + ")")
                        else:
                            m.addConstr(m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t)) * a['data'][
                                'inf_data'].in_space(s.id), GRB.LESS_EQUAL,
                                        m.getVarByName('z_' + str(s.id) + "," + str(t)),
                                        "Geographical space constraint for arc (" + str(u) + "," + str(v) + ")")
        # # Demand completion constraints for dependee nodes.
        # all_dependees = [y[0] for x in interdep_nodes.values() for y in x]
        # for n, d in n_hat.nodes(data=True):
        #     if n in all_dependees:
        #         if T == 1:  # !!! add this constraint for td-INDP
        #             dc_lhs = abs(d['data']['inf_data'].demand) * m.getVarByName('w_' + str(n) + "," + str(t))
        #             dc_rhs = abs(d['data']['inf_data'].demand) - m.getVarByName('delta-_' + str(n) + "," + str(t))
        #             m.addConstr(dc_lhs, GRB.LESS_EQUAL, dc_rhs,
        #                         "Demand completion constraints for node " + str(n) + "," + str(t))
        #             for l, val in d['data']['inf_data'].extra_com.items():
        #                 dc_lhs = abs(val['demand']) * m.getVarByName('w_' + str(n) + "," + str(t))
        #                 dc_rhs = abs(val['demand']) - m.getVarByName('delta-_' + str(n) + "," + str(t) + "," + str(l))
        #                 m.addConstr(dc_lhs, GRB.LESS_EQUAL, dc_rhs,
        #                             "Demand completion constraints for node " + str(n) + "," + str(t) + "," + str(l))
    #    print "Solving..."
    m.update()
    if solution_pool:
        m.setParam('PoolSearchMode', 1)
        m.setParam('PoolSolutions', 10000)
        m.setParam('PoolGap', solution_pool)
    m.optimize()
    run_time = time.time() - start_time
    # Save results.
    if m.getAttr("Status") == GRB.OPTIMAL or m.status == 9:
        if m.status == 9:
            print('\nOptimizer time limit, gap = %1.3f\n' % m.MIPGap)
        results = collect_results(m, controlled_layers, T, n_hat, n_hat_prime, a_hat_prime, S, coloc=co_location)
        results.add_run_time(t, run_time)
        if solution_pool:
            sol_pool_results = collect_solution_pool(m, T, n_hat_prime, a_hat_prime)
            return [m, results, sol_pool_results]
        return [m, results]
    else:
        m.computeIIS()
        if m.status == 3:
            m.write("model.ilp")
            print(m.getAttr("Status"), ": SOLUTION NOT FOUND. (Check data and/or violated constraints).")
            print('\nThe following constraint(s) cannot be satisfied:')
            for c in m.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
        return None


def collect_results(m, controlled_layers, T, n_hat, n_hat_prime, a_hat_prime, S, coloc=True):
    """
    This function computes the results (actions and costs) of the optimal results and write them to a
    :class:`~indputils.INDPResults` object.

    Parameters
    ----------
    m : gurobi.Model
        The object containing the solved optimization problem.
    controlled_layers : list
        Layer IDs that can be recovered in this optimization.
    T : int
        Number of time steps in the optimization (T=1 for iINDP, and T>=1 for td-INDP).
    n_hat : list
        List of Damaged nodes in whole network.
    n_hat_prime : list
        List of damaged nodes in controlled network.
    a_hat_prime : list
        List of damaged arcs in controlled network.
    S : list
        List of geographical sites.
    coloc : bool, optional
        If false, exclude geographical interdependency from the results. . The default is True.

    Returns
    -------
    indp_results : INDPResults
    A :class:`~indputils.INDPResults` object containing the optimal restoration decisions.

    """
    layers = controlled_layers
    indp_results = INDPResults(layers)
    # compute total demand of all layers and each layer
    total_demand = 0.0
    total_demand_layer = {l: 0.0 for l in layers}
    for n, d in n_hat.nodes(data=True):
        demand_value = d['data']['inf_data'].demand
        if demand_value < 0:
            total_demand += demand_value
            total_demand_layer[n[1]] += demand_value
    for t in range(T):
        node_cost = 0.0
        arc_cost = 0.0
        flow_cost = 0.0
        over_supp_cost = 0.0
        under_supp_cost = 0.0
        under_supp = 0.0
        space_prep_cost = 0.0
        node_cost_layer = {l: 0.0 for l in layers}
        arc_cost_layer = {l: 0.0 for l in layers}
        flow_cost_layer = {l: 0.0 for l in layers}
        over_supp_cost_layer = {l: 0.0 for l in layers}
        under_supp_cost_layer = {l: 0.0 for l in layers}
        under_supp_layer = {l: 0.0 for l in layers}
        space_prep_cost_layer = {l: 0.0 for l in layers}  # !!! populate this for each layer
        # Record node recovery actions.
        for n, d in n_hat_prime:
            node_var = 'w_tilde_' + str(n) + "," + str(t)
            if T == 1:
                node_var = 'w_' + str(n) + "," + str(t)
            if round(m.getVarByName(node_var).x) == 1:
                action = str(n[0]) + "." + str(n[1])
                indp_results.add_action(t, action)
        # Record edge recovery actions.
        for u, v, a in a_hat_prime:
            arc_var = 'y_tilde_' + str(u) + "," + str(v) + "," + str(t)
            if T == 1:
                arc_var = 'y_' + str(u) + "," + str(v) + "," + str(t)
            if round(m.getVarByName(arc_var).x) == 1:
                action = str(u[0]) + "." + str(u[1]) + "/" + str(v[0]) + "." + str(v[1])
                indp_results.add_action(t, action)
        # Calculate space preparation costs.
        if coloc:
            for s in S:
                space_prep_cost += s.cost * m.getVarByName('z_' + str(s.id) + "," + str(t)).x
        indp_results.add_cost(t, "Space Prep", space_prep_cost, space_prep_cost_layer)
        # Calculate arc preparation costs.
        for u, v, a in a_hat_prime:
            arc_var = 'y_tilde_' + str(u) + "," + str(v) + "," + str(t)
            if T == 1:
                arc_var = 'y_' + str(u) + "," + str(v) + "," + str(t)
            arc_cost += (a['data']['inf_data'].reconstruction_cost / 2.0) * m.getVarByName(arc_var).x
            arc_cost_layer[u[1]] += (a['data']['inf_data'].reconstruction_cost / 2.0) * m.getVarByName(arc_var).x
        indp_results.add_cost(t, "Arc", arc_cost, arc_cost_layer)
        # Calculate node preparation costs.
        for n, d in n_hat_prime:
            node_var = 'w_tilde_' + str(n) + "," + str(t)
            if T == 1:
                node_var = 'w_' + str(n) + "," + str(t)
            node_cost += d['data']['inf_data'].reconstruction_cost * m.getVarByName(node_var).x
            node_cost_layer[n[1]] += d['data']['inf_data'].reconstruction_cost * m.getVarByName(node_var).x
        indp_results.add_cost(t, "Node", node_cost, node_cost_layer)
        # Calculate under/oversupply costs.
        for n, d in n_hat.nodes(data=True):
            over_supp_cost += d['data']['inf_data'].oversupply_penalty * m.getVarByName(
                'delta+_' + str(n) + "," + str(t)).x
            over_supp_cost_layer[n[1]] += d['data']['inf_data'].oversupply_penalty * m.getVarByName(
                'delta+_' + str(n) + "," + str(t)).x
            under_supp += m.getVarByName('delta-_' + str(n) + "," + str(t)).x
            under_supp_layer[n[1]] += m.getVarByName('delta-_' + str(n) + "," + str(t)).x / total_demand_layer[n[1]]
            under_supp_cost += d['data']['inf_data'].undersupply_penalty * m.getVarByName(
                'delta-_' + str(n) + "," + str(t)).x
            under_supp_cost_layer[n[1]] += d['data']['inf_data'].undersupply_penalty * m.getVarByName(
                'delta-_' + str(n) + "," + str(t)).x
        indp_results.add_cost(t, "Over Supply", over_supp_cost, over_supp_cost_layer)
        indp_results.add_cost(t, "Under Supply", under_supp_cost, under_supp_cost_layer)
        indp_results.add_cost(t, "Under Supply Perc", under_supp / total_demand, under_supp_layer)
        # Calculate flow costs.
        for u, v, a in n_hat.edges(data=True):
            flow_cost += a['data']['inf_data'].flow_cost * m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t)).x
            flow_cost_layer[u[1]] += a['data']['inf_data'].flow_cost * m.getVarByName(
                'x_' + str(u) + "," + str(v) + "," + str(t)).x
        indp_results.add_cost(t, "Flow", flow_cost, flow_cost_layer)
        # Calculate total costs.
        total_lyr = {}
        total_nd_lyr = {}
        for l in layers:
            total_lyr[l] = flow_cost_layer[l] + arc_cost_layer[l] + node_cost_layer[l] + \
                           over_supp_cost_layer[l] + under_supp_cost_layer[l] + space_prep_cost_layer[l]
            total_nd_lyr[l] = space_prep_cost_layer[l] + arc_cost_layer[l] + flow_cost + node_cost_layer[l]
        indp_results.add_cost(t, "Total", flow_cost + arc_cost + node_cost + over_supp_cost + \
                              under_supp_cost + space_prep_cost, total_lyr)
        indp_results.add_cost(t, "Total no disconnection", space_prep_cost + arc_cost + \
                              flow_cost + node_cost, total_nd_lyr)
    return indp_results


def collect_solution_pool(m, T, n_hat_prime, a_hat_prime):
    """
    This function collect the result (list of repaired nodes and arcs) for all feasible solutions in the solution pool

    Parameters
    ----------
    m : gurobi.Model
        The object containing the solved optimization problem.
    T : int
        Number of time steps in the optimization (T=1 for iINDP, and T>=1 for td-INDP).
    n_hat_prime : list
        List of damaged nodes in controlled network.
    a_hat_prime : list
        List of damaged arcs in controlled network.

    Returns
    -------
    sol_pool_results : dict
    A dictionary containing one dictionary per solution that contain list of repaired node and arcs in the solution.

    """
    sol_pool_results = {}
    current_sol_count = 0
    for sol in range(m.SolCount):
        m.setParam('SolutionNumber', sol)
        # print(m.PoolObjVal)
        sol_pool_results[sol] = {'nodes': [], 'arcs': []}
        for t in range(T):
            # Record node recovery actions.
            for n, d in n_hat_prime:
                node_var = 'w_tilde_' + str(n) + "," + str(t)
                if T == 1:
                    node_var = 'w_' + str(n) + "," + str(t)
                if round(m.getVarByName(node_var).xn) == 1:
                    sol_pool_results[sol]['nodes'].append(n)
            # Record edge recovery actions.
            for u, v, a in a_hat_prime:
                arc_var = 'y_tilde_' + str(u) + "," + str(v) + "," + str(t)
                if T == 1:
                    arc_var = 'y_' + str(u) + "," + str(v) + "," + str(t)
                if round(m.getVarByName(arc_var).x) == 1:
                    sol_pool_results[sol]['arcs'].append((u, v))
        if sol > 0 and sol_pool_results[sol] == sol_pool_results[current_sol_count]:
            del sol_pool_results[sol]
        elif sol > 0:
            current_sol_count = sol
    return sol_pool_results


