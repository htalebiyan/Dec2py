from infrastructure import *
from indputils import *
from gurobipy import GRB, Model, LinExpr
import string
import networkx as nx
import matplotlib.pyplot as plt
import copy
import time
import dislocationutils


def indp(N, v_r, T=1, layers=None, controlled_layers=None, functionality=None, forced_actions=False, fixed_nodes=None,
         print_cmd=True, time_limit=None, co_location=True, solution_pool=None):
    """
    INDP optimization problem. It also solves td-INDP if T > 1.

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


def apply_recovery(N, indp_results, t):
    """
    This function applies the restoration decisions (solution of INDP) to a gurobi model by changing the state of
    repaired elements to functional

    Parameters
    ----------
    N : :class:`~infrastructure.InfrastructureNetwork`
        The model of the interdependent network.
    indp_results : INDPResults
        A :class:`~indputils.INDPResults` object containing the optimal restoration decisions..
    t : int
        The time step to which the results should apply.

    Returns
    -------
    None.

    """
    for action in indp_results[t]['actions']:
        if "/" in action:
            # Edge recovery action.
            data = action.split("/")
            src = tuple([int(x) for x in data[0].split(".")])
            dst = tuple([int(x) for x in data[1].split(".")])
            N.G[src][dst]['data']['inf_data'].functionality = 1.0
        else:
            # Node recovery action.
            node = tuple([int(x) for x in action.split(".")])
            # print "Applying recovery:",node
            N.G.nodes[node]['data']['inf_data'].repaired = 1.0
            N.G.nodes[node]['data']['inf_data'].functionality = 1.0


def create_functionality_matrix(N, T, layers, actions, strategy_type="OPTIMISTIC"):
    """
    Creates a functionality map for input into the functionality parameter in the indp function.

    Parameters
    ----------
    N : :class:`~infrastructure.InfrastructureNetwork`
        An InfrastructureNetwork instance .
    T : int
        Number of time steps to optimize over.
    layers : list
        Layer IDs of N included in the optimization..
    actions : list
        An array of actions from a previous optimization. Likely taken from an
        INDPResults variable 'indp_result[t]['actions']'..
    strategy_type : str, optional
        If no actions are provided, assigns a default functionality. Options are:
        "OPTIMISTIC", "PESSIMISTIC" or "REALISTIC". The default is "OPTIMISTIC".

    Returns
    -------
    functionality : dict
    A functionality dictionary used for input into indp.

    """
    functionality = {}
    g_prime_nodes = [n[0] for n in N.G.nodes(data=True) if n[1]['data']['inf_data'].net_id in layers]
    g_prime = N.G.subgraph(g_prime_nodes)
    n_prime = [n for n in g_prime.nodes(data=True) if n[1]['data']['inf_data'].repaired == 0.0]
    for t in range(T):
        functionality[t] = {}
        functional_nodes = []
        for t_p in range(t):
            for key in functionality[t_p]:
                if functionality[t_p][key] == 1.0:
                    functionality[t][key] = 1.0
        if strategy_type == "INFO_SHARE":
            for a in actions[t]:
                if a and "/" not in a:
                    node = int(string.split(a, ".")[0])
                    layer = int(string.split(a, ".")[1])
                    if layer in layers:
                        functional_nodes.append((node, layer))
        for n, d in g_prime.nodes(data=True):
            # print "layers=",layers,"n=",n
            if d['data']['inf_data'].net_id in layers:
                if (n, d) in n_prime and n in functional_nodes:
                    functionality[t][n] = 1.0
                elif g_prime.has_node(n) and (n, d) not in n_prime:
                    functionality[t][n] = 1.0
                else:
                    if strategy_type == "OPTIMISTIC":
                        functionality[t][n] = 1.0
                    elif strategy_type == "PESSIMISTIC":
                        functionality[t][n] = 0.0
                    elif strategy_type == "REALISTIC":
                        functionality[t][n] = d['data']['inf_data'].functionality
                    else:
                        if not n in functionality[t]:
                            functionality[t][n] = 0.0
    return functionality


def initialize_network(base_dir="../data/INDP_7-20-2015/", external_interdependency_dir=None,
                       sim_number=1, cost_scale=1, magnitude=6, sample=0, v=3,
                       infrastructure_data=True, topology='Random', extra_commodity=None):
    """
     Initializes an :class:`~infrastructure.InfrastructureNetwork` object from
     Shelby County data or synthetic networks.

     Parameters
     ----------
     base_dir : str, optional
         Base directory of Shelby County data or synthetic networks. The default is
         "../data/INDP_7-20-2015/".
     external_interdependency_dir : str, optional
         Directory of external interdependencies for Shelby County data. The default is None.
     sim_number : int, optional
         Which simulation number to use as input. The default is 1.
     cost_scale : float, optional
         Scales the cost to improve efficiency. The default is 1.
     magnitude : int, optional
         Magnitude parameter of the initial disruption. The default is 6.
     sample : int, optional
         Sample number parameter of the initial disruption. The default is 0.
     v : int, list, optional
         Number of available resources or resource cap. The default is 3.
     infrastructure_data : bool, optional
         If the data are for infrastructure. It should be set to False if a synthetic network
         is being analyzed. The default is True.
     topology : str, optional
         Topology of the synthetic network that is being analyzed. The default is 'Random'.
     extra_commodity : dict, optional
        Dictionary of commodities other than the default one for each layer of the network. The default is 'None',
        which means that there is only one commodity per layer.

     Returns
     -------
     interdep_net : :class:`~infrastructure.InfrastructureNetwork`
         The object containing the network data.
     v_temp : int, list
         Number of available resources or resource cap. Used for synthetic networks
         where each sample network has a different resource cap.
     layers_temp : list
         List of layers in the analysis. Used for synthetic networks where each sample
         network has different layers.

     """
    layers_temp = []
    v_temp = 0
    if infrastructure_data:
        interdep_net = load_infrastructure_data(base_dir=base_dir,
                                                external_interdependency_dir=external_interdependency_dir,
                                                sim_number=sim_number, cost_scale=cost_scale, magnitude=magnitude, v=v,
                                                data_format=infrastructure_data, extra_commodity=extra_commodity)
    else:
        interdep_net, v_temp, layers_temp = load_synthetic_network(BASE_DIR=base_dir, topology=topology,
                                                                   config=magnitude, sample=sample,
                                                                   cost_scale=cost_scale)
    return interdep_net, {'': v_temp}, layers_temp


def run_indp(params, layers=None, controlled_layers=None, functionality=None, T=1, save=True, suffix="",
             forced_actions=False, save_model=False, print_cmd_line=True, co_location=True):
    """
    This function runs iINDP (T=1) or td-INDP for a given number of time steps and input parameters

    Parameters
    ----------
    params : dict
        Parameters that are needed to run the indp optimization.
    layers : list
        List of layers in the interdependent network. The default is 'None', which sets the list to [1, 2, 3].
    controlled_layers : list
        List of layers that are included in the analysis. The default is 'None', which sets the list equal to layers.
    functionality : dict
        This dictionary is used to assign functionality values elements in the network before the analysis starts. The
        default is 'None'.
    T : int, optional
        Number of time steps to optimize over. T=1 shows an iINDP analysis and T>1 shows a td-INDP. The default is 1.
    save : bool
        If the results should be saved to file. The default is True.
    suffix : str
        The suffix that should be added to the output files when saved. The default is ''.
    forced_actions : bool
        If True, the optimizer is forced to repair at least one element in each time step. The default is False.
    save_model : bool
        If the gurobi optimization model should be saved to file. The default is False.
    print_cmd_line : bool
        If full information about the analysis should be written to the console. The default is True.
    co_location : bool
        If co-location and geographical interdependency should be considered in the analysis. The default is True.

    Returns
    -------
    indp_results : INDPResults
    A :class:`~indputils.INDPResults` object containing the optimal restoration decisions.

    """

    # Initialize failure scenario.
    global original_N
    if functionality is None:
        functionality = {}
    if layers is None:
        layers = [1, 2, 3]
    if controlled_layers is None:
        controlled_layers = layers

    if "N" not in params:
        interdependent_net = initialize_network(base_dir="../data/INDP_7-20-2015/", sim_number=params['SIM_NUMBER'],
                                                magnitude=params["MAGNITUDE"])
    else:
        interdependent_net = params["N"]
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"] = 1

    out_dir_suffix_res = get_resource_suffix(params)
    indp_results = INDPResults(params["L"])
    if T == 1:
        print("--Running INDP (T=1) or iterative INDP.")
        if print_cmd_line:
            print("Num iters=", params["NUM_ITERATIONS"])

        # Run INDP for 1 time step (original INDP).
        output_dir = params["OUTPUT_DIR"] + '_L' + str(len(layers)) + '_m' + str(params["MAGNITUDE"]) + \
                     "_v" + out_dir_suffix_res
        # Initial calculations.
        if params['DYNAMIC_PARAMS']:
            original_N = copy.deepcopy(interdependent_net)  # !!! deepcopy
            dislocationutils.dynamic_parameters(interdependent_net, original_N, 0,
                                                params['DYNAMIC_PARAMS']['DEMAND_DATA'])
        v_0 = {x: 0 for x in params["V"].keys()}
        results = indp(interdependent_net, v_0, 1, layers, controlled_layers=controlled_layers,
                       functionality=functionality, co_location=co_location)
        indp_results = results[1]
        indp_results.add_components(0, INDPComponents.calculate_components(results[0], interdependent_net,
                                                                           layers=controlled_layers))
        for i in range(params["NUM_ITERATIONS"]):
            print("-Time Step (iINDP)", i + 1, "/", params["NUM_ITERATIONS"])
            if params['DYNAMIC_PARAMS']:
                dislocationutils.dynamic_parameters(interdependent_net, original_N, i + 1,
                                                    params['DYNAMIC_PARAMS']['DEMAND_DATA'])
            results = indp(interdependent_net, params["V"], T, layers, controlled_layers=controlled_layers,
                           forced_actions=forced_actions, co_location=co_location)
            indp_results.extend(results[1], t_offset=i + 1)
            if save_model:
                save_indp_model_to_file(results[0], output_dir + "/Model", i + 1)
            # Modify network to account for recovery and calculate components.
            apply_recovery(interdependent_net, indp_results, i + 1)
            indp_results.add_components(i + 1, INDPComponents.calculate_components(results[0], interdependent_net,
                                                                                   layers=controlled_layers))
    #            print "Num_iters=",params["NUM_ITERATIONS"]
    else:
        # td-INDP formulations. Includes "DELTA_T" parameter for sliding windows to increase
        # efficiency.
        # Edit 2/8/16: "Sliding window" now overlaps.
        num_time_windows = 1
        time_window_length = T
        is_first_iteration = True
        if "WINDOW_LENGTH" in params:
            time_window_length = params["WINDOW_LENGTH"]
            num_time_windows = T
        output_dir = params["OUTPUT_DIR"] + '_L' + str(len(layers)) + "_m" + str(
            params["MAGNITUDE"]) + "_v" + out_dir_suffix_res

        print("Running td-INDP (T=" + str(T) + ", Window size=" + str(time_window_length) + ")")
        # Initial percolation calculations.
        v_0 = {x: 0 for x in params["V"].keys()}
        results = indp(interdependent_net, v_0, 1, layers, controlled_layers=controlled_layers,
                       functionality=functionality, co_location=co_location)
        indp_results = results[1]
        indp_results.add_components(0, INDPComponents.calculate_components(results[0], interdependent_net,
                                                                           layers=controlled_layers))
        for n in range(num_time_windows):
            print("-Time window (td-INDP)", n + 1, "/", num_time_windows)
            functionality_t = {}
            # Slide functionality matrix according to sliding time window.
            if functionality:
                for t in functionality:
                    if t in range(n, time_window_length + n + 1):
                        functionality_t[t - n] = functionality[t]
                if len(functionality_t) < time_window_length + 1:
                    diff = time_window_length + 1 - len(functionality_t)
                    max_t = max(functionality_t.keys())
                    for d in range(diff):
                        functionality_t[max_t + d + 1] = functionality_t[max_t]
            # Run td-INDP.
            results = indp(interdependent_net, params["V"], time_window_length + 1, layers,
                           controlled_layers=controlled_layers,
                           functionality=functionality_t, forced_actions=forced_actions,
                           co_location=co_location)
            if save_model:
                save_indp_model_to_file(results[0], output_dir + "/Model", n + 1)
            if "WINDOW_LENGTH" in params:
                indp_results.extend(results[1], t_offset=n + 1, t_start=1, t_end=2)
                # Modify network for recovery actions and calculate components.
                apply_recovery(interdependent_net, results[1], 1)
                indp_results.add_components(n + 1,
                                            INDPComponents.calculate_components(results[0],
                                                                                interdependent_net, 1,
                                                                                layers=controlled_layers))
            else:
                indp_results.extend(results[1], t_offset=0)
                for t in range(1, T):
                    # Modify network to account for recovery actions.
                    apply_recovery(interdependent_net, indp_results, t)
                    indp_results.add_components(1,
                                                INDPComponents.calculate_components(results[0], interdependent_net, t,
                                                                                    layers=controlled_layers))
    # Save results of current simulation.
    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        indp_results.to_csv(output_dir, params["SIM_NUMBER"], suffix=suffix)
        if not os.path.exists(output_dir + '/agents'):
            os.makedirs(output_dir + '/agents')
        indp_results.to_csv_layer(output_dir + '/agents', params["SIM_NUMBER"], suffix=suffix)
    return indp_results


def save_indp_model_to_file(model, out_model_dir, t, l=0, suffix=''):
    """
    This function saves gurobi optimization model to file.

    Parameters
    ----------
    model : gurobipy.Model
        Gurobi optimization model
    out_model_dir : str
        Directory to which the models should be written
    t : int
        The time step corresponding to the model
    l : int
        The layer number corresponding to the model. The default is 0, which means the model include all layers in the
        analysis
    suffix : str
        The suffix that should be added to files when saved. The default is ''.
    Returns
    -------
    None.

    """
    if not os.path.exists(out_model_dir):
        os.makedirs(out_model_dir)
    # Write models to file
    l_name = "/Model_t%d_l%d_%s.lp" % (t, l, suffix)
    model.write(out_model_dir + l_name)
    model.update()
    # Write solution to file
    s_name = "/Solution_t%d_l%d_%s.txt" % (t, l, suffix)
    file_id = open(out_model_dir + s_name, 'w')
    for vv in model.getVars():
        file_id.write('%s %g\n' % (vv.varName, vv.x))
    file_id.write('Obj: %g' % model.objVal)
    file_id.close()


def initialize_sample_network(layers=None):
    """
    This function generate the sample toy example with either 2 or 3 layers.

    Parameters
    ----------
    layers : list
        List of layers in the toy example that can be [1, 2] or [1, 2, 3]. The default is None, which sets layers
        to [1, 2].

    Returns
    -------
    interdependent_net : InfrastructureNetwork
    A :class:`~Infrastructure.InfrastructureNetwork` object containing the interdependent network and all the
        attributes of its nodes and arcs.
    """
    if layers is None:
        layers = [1, 2]

    interdependent_net = InfrastructureNetwork("sample_network")
    node_to_demand_dict = {(1, 1): 5, (2, 1): -1, (3, 1): -2, (4, 1): -2, (5, 1): -4, (6, 1): 4,
                           (7, 2): -2, (8, 2): 6, (9, 2): 1, (10, 2): -5, (11, 2): 4, (12, 2): -4}
    space_to_nodes_dict = {1: [(1, 1), (7, 2)], 2: [(2, 1), (8, 2)],
                           3: [(3, 1), (5, 1), (9, 2), (11, 2)], 4: [(4, 1), (6, 1), (10, 2), (12, 2)]}
    arc_list = [((1, 1), (2, 1)), ((1, 1), (4, 1)), ((1, 1), (3, 1)), ((6, 1), (4, 1)), ((6, 1), (5, 1)),
                ((8, 2), (7, 2)), ((8, 2), (10, 2)), ((9, 2), (7, 2)), ((9, 2), (10, 2)), ((9, 2), (12, 2)),
                ((11, 2), (12, 2))]
    interdep_list = [((1, 1), (7, 2)), ((2, 1), (8, 2)), ((9, 2), (3, 1)), ((4, 1), (10, 2))]
    failed_nodes = [(1, 1), (2, 1), (3, 1), (5, 1), (6, 1),
                    (7, 2), (8, 2), (9, 2), (11, 2), (12, 2)]
    if 3 in layers:
        node_to_demand_dict.update({(13, 3): 3, (14, 3): 6, (15, 3): -5, (16, 3): -6,
                                    (17, 3): 4, (18, 3): -2})
        space_to_nodes_dict[1].extend([(13, 3), (14, 3), (15, 3)])
        space_to_nodes_dict[2].extend([(16, 3), (17, 3), (18, 3)])
        arc_list.extend([((13, 3), (15, 3)), ((14, 3), (15, 3)), ((14, 3), (16, 3)),
                         ((17, 3), (15, 3)), ((17, 3), (16, 3)), ((17, 3), (18, 3))])
        interdep_list.extend([((11, 2), (17, 3)), ((9, 2), (15, 3)), ((14, 3), (8, 2)), ((14, 3), (9, 2))])
        failed_nodes.extend([(14, 3), (15, 3), (16, 3), (17, 3), (18, 3)])
    global_index = 1
    for n in node_to_demand_dict:
        nn = InfrastructureNode(global_index, n[1], n[0])
        nn.demand = node_to_demand_dict[n]
        nn.reconstruction_cost = abs(nn.demand)
        nn.oversupply_penalty = 50
        nn.undersupply_penalty = 50
        nn.resource_usage['p_'] = 1
        if n in failed_nodes:
            nn.functionality = 0.0
            nn.repaired = 0.0
        interdependent_net.G.add_node((nn.local_id, nn.net_id), data={'inf_data': nn})
        global_index += 1
    for s in space_to_nodes_dict:
        interdependent_net.S.append(InfrastructureSpace(s, 0))
        for n in space_to_nodes_dict[s]:
            interdependent_net.G.nodes[n]['data']['inf_data'].space = s
    for a in arc_list:
        aa = InfrastructureArc(a[0][0], a[1][0], a[0][1])
        aa.flow_cost = 1
        aa.capacity = 50
        interdependent_net.G.add_edge((aa.source, aa.layer), (aa.dest, aa.layer), data={'inf_data': aa})
    for g in interdep_list:
        aa = InfrastructureInterdepArc(g[0][0], g[1][0], g[0][1], g[1][1], 1.0)
        interdependent_net.G.add_edge((aa.source, aa.source_layer), (aa.dest, aa.dest_layer), data={'inf_data': aa})
    return interdependent_net


def plot_indp_sample(params, folder_suffix="", suffix=""):
    """
    This function plots the toy example in all time steps of the restoration, and saves plots to file.

    Parameters
    ----------
    params : dict
        Parameters that are needed to run the indp optimization.
    folder_suffix : str
        The suffix that should be added to the target folder. The default is ''.
    suffix : str
        The suffix that should be added to plots when saved. The default is ''.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(16, 8))
    if 3 in params["L"]:
        plt.figure(figsize=(16, 10))
    interdep_net = initialize_sample_network(layers=params["L"])
    pos = nx.spring_layout(interdep_net.G)
    pos[(1, 1)][0] = 0.5
    pos[(7, 2)][0] = 0.5
    pos[(2, 1)][0] = 0.0
    pos[(8, 2)][0] = 0.0
    pos[(3, 1)][0] = 2.0
    pos[(9, 2)][0] = 2.0
    pos[(4, 1)][0] = 1.5
    pos[(10, 2)][0] = 1.5
    pos[(5, 1)][0] = 3.0
    pos[(11, 2)][0] = 3.0
    pos[(6, 1)][0] = 2.5
    pos[(12, 2)][0] = 2.5
    pos[(2, 1)][1] = 2.0
    pos[(4, 1)][1] = 2.0
    pos[(6, 1)][1] = 2.0
    pos[(1, 1)][1] = 3.0
    pos[(3, 1)][1] = 3.0
    pos[(5, 1)][1] = 3.0
    pos[(8, 2)][1] = 0.0
    pos[(10, 2)][1] = 0.0
    pos[(12, 2)][1] = 0.0
    pos[(7, 2)][1] = 1.0
    pos[(9, 2)][1] = 1.0
    pos[(11, 2)][1] = 1.0
    node_dict = {1: [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
                 11: [(4, 1)],  # Undamaged
                 12: [(1, 1), (2, 1), (3, 1), (5, 1), (6, 1)],  # Damaged
                 2: [(7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (12, 2)],
                 21: [(10, 2)],
                 22: [(7, 2), (8, 2), (9, 2), (11, 2), (12, 2)]}
    arc_dict = {1: [((1, 1), (2, 1)), ((1, 1), (3, 1)), ((1, 1), (4, 1)), ((6, 1), (4, 1)),
                    ((6, 1), (5, 1))],
                2: [((8, 2), (7, 2)), ((8, 2), (10, 2)), ((9, 2), (7, 2)), ((9, 2), (10, 2)),
                    ((9, 2), (12, 2)), ((11, 2), (12, 2))]}
    if 3 in params["L"]:
        pos[(13, 3)][0] = 0.5
        pos[(14, 3)][0] = 0.0
        pos[(15, 3)][0] = 2.0
        pos[(16, 3)][0] = 1.5
        pos[(17, 3)][0] = 3.0
        pos[(18, 3)][0] = 2.5
        pos[(13, 3)][1] = -2.0
        pos[(14, 3)][1] = -1.0
        pos[(15, 3)][1] = -2.0
        pos[(16, 3)][1] = -1.0
        pos[(17, 3)][1] = -2.0
        pos[(18, 3)][1] = -1.0
        node_dict[3] = [(13, 3), (14, 3), (15, 3), (16, 3), (17, 3), (18, 3)]
        node_dict[31] = [(13, 3)]
        node_dict[32] = [(14, 3), (15, 3), (16, 3), (17, 3), (18, 3)]
        arc_dict[3] = [((13, 3), (15, 3)), ((14, 3), (15, 3)), ((14, 3), (16, 3)),
                       ((17, 3), (15, 3)), ((17, 3), (16, 3)), ((17, 3), (18, 3))]

    labels = {}
    for n, d in interdep_net.G.nodes(data=True):
        labels[n] = "%d[%d]" % (n[0], d['data']['inf_data'].demand)
        labels[n] = "%d" % (n[0])
    pos_moved = {}
    for key, value in pos.items():
        pos_moved[key] = [0, 0]
        pos_moved[key][0] = pos[key][0] - 0.17
        pos_moved[key][1] = pos[key][1] + 0.17

    v_r = params["V"]
    if isinstance(v_r, (int)):
        total_resource = v_r
    else:
        total_resource = sum([val for _, val in v_r.items()])

    output_dir = params["OUTPUT_DIR"] + '_L' + str(len(params["L"])) + '_m' + str(params["MAGNITUDE"]) + "_v" + str(
        total_resource) + folder_suffix
    action_file = output_dir + "/actions_" + str(params["SIM_NUMBER"]) + "_" + suffix + ".csv"
    actions = {0: []}
    if os.path.isfile(action_file):
        with open(action_file) as f:
            lines = f.readlines()[1:]
            for line in lines:
                data = line.split(",")
                t = int(data[0])
                action = str.strip(data[1])
                if t not in actions:
                    actions[t] = []
                actions[t].append(action)

    T = max(actions.keys())
    for t, value in actions.items():
        plt.subplot(2, int((T + 1) / 2) + 1, t + 1, aspect='equal')
        plt.title('Time = %d' % t)
        for a in value:
            data = a.split(".")
            node_dict[int(data[1]) * 10 + 1].append((int(data[0]), int(data[1])))
            node_dict[int(data[1]) * 10 + 2].remove((int(data[0]), int(data[1])))
        nx.draw(interdep_net.G, pos, node_color='w', arrowsize=25, arrowstyle='-|>')
        nx.draw_networkx_labels(interdep_net.G, labels=labels, pos=pos, font_weight='bold',
                                font_color='w', font_family='CMU Serif', font_size=18)
        nx.draw_networkx_nodes(interdep_net.G, pos, nodelist=node_dict[1], node_color='#b51717', node_size=1100,
                               alpha=0.7)
        nx.draw_networkx_nodes(interdep_net.G, pos, nodelist=node_dict[2], node_color='#005f98', node_size=1100,
                               alpha=0.7)
        nx.draw_networkx_nodes(interdep_net.G, pos_moved, nodelist=node_dict[12], node_color='k', node_shape="X",
                               node_size=150)
        nx.draw_networkx_nodes(interdep_net.G, pos_moved, nodelist=node_dict[22], node_color='k', node_shape="X",
                               node_size=150)
        nx.draw_networkx_edges(interdep_net.G, pos, edgelist=arc_dict[1], width=1.5, alpha=0.9, edge_color='r',
                               arrowsize=25, arrowstyle='-|>')
        nx.draw_networkx_edges(interdep_net.G, pos, edgelist=arc_dict[2], width=1.5, alpha=0.9, edge_color='b',
                               arrowsize=25, arrowstyle='-|>')
        if 3 in params["L"]:
            nx.draw_networkx_nodes(interdep_net.G, pos, nodelist=node_dict[3], node_color='#009181', node_size=1100,
                                   alpha=0.7)
            nx.draw_networkx_nodes(interdep_net.G, pos_moved, nodelist=node_dict[32], node_color='k', node_shape="X",
                                   node_size=150)
            nx.draw_networkx_edges(interdep_net.G, pos, edgelist=arc_dict[3], width=1.5, alpha=0.9, edge_color='g',
                                   arrowsize=25, arrowstyle='-|>')
    plt.tight_layout()
    plt.savefig(output_dir + '/plot_net' + suffix + '.png', dpi=300)


def get_resource_suffix(params):
    """
    This function generates the part of suffix of result folders that pertains to resource cap(s).

    Parameters
    ----------
    params : dict
        Parameters that are needed to run the indp optimization.

    Returns
    -------
     out_dir_suffix_res : str
     The part of suffix of result folders that pertains to resource cap(s).

    """
    out_dir_suffix_res = ''
    for rc, val in params["V"].items():
        if isinstance(val, int):
            if rc != '':
                out_dir_suffix_res += rc[0] + str(val)
            else:
                out_dir_suffix_res += str(val)
        else:
            out_dir_suffix_res += rc[0] + str(sum([lval for _, lval in val.items()])) + '_fixed_layer_Cap'
    return out_dir_suffix_res


def time_resource_usage_curves(base_dir, damage_dir, sample_num):
    """
    This module calculates the repair time for nodes and arcs for the current scenario based on their damage state, and
    writes them to the input files of INDP. Currently, it is only compatible with NIST testbeds.

    .. todo::
        The calculated repair time and costs are written to node and arc info input
        files. It makes it impossible to run the analyses in parallel because there
        might be a conflict between two processes. Consider correcting this.

    Parameters
    ----------
    base_dir : dir
        The address of the folder where the basic network information (topology, parameters, etc.) are stored.
    damage_dir : dir
        the address of the folder where the damage information are stored.
    sample_num : int
        The sample number of the damage scenarios for which the repair data are calculated.

    Returns
    -------
    None.

    """
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    nodes_reptime_func = pd.read_csv(base_dir + 'repair_time_curves_nodes.csv')
    nodes_damge_ratio = pd.read_csv(base_dir + 'damage_ratio_nodes.csv')
    arcs_reptime_func = pd.read_csv(base_dir + 'repair_time_curves_arcs.csv')
    arcs_damge_ratio = pd.read_csv(base_dir + 'damage_ratio_arcs.csv')
    dmg_sce_data = pd.read_csv(damage_dir + 'Initial_node_ds.csv', delimiter=',', header=None)
    net_names = {'Water': 1, 'Power': 3}

    for file in files:
        fname = file[0:-4]
        if fname[-5:] == 'Nodes':
            with open(base_dir + file) as f:
                node_data = pd.read_csv(f, delimiter=',')
                for v in node_data.iterrows():
                    try:
                        node_type = v[1]['Node Type']
                        node_id = v[1]['ID']
                    except KeyError:
                        node_type = v[1]['utilfcltyc']
                        node_id = v[1]['nodenwid']

                    reptime_func_node = nodes_reptime_func[nodes_reptime_func['Type'] == node_type]
                    dr_data = nodes_damge_ratio[nodes_damge_ratio['Type'] == node_type]
                    rep_time = 0
                    repair_cost = 0
                    if not reptime_func_node.empty:
                        node_name = '(' + str(node_id) + ',' + str(net_names[fname[:5]]) + ')'
                        ds = dmg_sce_data[dmg_sce_data[0] == node_name].iloc[0][sample_num + 1]
                        rep_time = reptime_func_node.iloc[0]['ds_' + ds + '_mean']
                        # ..todo Add repair time uncertainty here
                        # rep_time = np.random.normal(reptime_func_node['ds_'+ds+'_mean'],
                        #                             reptime_func_node['ds_'+ds+'_sd'], 1)[0]

                        dr = dr_data.iloc[0]['dr_' + ds + '_be']
                        # ..todo Add damage ratio uncertainty here
                        # dr = np.random.uniform(dr_data.iloc[0]['dr_'+ds+'_min'],
                        #                       dr_data.iloc[0]['dr_'+ds+'_max'], 1)[0]
                        repair_cost = v[1]['q (complete DS)'] * dr
                    node_data.loc[v[0], 'p_time'] = rep_time if rep_time > 0 else 0
                    node_data.loc[v[0], 'p_budget'] = repair_cost
                    node_data.loc[v[0], 'q'] = repair_cost
                node_data.to_csv(base_dir + file, sep=',', index=False)

    for file in files:
        fname = file[0:-4]
        if fname[-4:] == 'Arcs':
            with open(base_dir + file) as f:
                data = pd.read_csv(f, delimiter=',')
                dmg_data_all = pd.read_csv(damage_dir + 'pipe_dmg.csv', delimiter=',')
                for v in data.iterrows():
                    dmg_data_arc = dmg_data_all[dmg_data_all['guid'] == v[1]['guid']]
                    rep_time = 0
                    repair_cost = 0
                    if not dmg_data_arc.empty:
                        if v[1]['diameter'] > 20:
                            reptime_func_arc = arcs_reptime_func[arcs_reptime_func['Type'] == '>20 in']
                            dr_data = arcs_damge_ratio[arcs_damge_ratio['Type'] == '>20 in']
                        else:
                            reptime_func_arc = arcs_reptime_func[arcs_reptime_func['Type'] == '<20 in']
                            dr_data = arcs_damge_ratio[arcs_damge_ratio['Type'] == '<20 in']
                        try:
                            pipe_length = v[1]['Length (km)']
                            pipe_length_ft = v[1]['Length (ft)']
                        except KeyError:
                            pipe_length = v[1]['length_km']
                            pipe_length_ft = v[1]['Length']
                        rep_rate = {'break': dmg_data_arc.iloc[0]['breakrate'],
                                    'leak': dmg_data_arc.iloc[0]['leakrate']}
                        rep_time = (rep_rate['break'] * reptime_func_arc['# Fixed Breaks/Day/Worker'] + \
                                    rep_rate['leak'] * reptime_func_arc['# Fixed Leaks/Day/Worker']) * \
                                   pipe_length / 4  # assuming a 4-person crew per HAZUS
                        dr = {'break': dr_data.iloc[0]['break_be'], 'leak': dr_data.iloc[0]['leak_be']}
                        # ..todo Add repair cost uncertainty here
                        # dr = {'break': np.random.uniform(dr_data.iloc[0]['break_min'],
                        #                                  dr_data.iloc[0]['break_max'], 1)[0],
                        #       'leak': np.random.uniform(dr_data.iloc[0]['leak_min'],
                        #                                  dr_data.iloc[0]['leak_max'], 1)[0]}

                        num_20_ft_seg = pipe_length_ft / 20
                        num_breaks = rep_rate['break'] * pipe_length
                        if num_breaks > num_20_ft_seg:
                            repair_cost += v[1]['f (complete)'] * dr['break']
                        else:
                            repair_cost += v[1]['f (complete)'] / num_20_ft_seg * num_breaks * dr['break']
                        num_leaks = rep_rate['leak'] * pipe_length
                        if num_leaks > num_20_ft_seg:
                            repair_cost += v[1]['f (complete)'] * dr['leak']
                        else:
                            repair_cost += v[1]['f (complete)'] / num_20_ft_seg * num_leaks * dr['leak']
                        repair_cost = min(repair_cost, v[1]['f (complete)'])
                    data.loc[v[0], 'h_time'] = float(rep_time)
                    data.loc[v[0], 'h_budget'] = repair_cost
                    data.loc[v[0], 'f'] = repair_cost
                data.to_csv(base_dir + file, sep=',', index=False)
