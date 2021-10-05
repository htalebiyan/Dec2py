import sys

from infrastructure import *
import indp
from indputils import *
from gurobipy import GRB, Model, LinExpr
import string
import networkx as nx
import matplotlib.pyplot as plt
import copy
import time
import dislocationutils


def inmrp(N, v_r, T=1, layers=None, controlled_layers=None, functionality=None, forced_actions=False,
          fixed_nodes=None, print_cmd=True, time_limit=None, co_location=True, solution_pool=None):
    """
    INMRP optimization problem.

    Parameters
    ----------
    N : :class:`~infrastructure.InfrastructureNetwork`
        An InfrastructureNetwork instance.
    v_r : dict
        Dictionary of the number of restoration resources of different types in the analysis.
        If the value is a scale for a type, it shows the total number of resources of that type for all layers .
        If the value is a list for a type, it shows the total number of resources of that type given to each layer.
    v_r_hat : dict
        Dictionary of the number of protection resources of different types in the analysis.
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
        sys.exit('No list of layers')
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
    a_prime = [(u, v, a) for u, v, a in g_prime.edges(data=True) if a['data']['inf_data'].repaired == 0.0]
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

    for t in range(T + 1):
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
            m.addVar(name='w_tilde_' + str(n) + "," + str(t), vtype=GRB.BINARY)
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
            m.addVar(name='y_tilde_' + str(u) + "," + str(v) + "," + str(t), vtype=GRB.BINARY)
    m.update()

    # Populate objective function.
    obj_func = LinExpr()
    for t in range(T + 1):
        if co_location:
            for s in S:
                obj_func += s.cost * m.getVarByName('z_' + str(s.id) + "," + str(t))
        for u, v, a in a_hat_prime:
            obj_func += (float(a['data']['inf_data'].reconstruction_cost[t]) / 2.0) * \
                        m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t))
        for n, d in n_hat_prime:
            obj_func += d['data']['inf_data'].reconstruction_cost[t] * m.getVarByName(
                'w_tilde_' + str(n) + "," + str(t))
        for n, d in n_hat.nodes(data=True):
            obj_func += d['data']['inf_data'].oversupply_penalty[t] * m.getVarByName('delta+_' + str(n) + "," + str(t))
            obj_func += d['data']['inf_data'].undersupply_penalty[t] * m.getVarByName('delta-_' + str(n) + "," + str(t))
            for l, val in d['data']['inf_data'].extra_com.items():
                obj_func += val['oversupply_penalty'][t] * m.getVarByName(
                    'delta+_' + str(n) + "," + str(t) + "," + str(l))
                obj_func += val['undersupply_penalty'][t] * m.getVarByName(
                    'delta-_' + str(n) + "," + str(t) + "," + str(l))
        for u, v, a in n_hat.edges(data=True):
            if not a['data']['inf_data'].is_interdep:
                obj_func += a['data']['inf_data'].flow_cost[t] * m.getVarByName(
                    'x_' + str(u) + "," + str(v) + "," + str(t))
                for l, val in a['data']['inf_data'].extra_com.items():
                    obj_func += val['flow_cost'][t] * m.getVarByName(
                        'x_' + str(u) + "," + str(v) + "," + str(t) + "," + str(l))
    m.setObjective(obj_func, GRB.MINIMIZE)
    m.update()

    # Constraints.
    for t in range(T + 1):
        # Time-dependent constraint.
        for n, d in n_hat_prime:
            w_tilde_sum = LinExpr()
            for t_prime in range(0, t + 1):
                w_tilde_sum += m.getVarByName('w_tilde_' + str(n) + "," + str(t_prime))
            m.addConstr(m.getVarByName('w_' + str(n) + "," + str(t)), GRB.LESS_EQUAL, w_tilde_sum,
                        "Time dependent recovery constraint at node " + str(n) + "," + str(t))
        for u, v, a in a_hat_prime:
            y_tilde_sum = LinExpr()
            for t_prime in range(0, t + 1):
                y_tilde_sum += m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t_prime))
            m.addConstr(m.getVarByName('y_' + str(u) + "," + str(v) + "," + str(t)), GRB.LESS_EQUAL, y_tilde_sum,
                        "Time dependent recovery constraint at arc " + str(u) + "," + str(v) + "," + str(t))
        # Enforce a_i,j to be fixed if a_j,i is fixed (and vice versa).
        for u, v, a in a_hat_prime:
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
            demand_constr += d['data']['inf_data'].demand[t] - m.getVarByName(
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
                demand_constr += val['demand'][t] - \
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
            if not a['data']['inf_data'].is_interdep:
                lhs = m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t)) + \
                      sum([m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t) + "," + str(l)) \
                           for l in a['data']['inf_data'].extra_com.keys()])
                if (u in [n for (n, d) in n_hat_prime]) | (u in interdep_nodes_list):
                    m.addConstr(lhs, GRB.LESS_EQUAL,
                                a['data']['inf_data'].capacity[t] * m.getVarByName('w_' + str(u) + "," + str(t)),
                                "Flow in functionality constraint(" + str(u) + "," + str(v) + "," + str(t) + ")")
                else:
                    m.addConstr(lhs, GRB.LESS_EQUAL,
                                a['data']['inf_data'].capacity[t] * N.G.nodes[u]['data']['inf_data'].functionality,
                                "Flow in functionality constraint (" + str(u) + "," + str(v) + "," + str(t) + ")")
                if (v in [n for (n, d) in n_hat_prime]) | (v in interdep_nodes_list):
                    m.addConstr(lhs, GRB.LESS_EQUAL,
                                a['data']['inf_data'].capacity[t] * m.getVarByName('w_' + str(v) + "," + str(t)),
                                "Flow out functionality constraint(" + str(u) + "," + str(v) + "," + str(t) + ")")
                else:
                    m.addConstr(lhs, GRB.LESS_EQUAL,
                                a['data']['inf_data'].capacity[t] * N.G.nodes[v]['data']['inf_data'].functionality,
                                "Flow out functionality constraint (" + str(u) + "," + str(v) + "," + str(t) + ")")
                if (u, v, a) in a_hat_prime:
                    m.addConstr(lhs, GRB.LESS_EQUAL,
                                a['data']['inf_data'].capacity[t] * m.getVarByName(
                                    'y_' + str(u) + "," + str(v) + "," + str(t)),
                                "Flow arc functionality constraint (" + str(u) + "," + str(v) + "," + str(t) + ")")
                else:
                    m.addConstr(lhs, GRB.LESS_EQUAL,
                                a['data']['inf_data'].capacity[t] * N.G[u][v]['data']['inf_data'].functionality,
                                "Flow arc functionality constraint(" + str(u) + "," + str(v) + "," + str(t) + ")")

        # Restoration resource availability constraints.
        for rc, val in v_r.items():
            is_sep_res = False
            if isinstance(val[t], int):
                total_resource = val[t]
            else:
                is_sep_res = True
                total_resource = sum([lval for _, lval in val[t].items()])
                assert len(val[t].keys()) == len(
                    layers), "The number of restoration resource values does not match the " \
                             "number of layers. "
            resource_left_constr = LinExpr()
            if is_sep_res:
                res_left_constr_sep = {key: LinExpr() for key in val[t].keys()}

            for u, v, a in a_hat_prime:
                idx_lyr = a['data']['inf_data'].layer
                res_use = 0.5 * a['data']['inf_data'].resource_usage['h_' + rc][t]
                resource_left_constr += res_use * m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t))
                if is_sep_res:
                    res_left_constr_sep[idx_lyr] += res_use * m.getVarByName(
                        'y_tilde_' + str(u) + "," + str(v) + "," + str(t))

            for n, d in n_hat_prime:
                idx_lyr = n[1]
                res_use = d['data']['inf_data'].resource_usage['p_' + rc][t]
                resource_left_constr += res_use * m.getVarByName('w_tilde_' + str(n) + "," + str(t))
                if is_sep_res:
                    res_left_constr_sep[idx_lyr] += res_use * m.getVarByName('w_tilde_' + str(n) + "," + str(t))

            m.addConstr(resource_left_constr, GRB.LESS_EQUAL, total_resource,
                        "Restoration resource availability constraint for " + rc + " at " + str(t) + ".")
            if is_sep_res:
                for k, lval in val.items():
                    m.addConstr(res_left_constr_sep[k], GRB.LESS_EQUAL, lval,
                                "Restoration resource availability constraint for " + rc + " at " + str(t) + \
                                " for layer " + str(k) + ".")

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
                    recovery_sum += m.getVarByName('w_tilde_' + str(n) + "," + str(t))
                for u, v, a in a_hat_prime:
                    recovery_sum += m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t))
                m.addConstr(recovery_sum, GRB.GREATER_EQUAL, 1, "Forced action constraint")

        # Geographic space constraints
        if co_location:
            for s in S:
                for n, d in n_hat_prime:
                    if d['data']['inf_data'].in_space(s.id):
                        m.addConstr(
                            m.getVarByName('w_tilde_' + str(n) + "," + str(t)) * d['data']['inf_data'].in_space(s.id),
                            GRB.LESS_EQUAL, m.getVarByName('z_' + str(s.id) + "," + str(t)),
                            "Geographical space constraint for node " + str(n) + "," + str(t))
                for u, v, a in a_hat_prime:
                    if a['data']['inf_data'].in_space(s.id):
                        m.addConstr(m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t)) * a['data'][
                            'inf_data'].in_space(s.id), GRB.LESS_EQUAL, m.getVarByName('z_' + str(s.id) + "," + str(t)),
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
        # if solution_pool:
        #     sol_pool_results = collect_solution_pool(m, T, n_hat_prime, a_hat_prime)
        #     return [m, results, sol_pool_results]
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
    total_demand = {t: 0 for t in range(T + 1)}
    total_demand_layer = {t: {l: 0.0 for l in layers} for t in range(T + 1)}
    for t in range(T + 1):
        for n, d in n_hat.nodes(data=True):
            demand_value = d['data']['inf_data'].demand[t]
            if demand_value < 0:
                total_demand[t] += demand_value
                total_demand_layer[t][n[1]] += demand_value
    for t in range(T + 1):
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
            if round(m.getVarByName(node_var).x) == 1:
                action = str(n[0]) + "." + str(n[1])
                indp_results.add_action(t, action)
        # Record edge recovery actions.
        for u, v, a in a_hat_prime:
            arc_var = 'y_tilde_' + str(u) + "," + str(v) + "," + str(t)
            if round(m.getVarByName(arc_var).x) == 1:
                action = str(u[0]) + "." + str(u[1]) + "/" + str(v[0]) + "." + str(v[1])
                indp_results.add_action(t, action)
        # Calculate space preparation costs.
        if coloc:
            for s in S:
                space_prep_cost += s.cost * m.getVarByName('z_' + str(s.id) + "," + str(t)).x
        indp_results.add_cost(t, "Space Prep", space_prep_cost, space_prep_cost_layer)
        # Calculate arc restoration costs.
        for u, v, a in a_hat_prime:
            arc_var = 'y_tilde_' + str(u) + "," + str(v) + "," + str(t)
            arc_cost += (a['data']['inf_data'].reconstruction_cost[t] / 2.0) * m.getVarByName(arc_var).x
            arc_cost_layer[u[1]] += (a['data']['inf_data'].reconstruction_cost[t] / 2.0) * m.getVarByName(arc_var).x
        indp_results.add_cost(t, "Arc", arc_cost, arc_cost_layer)
        # Calculate node restoration costs.
        for n, d in n_hat_prime:
            node_var = 'w_tilde_' + str(n) + "," + str(t)
            node_cost += d['data']['inf_data'].reconstruction_cost[t] * m.getVarByName(node_var).x
            node_cost_layer[n[1]] += d['data']['inf_data'].reconstruction_cost[t] * m.getVarByName(node_var).x
        indp_results.add_cost(t, "Node", node_cost, node_cost_layer)
        # Calculate under/oversupply costs.
        for n, d in n_hat.nodes(data=True):
            over_supp_cost += d['data']['inf_data'].oversupply_penalty[t] * m.getVarByName(
                'delta+_' + str(n) + "," + str(t)).x
            over_supp_cost_layer[n[1]] += d['data']['inf_data'].oversupply_penalty[t] * m.getVarByName(
                'delta+_' + str(n) + "," + str(t)).x
            under_supp += m.getVarByName('delta-_' + str(n) + "," + str(t)).x
            under_supp_layer[n[1]] += m.getVarByName('delta-_' + str(n) + "," + str(t)).x / total_demand_layer[t][n[1]]
            under_supp_cost += d['data']['inf_data'].undersupply_penalty[t] * m.getVarByName(
                'delta-_' + str(n) + "," + str(t)).x
            under_supp_cost_layer[n[1]] += d['data']['inf_data'].undersupply_penalty[t] * m.getVarByName(
                'delta-_' + str(n) + "," + str(t)).x
        indp_results.add_cost(t, "Over Supply", over_supp_cost, over_supp_cost_layer)
        indp_results.add_cost(t, "Under Supply", under_supp_cost, under_supp_cost_layer)
        indp_results.add_cost(t, "Under Supply Perc", under_supp / total_demand[t], under_supp_layer)
        # Calculate flow costs.
        for u, v, a in n_hat.edges(data=True):
            if not a['data']['inf_data'].is_interdep:
                flow_cost += a['data']['inf_data'].flow_cost[t] * m.getVarByName(
                    'x_' + str(u) + "," + str(v) + "," + str(t)).x
                flow_cost_layer[u[1]] += a['data']['inf_data'].flow_cost[t] * m.getVarByName(
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


def run_inmrp(params, layers=None, controlled_layers=None, functionality=None, T=1, save=True, suffix="",
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
        interdependent_net = indp.initialize_network(base_dir="../data/INDP_7-20-2015/",
                                                     sim_number=params['SIM_NUMBER'], magnitude=params["MAGNITUDE"])
    else:
        interdependent_net = params["N"]
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"] = 1

    out_dir_suffix_res = get_resource_suffix(params)
    indp_results = INDPResults(params["L"])
    num_time_windows = 1
    time_window_length = T
    is_first_iteration = True
    if "WINDOW_LENGTH" in params:
        time_window_length = params["WINDOW_LENGTH"]
        num_time_windows = T
    output_dir = params["OUTPUT_DIR"] + '_L' + str(len(layers)) + "_m" + str(
        params["MAGNITUDE"]) + "_v" + out_dir_suffix_res

    print("Running INMRP (T=" + str(T) + ", Window size=" + str(time_window_length) + ")")
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
        results = inmrp(interdependent_net, params["V"], time_window_length, layers,
                        controlled_layers=controlled_layers, functionality=functionality_t,
                        forced_actions=forced_actions, co_location=co_location)
        if save_model:
            indp.save_indp_model_to_file(results[0], output_dir + "/Model", n + 1)
        if "WINDOW_LENGTH" in params:
            indp_results.extend(results[1], t_offset=n, t_start=0, t_end=time_window_length+1)
            # Modify network for recovery actions and calculate components.
            indp.apply_recovery(interdependent_net, results[1], 1)
            indp_results.add_components(n + 1, INDPComponents.calculate_components(results[0], interdependent_net, 1,
                                                                                   layers=controlled_layers))
        else:
            indp_results.extend(results[1], t_offset=0)
            for t in range(1, T):
                # Modify network to account for recovery actions.
                indp.apply_recovery(interdependent_net, indp_results, t)
                indp_results.add_components(1, INDPComponents.calculate_components(results[0], interdependent_net, t,
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
        if isinstance(val[0], int):
            sum_res = sum([valt for _, valt in val.items()])
            if rc != '':
                out_dir_suffix_res += rc[0] + str(sum_res)
            else:
                out_dir_suffix_res += str(sum_res)
        else:
            sum_res = sum([sum([lval for _, lval in valt.items()]) for _, valt in val.items()])
            out_dir_suffix_res += rc[0] + str(sum_res) + '_fixed_layer_Cap'
    return out_dir_suffix_res

def initialize_sample_network(layers=None, T=1):
    """
    This function generate the sample toy example with either 2 or 3 layers.

    Parameters
    ----------
    layers : list
        List of layers in the toy example that can be [1, 2] or [1, 2, 3]. The default is None, which sets layers
        to [1, 2].
    T : integer
        Number of time steps of analysis

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
        nn.demand = {t: node_to_demand_dict[n] for t in range(T+1)}
        nn.reconstruction_cost = {t: abs(nn.demand[t]) for t in range(T+1)}
        nn.reconstruction_cost[0] = nn.reconstruction_cost[0]+1
        nn.oversupply_penalty = {t: 50 for t in range(T+1)}
        nn.undersupply_penalty = {t: 50 for t in range(T+1)}
        nn.resource_usage['p_'] = {t: 1 for t in range(T+1)}
        nn.resource_usage['p_'][0] = 2
        if n in failed_nodes:
            nn.repaired = 0.0
        interdependent_net.G.add_node((nn.local_id, nn.net_id), data={'inf_data': nn})
        global_index += 1
    for s in space_to_nodes_dict:
        interdependent_net.S.append(InfrastructureSpace(s, 0))
        for n in space_to_nodes_dict[s]:
            interdependent_net.G.nodes[n]['data']['inf_data'].space = s
    for a in arc_list:
        aa = InfrastructureArc(a[0][0], a[1][0], a[0][1])
        aa.flow_cost = {t: 1 for t in range(T+1)}
        aa.capacity = {t: 50 for t in range(T+1)}
        interdependent_net.G.add_edge((aa.source, aa.layer), (aa.dest, aa.layer), data={'inf_data': aa})
    for g in interdep_list:
        aa = InfrastructureInterdepArc(g[0][0], g[1][0], g[0][1], g[1][1], 1.0)
        interdependent_net.G.add_edge((aa.source, aa.source_layer), (aa.dest, aa.dest_layer), data={'inf_data': aa})
    return interdependent_net