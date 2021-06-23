from infrastructure import *
from indputils import *
from gurobipy import GRB, Model, LinExpr
import string
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random
import time
import math
import pickle
import sys


# HOME_DIR="/Users/Andrew/"
# if platform.system() == "Linux":
#    HOME_DIR="/home/andrew/"

def indp(N, v_r, T=1, layers=[1, 3], controlled_layers=[1, 3], functionality={},
         forced_actions=False, fixed_nodes={}, print_cmd=True, time_limit=None,
         co_location=True, solution_pool=None):
    """
    INDP optimization problem. It also solves td-INDP if T > 1.

    Parameters
    ----------
    N : :class:`~infrastructure.InfrastructureNetwork`
        An InfrastructureNetwork instance.
    v_r : list
        Vector of the number of resources given to each layer in each timestep.
        If the size of the vector is 1, it shows the total number of resources for all layers.
    T : int, optional
        Number of time steps to optimize over. The default is 1.
    layers : list, optional
        Layer IDs of N included in the optimization. The default is [1,3]
        (1 for water and 3 for power in the Shelby County database).
    controlled_layers : list, optional
        Layer IDs that can be recovered in this optimization. Used for decentralized
        optimization. The default is [1,3].
    functionality : dict, optional
        Dictionary of nodes to functionality values for non-controlled nodes.
        Used for decentralized optimization. The default is {}.
    forced_actions : bool, optional
        If true, it forces the optimizer to repair at least one element. The default is False.
    fixed_nodes : dict, optional
        It fixes the functionality of given elements to a given value. The default is {}.
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
        A list of the form ``[m, results]`` for a successful optimization where m is the
        Gurobi optimization model and results is a :class:`~indputils.INDPResults` object
        generated using :func:`collect_results`.
        If :envvar:`solution_pool` is set to a number, the function returns ``[m, results, sol_pool_results]``
        where `sol_pool_results` is dictionary of solution that should be retrieved from the
        optimizer in addition to the optimal one collected using :func:`collect_solution_pool`.

    """
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

    # print "N'=",[n for (n,d) in N_prime]
    for t in range(T):
        # Add geographical space variables.
        if co_location:
            for s in S:
                m.addVar(name='z_' + str(s.id) + "," + str(t), vtype=GRB.BINARY)
        # Add over/under-supply variables for each node.
        for n, d in n_hat.nodes(data=True):
            m.addVar(name='delta+_' + str(n) + "," + str(t), lb=0.0)
            m.addVar(name='delta-_' + str(n) + "," + str(t), lb=0.0)
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
        # Add flow variables for each arc.
        for u, v, a in n_hat.edges(data=True):
            m.addVar(name='x_' + str(u) + "," + str(v) + "," + str(t), lb=0.0)
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
        for u, v, a in n_hat.edges(data=True):
            obj_func += a['data']['inf_data'].flow_cost * m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t))

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
                wTildeSum = LinExpr()
                for t_prime in range(1, t + 1):
                    wTildeSum += m.getVarByName('w_tilde_' + str(n) + "," + str(t_prime))
                m.addConstr(m.getVarByName('w_' + str(n) + "," + str(t)), GRB.LESS_EQUAL, wTildeSum,
                            "Time dependent recovery constraint at node " + str(n) + "," + str(t))
        for u, v, a in a_hat_prime:
            if t > 0:
                yTildeSum = LinExpr()
                for t_prime in range(1, t + 1):
                    yTildeSum += m.getVarByName('y_tilde_' + str(u) + "," + str(v) + "," + str(t_prime))
                m.addConstr(m.getVarByName('y_' + str(u) + "," + str(v) + "," + str(t)), GRB.LESS_EQUAL, yTildeSum,
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
            outFlowConstr = LinExpr()
            inFlowConstr = LinExpr()
            demandConstr = LinExpr()
            for u, v, a in n_hat.out_edges(n, data=True):
                outFlowConstr += m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t))
            for u, v, a in n_hat.in_edges(n, data=True):
                inFlowConstr += m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t))
            demandConstr += d['data']['inf_data'].demand - m.getVarByName(
                'delta+_' + str(n) + "," + str(t)) + m.getVarByName('delta-_' + str(n) + "," + str(t))
            m.addConstr(outFlowConstr - inFlowConstr, GRB.EQUAL, demandConstr,
                        "Flow conservation constraint " + str(n) + "," + str(t))
        # Flow functionality constraints.
        if not functionality:
            interdep_nodes_list = interdep_nodes.keys()  # Interdepndent nodes with a damaged dependee node
        else:
            interdep_nodes_list = interdep_nodes[t].keys()  # Interdepndent nodes with a damaged dependee node
        for u, v, a in n_hat.edges(data=True):
            if (u in [n for (n, d) in n_hat_prime]) | (u in interdep_nodes_list):
                m.addConstr(m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t)), GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * m.getVarByName('w_' + str(u) + "," + str(t)),
                            "Flow in functionality constraint(" + str(u) + "," + str(v) + "," + str(t) + ")")
            else:
                m.addConstr(m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t)), GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * N.G.nodes[u]['data']['inf_data'].functionality,
                            "Flow in functionality constraint (" + str(u) + "," + str(v) + "," + str(t) + ")")
            if (v in [n for (n, d) in n_hat_prime]) | (v in interdep_nodes_list):
                m.addConstr(m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t)), GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * m.getVarByName('w_' + str(v) + "," + str(t)),
                            "Flow out functionality constraint(" + str(u) + "," + str(v) + "," + str(t) + ")")
            else:
                m.addConstr(m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t)), GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * N.G.nodes[v]['data']['inf_data'].functionality,
                            "Flow out functionality constraint (" + str(u) + "," + str(v) + "," + str(t) + ")")
            if (u, v, a) in a_hat_prime:
                m.addConstr(m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t)), GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * m.getVarByName(
                                'y_' + str(u) + "," + str(v) + "," + str(t)),
                            "Flow arc functionality constraint (" + str(u) + "," + str(v) + "," + str(t) + ")")
            else:
                m.addConstr(m.getVarByName('x_' + str(u) + "," + str(v) + "," + str(t)), GRB.LESS_EQUAL,
                            a['data']['inf_data'].capacity * N.G[u][v]['data']['inf_data'].functionality,
                            "Flow arc functionality constraint(" + str(u) + "," + str(v) + "," + str(t) + ")")

        # Resource availability constraints.
        is_sep_resource = False
        if isinstance(v_r, int):
            total_resource = v_r
        else:
            is_sep_resource = True
            total_resource = sum([val for _, val in v_r.items()])
            if len(v_r.keys()) != len(layers):
                sys.exit("The number of resource cap values does not match the number of layers.\n")

        resource_left_constr = LinExpr()
        if is_sep_resource:
            resource_left_constr_sep = {key: LinExpr() for key, _ in v_r.items()}

        for u, v, a in a_hat_prime:
            index_layer = a['data']['inf_data'].layer
            if T == 1:
                resource_left_constr += 0.5 * a['data']['inf_data'].resource_usage * m.getVarByName(
                    'y_' + str(u) + "," + str(v) + "," + str(t))
                if is_sep_resource:
                    resource_left_constr_sep[index_layer] += 0.5 * a['data']['inf_data'].resource_usage * m.getVarByName(
                        'y_' + str(u) + "," + str(v) + "," + str(t))
            else:
                resource_left_constr += 0.5 * a['data']['inf_data'].resource_usage * m.getVarByName(
                    'y_tilde_' + str(u) + "," + str(v) + "," + str(t))
                if is_sep_resource:
                    resource_left_constr_sep[index_layer] += 0.5 * a['data']['inf_data'].resource_usage * m.getVarByName(
                        'y_tilde_' + str(u) + "," + str(v) + "," + str(t))

        for n, d in n_hat_prime:
            index_layer = n[1]
            if T == 1:
                resource_left_constr += d['data']['inf_data'].resource_usage * m.getVarByName(
                    'w_' + str(n) + "," + str(t))
                if is_sep_resource:
                    resource_left_constr_sep[index_layer] += d['data']['inf_data'].resource_usage * m.getVarByName(
                        'w_' + str(n) + "," + str(t))
            else:
                resource_left_constr += d['data']['inf_data'].resource_usage * m.getVarByName(
                    'w_tilde_' + str(n) + "," + str(t))
                if is_sep_resource:
                    resource_left_constr_sep[index_layer] += d['data']['inf_data'].resource_usage * m.getVarByName(
                        'w_tilde_' + str(n) + "," + str(t))

        m.addConstr(resource_left_constr, GRB.LESS_EQUAL, total_resource,
                    "Resource availability constraint at " + str(t) + ".")
        if is_sep_resource:
            for k, _ in v_r.items():
                m.addConstr(resource_left_constr_sep[k], GRB.LESS_EQUAL, v_r[k],
                            "Resource availability constraint at " + str(t) + " for layer " + str(k) + ".")

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
        if m.status == 9:
            print(m.getAttr("Status"), ": SOLUTION NOT FOUND. (Check data and/or violated constraints).")
            print('\nThe following constraint(s) cannot be satisfied:')
            for c in m.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
        return None


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


def collect_results(m, controlled_layers, T, n_hat, n_hat_prime, a_hat_prime, S, coloc=True):
    """
    THis function compute the results (actions and costs) of the optimal results and write them to a
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


def apply_recovery(N, indp_results, t):
    """
    This function apply the restoration decisions (solution of INDP) to a gurobi model by changing the state of repaired
    elements to functional

    Parameters
    ----------
    N : :class:`~infrastructure.InfrastructureNetwork`
        The model of the interdependent network.
    indp_results : INDPResults
        A :class:`~indputils.INDPResults` object containing the optimal restoration decisions..
    t : int
        The time step to which the resukts should apply.

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
        "OPTIMISTIC", "PESSIMISTIC" or "INFO_SHARE". The default is "OPTIMISTIC".

    Returns
    -------
    functionality : dict
        A functionality dictionary used for input into indp..

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
                if a and not "/" in a:
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
                       infrastructure_data=True, topology='Random'):
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
        Sample number paramter of the initial disruption. The default is 0.
    v : int, list, optional
        Number of avaialable resources or resource cap. The default is 3.
    infrastructure_data : bool, optional
        If the data are for infrastructure. It should be set to False if a synthetic network
        is being analyzed. The default is True.
    topology : str, optional
        Topology of the synthetic network that is being analyzed. The default is 'Random'.

    Returns
    -------
    InterdepNet : :class:`~infrastructure.InfrastructureNetwork`
        The object containing the network data.
    v_temp : int, list
        Number of avaialable resources or resource cap. Used for synthetic networks
        where each sample network has a different resource cap.
    layers_temp : list
        List of layers in the analysis. Used for synthetic networks where each sample
        network has different layers.

    """
    layers_temp = []
    v_temp = 0
    if infrastructure_data:
        #    print "Loading Shelby County data..." #!!!
        InterdepNet = load_infrastructure_data(BASE_DIR=base_dir,
                                               external_interdependency_dir=external_interdependency_dir,
                                               sim_number=sim_number, cost_scale=cost_scale,
                                               magnitude=magnitude, v=v,
                                               data_format=infrastructure_data)
    #    print "Data loaded." #!!!
    else:
        InterdepNet, v_temp, layers_temp = load_synthetic_network(BASE_DIR=base_dir, topology=topology,
                                                                  config=magnitude, sample=sample,
                                                                  cost_scale=cost_scale)
    return InterdepNet, v_temp, layers_temp


def run_indp(params, layers=[1, 2, 3], controlled_layers=[], functionality={}, T=1, validate=False,
             save=True, suffix="", forced_actions=False, saveModel=False, print_cmd_line=True,
             co_location=True):
    """ Runs an INDP problem with specified parameters. Outputs to directory specified in params['OUTPUT_DIR'].
    :param params: Global parameters.
    :param layers: Layers to consider in the infrastructure network.
    :param T: Number of timesteps to optimize over.
    :param validate: Validate solution.
    """
    # Initialize failure scenario.
    InterdepNet = None
    if "N" not in params:
        InterdepNet = initialize_network(base_dir="../data/INDP_7-20-2015/",
                                         sim_number=params['SIM_NUMBER'],
                                         magnitude=params["MAGNITUDE"])
    else:
        InterdepNet = params["N"]
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"] = 1
    if not controlled_layers:
        controlled_layers = layers

    v_r = params["V"]
    if isinstance(v_r, (int)):
        outDirSuffixRes = str(v_r)
    else:
        outDirSuffixRes = str(sum([val for _, val in v_r.items()])) + '_fixed_layer_Cap'

    indp_results = INDPResults(params["L"])
    if T == 1:
        print("--Running INDP (T=1) or iterative INDP.")
        if print_cmd_line:
            print("Num iters=", params["NUM_ITERATIONS"])

        # Run INDP for 1 time step (original INDP).
        output_dir = params["OUTPUT_DIR"] + '_L' + str(len(layers)) + '_m' + str(
            params["MAGNITUDE"]) + "_v" + outDirSuffixRes
        # Initial calculations.
        if params['DYNAMIC_PARAMS']:
            original_N = copy.deepcopy(InterdepNet)  # !!! deepcopy
            dynamic_params = create_dynamic_param(params, N=original_N)
            dynamic_parameters(InterdepNet, original_N, 0, dynamic_params)
        results = indp(InterdepNet, 0, 1, layers, controlled_layers=controlled_layers,
                       functionality=functionality, co_location=co_location)
        indp_results = results[1]
        indp_results.add_components(0, INDPComponents.calculate_components(results[0], InterdepNet,
                                                                           layers=controlled_layers))
        for i in range(params["NUM_ITERATIONS"]):
            print("-Time Step (iINDP)", i + 1, "/", params["NUM_ITERATIONS"])
            if params['DYNAMIC_PARAMS']:
                dynamic_parameters(InterdepNet, original_N, i + 1, dynamic_params)
            results = indp(InterdepNet, v_r, T, layers, controlled_layers=controlled_layers,
                           forced_actions=forced_actions, co_location=co_location)
            indp_results.extend(results[1], t_offset=i + 1)
            if saveModel:
                save_INDP_model_to_file(results[0], output_dir + "/Model", i + 1)
            # Modify network to account for recovery and calculate components.
            apply_recovery(InterdepNet, indp_results, i + 1)
            indp_results.add_components(i + 1, INDPComponents.calculate_components(results[0], InterdepNet,
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
            params["MAGNITUDE"]) + "_v" + outDirSuffixRes

        print("Running td-INDP (T=" + str(T) + ", Window size=" + str(time_window_length) + ")")
        # Initial percolation calculations.
        results = indp(InterdepNet, 0, 1, layers, controlled_layers=controlled_layers,
                       functionality=functionality, co_location=co_location)
        indp_results = results[1]
        indp_results.add_components(0, INDPComponents.calculate_components(results[0], InterdepNet,
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
            results = indp(InterdepNet, v_r, time_window_length + 1, layers,
                           controlled_layers=controlled_layers,
                           functionality=functionality_t, forced_actions=forced_actions,
                           co_location=co_location)
            if saveModel:
                save_INDP_model_to_file(results[0], output_dir + "/Model", n + 1)
            if "WINDOW_LENGTH" in params:
                indp_results.extend(results[1], t_offset=n + 1, t_start=1, t_end=2)
                # Modify network for recovery actions and calculate components.
                apply_recovery(InterdepNet, results[1], 1)
                indp_results.add_components(n + 1,
                                            INDPComponents.calculate_components(results[0],
                                                                                InterdepNet, 1,
                                                                                layers=controlled_layers))
            else:
                indp_results.extend(results[1], t_offset=0)
                for t in range(1, T):
                    # Modify network to account for recovery actions.
                    apply_recovery(InterdepNet, indp_results, t)
                    indp_results.add_components(1, INDPComponents.calculate_components(results[0], InterdepNet, t,
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

def run_inrg(params, layers=[1, 2, 3], validate=False, player_ordering=[3, 1], suffix=""):
    InterdepNet = None
    output_dir = params["OUTPUT_DIR"] + "_m" + str(params["MAGNITUDE"]) + "_v" + str(params["V"])
    if "N" not in params:
        InterdepNet = initialize_network(base_dir="../data/INDP_7-20-2015/",
                                         sim_number=params['SIM_NUMBER'],
                                         magnitude=params["MAGNITUDE"])
        params["N"] = InterdepNet
    else:
        InterdepNet = params["N"]
    v_r = params["V"]
    # Initialize player result variables.
    player_strategies = {}
    for P in layers:
        player_strategies[P] = INDPResults()
    num_iterations = params["NUM_ITERATIONS"]
    params_temp = {}
    for key in params:
        params_temp[key] = params[key]
    params_temp["NUM_ITERATIONS"] = 1
    for i in range(num_iterations):
        curr_player_ordering = player_ordering
        if player_ordering == "RANDOM":
            curr_player_ordering = random.sample(layers, len(layers))
        for P in curr_player_ordering:
            print("Iteration", i, ", Player", P)
            # functionality=create_functionality_matrix(InterdepNet,1,[x for x in layers if x != P],strategy_type="REALISTIC")
            results = run_indp(params_temp, layers, controlled_layers=[P], T=1, save=False,
                               suffix="P" + str(P) + "_i" + str(i), forced_actions=True)
            # print params["N"].G.node[(5,3)]['data']['inf_data'].functionality
            if i == 0:
                player_strategies[P] = results
            else:
                player_strategies[P].extend(results, t_offset=i + 1, t_start=1, t_end=2)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for P in layers:
        player_strategies[P].to_csv(output_dir, params["SIM_NUMBER"], suffix="P" + str(P) + "_" + suffix)


def create_dynamic_param(params, N=None):
    print("Computing dislocation data...")
    dynamic_param_dict = params['DYNAMIC_PARAMS']
    return_type = dynamic_param_dict['RETURN']
    dp_dict_col = ['time', 'node', 'current pop', 'total pop']
    dynamic_params = {}
    if dynamic_param_dict['TYPE'] == 'shelby_adopted':
        print("Reading dislocation data from file...")
        net_names = {'water': 1, 'gas': 2, 'power': 3, 'telecom': 4}
        for key, val in net_names.items():
            filename = dynamic_param_dict['DIR'] + 'dynamic_demand_' + return_type + '_' + key + '.pkl'
            with open(filename, 'rb') as f:
                dd_df = pickle.load(f)
            dynamic_params[val] = dd_df[(dd_df['sce'] == params["MAGNITUDE"]) & \
                                        (dd_df['set'] == params["SIM_NUMBER"])]
    elif dynamic_param_dict['TYPE'] == 'incore':
        testbed = dynamic_param_dict['TESTBED']
        file_dir = dynamic_param_dict['DIR'] + testbed + '/Damage_scenarios/'
        if os.path.exists(file_dir + 'pop_dislocation_data.pkl'):
            print("Reading dislocation data from file...")
            with open(file_dir + 'pop_dislocation_data.pkl', 'rb') as f:
                dynamic_params = pickle.load(f)
            return dynamic_params

        pop_dislocation = pd.read_csv(file_dir + 'pop-dislocation-results.csv', low_memory=False)
        mapping_data = pd.read_csv(file_dir + testbed + '_interdependency_table.csv', low_memory=False)
        total_num_bldg = mapping_data.shape[0]
        # total_num_hh = pop_dislocation[~(pd.isna(pop_dislocation['guid']))]
        T = 10
        for n, d in N.G.nodes(data=True):
            if n[1] not in dynamic_params.keys():
                dynamic_params[n[1]] = pd.DataFrame(columns=dp_dict_col)
            guid = d['data']['inf_data'].guid
            serv_area = mapping_data[mapping_data['substations_guid'] == guid]
            # compute dynamic_params
            num_dilocated = {t: 0 for t in range(T + 1)}
            total_pop = 0
            for _, bldg in serv_area.iterrows():
                pop_bldg_dict = pop_dislocation[pop_dislocation['guid'] == bldg['buildings_guid']]
                for _, hh in pop_bldg_dict.iterrows():
                    total_pop += hh['numprec'] if ~np.isnan(hh['numprec']) else 0
                    if hh['dislocated']:
                        # !!! Lumebrton dislocation time paramters
                        dt_params = {'DS1': 1.00, 'DS2': 2.33, 'DS3': 2.49, 'DS4': 3.62,
                                     'white': 0.78, 'black': 0.88, 'hispanic': 0.83,
                                     'income': -0.00, 'insurance': 1.06}
                        race_white = 1 if hh['race'] == 1 else 0
                        race_balck = 1 if hh['race'] == 2 else 0
                        hispan = hh['hispan'] if ~np.isnan(hh['hispan']) else 0
                        # !!! verfy that the explanatory variable correspond to columns in dt_params
                        linear_term = hh['insignific'] * dt_params['DS1'] + \
                                      hh['moderate'] * dt_params['DS2'] + \
                                      hh['heavy'] * dt_params['DS3'] + \
                                      hh['complete'] * dt_params['DS4'] + \
                                      race_white * dt_params['white'] + \
                                      race_balck * dt_params['black'] + \
                                      hispan * dt_params['hispanic'] + \
                                      np.random.choice([0, 1], p=[.15, .85]) * dt_params[
                                          'insurance']  # !!! insurance data
                        # hh['randincome']/1000*dt_params['income']+\#!!! income data
                        disloc_time = np.exp(linear_term)
                        return_time = math.ceil(disloc_time / 7)  # !!! assume each time step is one week
                        for t in range(return_time):
                            if t <= T:
                                num_dilocated[t] += hh['numprec'] if ~np.isnan(hh['numprec']) else 0
            for t in range(T + 1):
                values = [t, n[0], total_pop - num_dilocated[t], total_pop]
                dynamic_params[n[1]] = dynamic_params[n[1]].append(dict(zip(dp_dict_col, values)),
                                                                   ignore_index=True)
        with open(file_dir + 'pop_dislocation_data.pkl', 'wb') as f:
            pickle.dump(dynamic_params, f)
    return dynamic_params


def dynamic_parameters(N, original_N, t, dynamic_params):
    for n, d in N.G.nodes(data=True):
        data = dynamic_params[d['data']['inf_data'].net_id]
        if d['data']['inf_data'].demand < 0:
            current_pop = data.loc[(data['node'] == n[0]) & (data['time'] == t), 'current pop'].iloc[0]
            total_pop = data.loc[(data['node'] == n[0]) & (data['time'] == t), 'total pop'].iloc[0]
            original_demand = original_N.G.nodes[n]['data']['inf_data'].demand
            d['data']['inf_data'].demand = original_demand * current_pop / total_pop


def save_INDP_model_to_file(model, outModelDir, t, l=0, suffix=''):
    if not os.path.exists(outModelDir):
        os.makedirs(outModelDir)
    # Write models to file
    lname = "/Model_t%d_l%d_%s.lp" % (t, l, suffix)
    model.write(outModelDir + lname)
    model.update()
    # Write solution to file
    sname = "/Solution_t%d_l%d_%s.txt" % (t, l, suffix)
    fileID = open(outModelDir + sname, 'w')
    for vv in model.getVars():
        fileID.write('%s %g\n' % (vv.varName, vv.x))
    fileID.write('Obj: %g' % model.objVal)
    fileID.close()


def initialize_sample_network(layers=[1, 2]):
    """ Initializes sample network
    :param layers: (Currently not used).
    :returns: An interdependent InfrastructureNetwork.
    """
    InterdepNet = InfrastructureNetwork("sample_network")
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
        nn.resource_usage = 1
        if n in failed_nodes:
            nn.functionality = 0.0
            nn.repaired = 0.0
        InterdepNet.G.add_node((nn.local_id, nn.net_id), data={'inf_data': nn})
        global_index += 1
    for s in space_to_nodes_dict:
        InterdepNet.S.append(InfrastructureSpace(s, 0))
        for n in space_to_nodes_dict[s]:
            InterdepNet.G.nodes[n]['data']['inf_data'].space = s
    for a in arc_list:
        aa = InfrastructureArc(a[0][0], a[1][0], a[0][1])
        aa.flow_cost = 1
        aa.capacity = 50
        InterdepNet.G.add_edge((aa.source, aa.layer), (aa.dest, aa.layer), data={'inf_data': aa})
    for g in interdep_list:
        aa = InfrastructureInterdepArc(g[0][0], g[1][0], g[0][1], g[1][1], 1.0)
        InterdepNet.G.add_edge((aa.source, aa.source_layer), (aa.dest, aa.dest_layer), data={'inf_data': aa})
    return InterdepNet


def plot_indp_sample(params, folderSuffix="", suffix=""):
    plt.figure(figsize=(16, 8))
    if 3 in params["L"]:
        plt.figure(figsize=(16, 10))
    InterdepNet = initialize_sample_network(layers=params["L"])
    pos = nx.spring_layout(InterdepNet.G)
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
    pos[(2, 1)][1] = 0.0
    pos[(4, 1)][1] = 0.0
    pos[(6, 1)][1] = 0.0
    pos[(1, 1)][1] = 1.0
    pos[(3, 1)][1] = 1.0
    pos[(5, 1)][1] = 1.0
    pos[(8, 2)][1] = 2.0
    pos[(10, 2)][1] = 2.0
    pos[(12, 2)][1] = 2.0
    pos[(7, 2)][1] = 3.0
    pos[(9, 2)][1] = 3.0
    pos[(11, 2)][1] = 3.0
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
        pos[(13, 3)][1] = 5.0
        pos[(14, 3)][1] = 4.0
        pos[(15, 3)][1] = 5.0
        pos[(16, 3)][1] = 4.0
        pos[(17, 3)][1] = 5.0
        pos[(18, 3)][1] = 4.0
        node_dict[3] = [(13, 3), (14, 3), (15, 3), (16, 3), (17, 3), (18, 3)]
        node_dict[31] = [(13, 3)]
        node_dict[32] = [(14, 3), (15, 3), (16, 3), (17, 3), (18, 3)]
        arc_dict[3] = [((13, 3), (15, 3)), ((14, 3), (15, 3)), ((14, 3), (16, 3)),
                       ((17, 3), (15, 3)), ((17, 3), (16, 3)), ((17, 3), (18, 3))]

    labels = {}
    for n, d in InterdepNet.G.nodes(data=True):
        labels[n] = "%d[%d]" % (n[0], d['data']['inf_data'].demand)
    pos_moved = {}
    for key, value in pos.items():
        pos_moved[key] = [0, 0]
        pos_moved[key][0] = pos[key][0] - 0.2
        pos_moved[key][1] = pos[key][1] + 0.2

    v_r = params["V"]
    if isinstance(v_r, (int)):
        totalResource = v_r
    else:
        totalResource = sum([val for _, val in v_r.items()])

    output_dir = params["OUTPUT_DIR"] + '_L' + str(len(params["L"])) + '_m' + str(params["MAGNITUDE"]) + "_v" + str(
        totalResource) + folderSuffix
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
        nx.draw(InterdepNet.G, pos, node_color='w')
        nx.draw_networkx_labels(InterdepNet.G, labels=labels, pos=pos,
                                font_color='w', font_family='CMU Serif')  # ,font_weight='bold'
        nx.draw_networkx_nodes(InterdepNet.G, pos, nodelist=node_dict[1], node_color='#b51717', node_size=1100,
                               alpha=0.7)
        nx.draw_networkx_nodes(InterdepNet.G, pos, nodelist=node_dict[2], node_color='#005f98', node_size=1100,
                               alpha=0.7)
        nx.draw_networkx_nodes(InterdepNet.G, pos_moved, nodelist=node_dict[12], node_color='k', node_shape="X",
                               node_size=150)
        nx.draw_networkx_nodes(InterdepNet.G, pos_moved, nodelist=node_dict[22], node_color='k', node_shape="X",
                               node_size=150)
        nx.draw_networkx_edges(InterdepNet.G, pos, edgelist=arc_dict[1], width=1, alpha=0.9, edge_color='r')
        nx.draw_networkx_edges(InterdepNet.G, pos, edgelist=arc_dict[2], width=1, alpha=0.9, edge_color='b')
        if 3 in params["L"]:
            nx.draw_networkx_nodes(InterdepNet.G, pos, nodelist=node_dict[3], node_color='#009181', node_size=1100,
                                   alpha=0.7)
            nx.draw_networkx_nodes(InterdepNet.G, pos_moved, nodelist=node_dict[32], node_color='k', node_shape="X",
                                   node_size=150)
            nx.draw_networkx_edges(InterdepNet.G, pos, edgelist=arc_dict[3], width=1, alpha=0.9, edge_color='g')
    plt.tight_layout()
    plt.savefig(output_dir + '/plot_net' + suffix + '.png', dpi=300)
