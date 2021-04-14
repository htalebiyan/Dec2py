from infrastructure import *
from indputils import *
from gurobipy import *
from indp import *
import string
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random
import time
import sys
import scipy.io
try:
    import matlab.engine
except ModuleNotFoundError:
    print("Can't find module 'matlab.engine'")

def apply_recovery(N,indp_results,t):
    for action in indp_results[t]['actions']:
        if "/" in action:
            # Edge recovery action.
            data=action.split("/")
            src=tuple([int(x) for x in data[0].split(".")])
            dst=tuple([int(x) for x in data[1].split(".")])
            N.G[src][dst]['data']['inf_data'].functionality=1.0
        else:
            # Node recovery action.
            node=tuple([int(x) for x in action.split(".")])
            #print "Applying recovery:",node
            N.G.nodes[node]['data']['inf_data'].repaired=1.0
            N.G.nodes[node]['data']['inf_data'].functionality=1.0

def run_mh(params, layers=[1,2,3], controlled_layers=[], functionality={},T=1, validate=False,
             save=True,suffix="", forced_actions=False, saveModel=False, print_cmd_line=True,
             dynamic_params=None, co_location=True):
    """ Runs an INDP problem with specified parameters. Outputs to directory specified in params['OUTPUT_DIR'].
    :param params: Global parameters.
    :param layers: Layers to consider in the infrastructure network.
    :param T: Number of timesteps to optimize over.
    :param validate: Validate solution.
    """
    # Initialize failure scenario.
    InterdepNet=None
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=params['SIM_NUMBER'],magnitude=params["MAGNITUDE"])
    else:
        InterdepNet=params["N"]
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"] = 1
    if not controlled_layers:
        controlled_layers = layers

    v_r=params["V"]
    if isinstance(v_r, (int)):
        outDirSuffixRes = str(v_r)
    else:
        outDirSuffixRes = str(sum([val for _, val in v_r.items()]))+'_fixed_layer_Cap'

    indp_results=INDPResults(params["L"])
    if T == 1:
        print("--Running iterative MH.")
        if print_cmd_line:
            print("Num iters=",params["NUM_ITERATIONS"])

        # Run INDP for 1 time step (original INDP).
        output_dir=params["OUTPUT_DIR"]+'_L'+str(len(layers))+'_m'+str(params["MAGNITUDE"])+"_v"+outDirSuffixRes
        # Initial calculations.
        if dynamic_params:
            original_N = copy.deepcopy(InterdepNet) #!!! deepcopy
            dynamic_parameters(InterdepNet, original_N, 0, dynamic_params)
        results=indp(InterdepNet,0,1,layers,controlled_layers=controlled_layers,
                     functionality=functionality, co_location=co_location)
        indp_results=results[1]
        indp_results.add_components(0,INDPComponents.calculate_components(results[0],InterdepNet,layers=controlled_layers))
        for i in range(params["NUM_ITERATIONS"]):
            print("-Time Step (MH)",i+1,"/",params["NUM_ITERATIONS"])
            if dynamic_params:
                dynamic_parameters(InterdepNet, original_N, i+1, dynamic_params)
            results=indp(InterdepNet, v_r, T, layers, controlled_layers=controlled_layers,
                         forced_actions=forced_actions, co_location=co_location)
            ### Extract matrices and vectors
            m = results[0]
            var_index = {v.VarName.replace("(", "").replace(")", "").replace(",", "_").\
                         replace(" ", "").replace("+", "_p").replace("-", "_m"):\
                             i for i, v in enumerate(m.getVars())}
            constr_rhs= {c.ConstrName.replace("(", "").replace(")", "").replace(",", "_").\
                           replace(" ", "").replace(".", ""): c.RHS for i, c in enumerate(m.getConstrs())}
            constr_sense= {c.ConstrName.replace("(", "").replace(")", "").replace(",", "_").\
                           replace(" ", "").replace(".", ""): c.sense for i, c in enumerate(m.getConstrs())}
            obj_coeffs = m.getAttr('Obj', m.getVars())
            A = m.getA()
            opt_sol = {}
            for v in m.getVars():
                opt_sol[v.varName.replace("(", "").replace(")", "").replace(",", "_").\
                         replace(" ", "").replace("+", "_p").replace("-", "_m")]= v.x
            scipy.io.savemat('./Metaheuristics/arrdata.mat', mdict={'A': A})
            ### Run GA in Matlab
            eng = matlab.engine.start_matlab("-desktop") #!!! Send as an argument for debugging in MATLAB: "-desktop"
            eng.cd('./Metaheuristics/')
            eng.eval('dbstop in main.m at 3', nargout=0) #!!! 
            result_mh = eng.main(var_index, constr_rhs, constr_sense, obj_coeffs, opt_sol)
    return result_mh #!!!
    #         if saveModel:
    #             save_INDP_model_to_file(results[0],output_dir+"/Model",i+1)
    #         # Modify network to account for recovery and calculate components.
    #         apply_recovery(InterdepNet,indp_results,i+1)
    # # Save results of current simulation.
    # if save:
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     indp_results.to_csv(output_dir,params["SIM_NUMBER"],suffix=suffix)
    #     if not os.path.exists(output_dir+'/agents'):
    #         os.makedirs(output_dir+'/agents')
    #     indp_results.to_csv_layer(output_dir+'/agents',params["SIM_NUMBER"],suffix=suffix)
    # return indp_results


