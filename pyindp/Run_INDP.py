import sys
import random
from indp import *
from Dindp import *
import gametree
import os.path

def batch_run(params,layers,player_ordering=[3,1],judgment_type="OPTIMISTIC"):
    """ Batch run INDP optimization problem for all samples (currently 1-1000), given global parameters. 
    Format for params:
    "NUM_ITERATIONS": For single timestep INDP, specifies how many iterations to run.
                      For InfoShare, specifies how many rounds of information sharing to perform.
    "OUTPUT_DIR"    : Directory to output results.
    "MAGNITUDE"     : Magnitude of earthquake. Used to look up failure scenarios.
    "V"             : How many resources used in a scenario. Used to look up failure scenarios, and specify v_r for indp.
    "T"             : Number of timesteps to optimize over. Used for td-INDP and InfoShare.
    "WINDOW_LENGTH" : Slding time window length for td-INDP (for efficiency).
    "N"             : InfrastructureNetwork to use in indp.
    "SIM_NUMBER"    : What failure scenario simulation to use. Used to look up failure scenarios."""
    if "N" not in params:
        print("Initializing network...")
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",external_interdependency_dir="../data/INDP_4-12-2016",sim_number=0,magnitude=params["MAGNITUDE"],v=params["V"])
        print(InterdepNet)
    else:
        InterdepNet=params["N"]
    for i in range(1,2):
        print("Running sample",i,"...")
        add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=params["MAGNITUDE"],v=params["V"],sim_number=i)
        params["N"]=InterdepNet
        params["SIM_NUMBER"]=i
        if params["ALGORITHM"]=="INDP":
            run_indp(params,validate=False,T=params["T"],layers=layers,controlled_layers=layers,saveModel=False)
        elif params["ALGORITHM"]=="INFO_SHARE":
            run_info_share(params,layers=layers,T=params["T"])
        elif params["ALGORITHM"]=="INRG":
            run_inrg(params,layers=layers,player_ordering=player_ordering)
        elif params["ALGORITHM"]=="BACKWARDS_INDUCTION":
            gametree.run_backwards_induction(InterdepNet,i,players=layers,player_ordering=player_ordering,T=params["T"],outdir=params["OUTPUT_DIR"])
        elif params["ALGORITHM"]=="JUDGMENT_CALL":
            run_judgment_call(params,layers=layers,T=params["T"],saveJCModel=True,judgment_type=judgment_type)

def single_scenario_run(params,layers,player_ordering=[3,1],num_samples=1):
    """ Batch run INDP optimization problem for all samples (currently 1-1000), given global parameters.                  
    Format for params:
    "NUM_ITERATIONS": For single timestep INDP, specifies how many iterations to run.
                      For InfoShare, specifies how many rounds of information sharing to perform. 
    "OUTPUT_DIR"    : Directory to output results.
    "MAGNITUDE"     : Magnitude of earthquake. Used to look up failure scenarios.                                                                                   
    "V"             : How many resources used in a scenario. Used to look up failure scenarios, and specify v_r for indp.                                           
    "T"             : Number of timesteps to optimize over. Used for td-INDP and InfoShare.                                                                         
    "WINDOW_LENGTH" : Slding time window length for td-INDP (for efficiency).                                                                                       
    "N"             : InfrastructureNetwork to use in indp.                                                                                                         
    "SIM_NUMBER"    : What failure scenario simulation to use. Used to look up failure scenarios."""
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=0,magnitude=params["MAGNITUDE"],v=params["V"])
    else:
        InterdepNet=params["N"]
    for i in range(num_samples):
        print("Running sample",i,"...")
        add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=params["MAGNITUDE"],v=params["V"],sim_number=params["SIM_NUMBER"])
        params["N"]=InterdepNet
        suffix=str(i)
        if params["ALGORITHM"]=="INDP":
            run_indp(params,validate=False,T=params["T"],layers=layers,controlled_layers=layers,suffix=suffix)
        elif params["ALGORITHM"]=="INFO_SHARE":
            run_info_share(params,layers=layers,T=params["T"],suffix=suffix)
        elif params["ALGORITHM"]=="INRG":
            run_inrg(params,layers=layers,player_ordering=player_ordering,suffix=suffix)

def max_damage_sample(mag):
    InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=0,magnitude=mag,v=2)
    max_damaged_nodes=0
    max_sim=0
    for i in range(1,1001):
        print(str(i))
        add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=mag,v=2,sim_number=i)
        damaged_nodes=len([n for n,d in InterdepNet.G.nodes_iter(data=True) if d['data']['inf_data'].functionality==0.0])
        if damaged_nodes > max_damaged_nodes:
            max_damaged_nodes=damaged_nodes
            max_sim=i
    return max_sim

def run_indp_sample():
    InterdepNet=load_sample()
    params={"NUM_ITERATIONS":4, "OUTPUT_DIR":'../results/sample_indp_2-1_samp2',"V":2,"T":4,"WINDOW_LENGTH":4,"ALGORITHM":"INDP"}
    params["N"]=InterdepNet
    params["MAGNITUDE"]=0
    params["SIM_NUMBER"]=0
    run_indp(params,layers=[1,2],T=params["T"],suffix="")

def run_indp_L3_V3(mags):
    for m in mags:
        params={"NUM_ITERATIONS":20,"OUTPUT_DIR":'../results/indp_results_L3',"MAGNITUDE":m,"V":3,"T":1,"ALGORITHM":"INDP"}
        batch_run(params,layers=[1,2,3])
        
def run_indp_L3_V3_Layer_Res_Cap(mags):
    for m in mags:
        params={"NUM_ITERATIONS":20,"OUTPUT_DIR":'../results/indp_results_L3',"MAGNITUDE":m,"V":[1,1,1],"T":1,"ALGORITHM":"INDP"}
        batch_run(params,layers=[1,2,3])

def run_indp_L2_V2(mags):
    for m in mags:
        params={"NUM_ITERATIONS":50,"OUTPUT_DIR":'../results/indp_results_L2',"MAGNITUDE":m,"V":2,"T":1,"ALGORITHM":"INDP"}
        batch_run(params,layers=[1,3])

def run_indp_L1_V1(mags):
    for m in mags:
        params={"NUM_ITERATIONS":50,"OUTPUT_DIR":'../results/indp_results_L1',"MAGNITUDE":m,"V":1,"T":1,"ALGORITHM":"INDP"}
        batch_run(params,layers=[3])

def run_indp_L2_V2_inf(mags):
    params={"SIM_NUMBER":"INF","NUM_ITERATIONS":212,"OUTPUT_DIR":'../results/indp_results_L1_inf',"MAGNITUDE":0,"V":2,"T":1,"ALGORITHM":"INDP"}
    single_scenario_run(params,layers=[1,3])

def run_tdindp_L2_V2(mags):
    for m in mags:
        params={"NUM_ITERATIONS":1,"OUTPUT_DIR":'../results/tdindp_results_L2',"MAGNITUDE":m,"V":2,"T":50,"WINDOW_LENGTH":3,"ALGORITHM":"INDP"}
        batch_run(params,layers=[1,3])

def run_tdindp_L2_V2_inf(mags):
    params={"SIM_NUMBER":"INF","NUM_ITERATIONS":1,"OUTPUT_DIR":'../results/tdindp_results_L2_inf',"MAGNITUDE":0,"V":2,"T":212,"WINDOW_LENGTH":3,"ALGORITHM":"INDP"}
    single_scenario_run(params,layers=[1,3])

def run_tdindp_L3_V3(mags):
    for m in mags:
        params={"NUM_ITERATIONS":1,"OUTPUT_DIR":'../results/tdindp_results_L3',"MAGNITUDE":m,"V":3,"T":20,"WINDOW_LENGTH":3,"ALGORITHM":"INDP"}
        batch_run(params,layers=[1,2,3])  
        
def run_tdindp_L3_V3_Layer_Res_Cap(mags):
    for m in mags:
        params={"NUM_ITERATIONS":1,"OUTPUT_DIR":'../results/tdindp_results_L3',"MAGNITUDE":m,"V":[1,1,1],"T":20,"WINDOW_LENGTH":3,"ALGORITHM":"INDP"}
        batch_run(params,layers=[1,2,3])

def run_dindp_L3_V3(mags,judgment_type="OPTIMISTIC"):
    for m in mags:
        params={"NUM_ITERATIONS":20,"OUTPUT_DIR":'../results/judgeCall_'+judgment_type+'_results_L3',"MAGNITUDE":m,"V":[1,1,1],"T":1,"ALGORITHM":"JUDGMENT_CALL"}
        batch_run(params,layers=[1,2,3],judgment_type=judgment_type)
        


def main():
    """ Run as: python Run_INDP.py <algorithm=indp|tdindp|infoshare> <num_layers=1|2|3> <num_resources=1|2|3> <magnitude=6|8|9> """
    args=sys.argv
    algorithm=args[1]
    num_layers=args[2]
    num_resources=args[3]
    magnitude=int(args[4])
    flip=""
    if len(args) > 5:
        flip=args[5]
    fun_name="run_"+algorithm+"_L"+num_layers+"_V"+num_resources
    if flip != "":
        if flip == "random":
            fun_name+="_random_flip"
        elif flip == "1":
            fun_name+="_random_flip1"
        elif flip == "random-order":
            fun_name+="_random_ordering"
        elif flip == "seeded-flip":
            fun_name+="_seeded_flip"
        elif flip == "random-order-seeded-flip":
            fun_name+="_random_ordering_seeded_flip"
        elif flip == "inf":
            fun_name+="_inf"
    mags=[magnitude]
    if fun_name in globals():
        globals()[fun_name](mags)

if __name__ == "__main__":
#    main()
    mags = [8]
#    failSce = read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/",magnitude=8)
#    for jc in ["PESSIMISTIC","OPTIMISTIC","DEMAND","DET-DEMAND","RANDOM"]:
#        run_dindp_L3_V3(mags,judgment_type=jc)
#    run_indp_L3_V3(mags)
#    run_indp_L3_V3_Layer_Res_Cap(mags)
#    run_tdindp_L3_V3(mags)
#    run_tdindp_L3_V3_Layer_Res_Cap(mags)
    run_dindp_L3_V3(mags,judgment_type='PESSIMISTIC')

#    method_name = ['judgeCall_OPTIMISTIC_results','judgeCall_PESSIMISTIC_results',
#                   'judgeCall_DEMAND_results','judgeCall_DET-DEMAND_results',
#                   'judgeCall_RANDOM_results','indp_results','indp_results',
#            'tdindp_results','tdindp_results']
#    resource_cap = ['_Layer_Res_Cap','_Layer_Res_Cap','_Layer_Res_Cap',
#                    '_Layer_Res_Cap','_Layer_Res_Cap',
#                    '','_Layer_Res_Cap','','_Layer_Res_Cap']
#    suffixes = ['Real_sum','Real_sum','Real_sum','Real_sum','Real_sum','','','','']
    
#    method_name = ['judgeCall_OPTIMISTIC_results','judgeCall_PESSIMISTIC_results',
#                   'judgeCall_RANDOM_results','indp_results','tdindp_results']
#    resource_cap = ['_Layer_Res_Cap','_Layer_Res_Cap','_Layer_Res_Cap','','']
#    suffixes = ['Real_sum','Real_sum','Real_sum','','']
#
#    
#    sample_range=range(1,51)
#    df = aggregate_results(mags,method_name,resource_cap,suffixes,3,3,sample_range)
#    df = correct_tdindp_results(df,mags,method_name,sample_range)
#    plot_results(df,cost_type='Total',mags=mags)