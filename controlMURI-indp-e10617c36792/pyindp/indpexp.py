import sys
import random
from indp import *
import gametree
import os.path

def batch_run(params,layers,player_ordering=[3,1]):
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
        print "Initializing network..."
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",external_interdependency_dir="../data/INDP_4-12-2016",sim_number=0,magnitude=params["MAGNITUDE"],v=params["V"])
        print InterdepNet
    else:
        InterdepNet=params["N"]
    for i in range(1,2):
        print "Running sample",i,"..."
        add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=params["MAGNITUDE"],v=params["V"],sim_number=i)
        params["N"]=InterdepNet
        params["SIM_NUMBER"]=i
        if params["ALGORITHM"]=="INDP":
            run_indp(params,validate=False,T=params["T"],layers=layers,controlled_layers=layers)
        elif params["ALGORITHM"]=="INFO_SHARE":
            run_info_share(params,layers=layers,T=params["T"])
        elif params["ALGORITHM"]=="INRG":
            run_inrg(params,layers=layers,player_ordering=player_ordering)
        elif params["ALGORITHM"]=="BACKWARDS_INDUCTION":
            gametree.run_backwards_induction(InterdepNet,i,players=layers,player_ordering=player_ordering,T=params["T"],outdir=params["OUTPUT_DIR"])

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
        print "Running sample",i,"..."
        add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=params["MAGNITUDE"],v=params["V"],sim_number=params["SIM_NUMBER"])
        params["N"]=InterdepNet
        suffix=`i`
        if params["ALGORITHM"]=="INDP":
            run_indp(params,validate=False,T=params["T"],layers=layers,controlled_layers=layers,suffix=suffix)
        elif params["ALGORITHM"]=="INFO_SHARE":
            run_info_share(params,layers=layers,T=params["T"],suffix=suffix)
        elif params["ALGORITHM"]=="INRG":
            run_inrg(params,layers=layers,player_ordering=player_ordering,suffix=suffix)

def one_flip_run(params,layers,player_ordering=[3,1]):
    InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=0,magnitude=params["MAGNITUDE"],v=params["V"])
    interdependent_arcs=[(u,v,a) for u,v,a in InterdepNet.G.edges_iter(data=True) if a['data']['inf_data'].is_interdep and u[1] in layers and v[1] in layers]
    print "Num interdep arcs.",len(interdependent_arcs)
    add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=params["MAGNITUDE"],v=params["V"],sim_number=params["SIM_NUMBER"])
    for i in range(len(interdependent_arcs)):
        print "Running sample",i,"..."
        InterdepNet_prime=InterdepNet.copy()
        suffix=`i`
        flipped_arc=interdependent_arcs[i]
        print "Flipping",flipped_arc[0],",",flipped_arc[1]
        InterdepNet_prime.G.add_edge(flipped_arc[1],flipped_arc[0],attr_dict=flipped_arc[2])
        InterdepNet_prime.G.remove_edge(flipped_arc[0],flipped_arc[1])
        
        params["N"]=InterdepNet_prime
        print [(u,v) for u,v,a in InterdepNet_prime.G.edges_iter(data=True) if a['data']['inf_data'].is_interdep and u[1] in layers and v[1] in layers]
        if params["ALGORITHM"]=="INDP":
            run_indp(params,validate=False,T=params["T"],layers=layers,controlled_layers=layers,suffix=suffix)
        elif params["ALGORITHM"]=="INFO_SHARE":
            run_info_share(params,layers=layers,T=params["T"],suffix=suffix)
        elif params["ALGORITHM"]=="INRG":
            run_inrg(params,layers=layers,suffix=suffix)

def random_flip_run(params,layers,player_ordering=[3,1],num_samples=100,flip_prob=0.1):
    InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=0,magnitude=params["MAGNITUDE"],v=params["V"])
    interdependent_arcs=[(u,v,a) for u,v,a in InterdepNet.G.edges_iter(data=True) if a['data']['inf_data'].is_interdep and u[1] in layers and v[1] in layers]
    add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=params["MAGNITUDE"],v=params["V"],sim_number=params["SIM_NUMBER"])
    print "Num interdep arcs.",len(interdependent_arcs)
    for i in range(num_samples):
        print "Running sample",i,"..."
        InterdepNet_prime=InterdepNet.copy()
        suffix=`i`
        for j in range(len(interdependent_arcs)):
            samp=random.random()
            if samp < flip_prob:
                flipped_arc=interdependent_arcs[j]
                print "Flipping",flipped_arc[0],",",flipped_arc[1]
                InterdepNet_prime.G.add_edge(flipped_arc[1],flipped_arc[0],attr_dict=flipped_arc[2])
                InterdepNet_prime.G.remove_edge(flipped_arc[0],flipped_arc[1])

        params["N"]=InterdepNet_prime
        print [(u,v) for u,v,a in InterdepNet_prime.G.edges_iter(data=True) if a['data']['inf_data'].is_interdep and u[1] in layers and v[1] in layers]
        if params["ALGORITHM"]=="INDP":
            run_indp(params,validate=False,T=params["T"],layers=layers,controlled_layers=layers,suffix=suffix)
        elif params["ALGORITHM"]=="INFO_SHARE":
            run_info_share(params,layers=layers,T=params["T"],suffix=suffix)
        elif params["ALGORITHM"]=="INRG":
            run_inrg(params,layers=layers,suffix=suffix)

def random_flip(params, layers, seed, flip_prob=0.1):
    InterdepNet=params["N"]
    interdependent_arcs=[(u,v,a) for u,v,a in InterdepNet.G.edges_iter(data=True) if a['data']['inf_data'].is_interdep and u[1] in layers and v[1] in layers]
    #add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=params["MAGNITUDE"],v=params["V"],sim_number=params["SIM_NUMBER"])
    InterdepNet_prime=InterdepNet.copy()
    random.seed(seed)
    for j in range(len(interdependent_arcs)):
        samp=random.random()
        if samp < flip_prob:
            flipped_arc=interdependent_arcs[j]
            print "Flipping",flipped_arc
            InterdepNet_prime.G.add_edge(flipped_arc[1],flipped_arc[0],attr_dict=flipped_arc[2])
            InterdepNet_prime.G.remove_edge(flipped_arc[0],flipped_arc[1])
    return InterdepNet_prime

def max_damage_sample(mag):
    InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=0,magnitude=mag,v=2)
    max_damaged_nodes=0
    max_sim=0
    for i in range(1,1001):
        print `i`
        add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=mag,v=2,sim_number=i)
        damaged_nodes=len([n for n,d in InterdepNet.G.nodes_iter(data=True) if d['data']['inf_data'].functionality==0.0])
        if damaged_nodes > max_damaged_nodes:
            max_damaged_nodes=damaged_nodes
            max_sim=i
    return max_sim

def run_infoshare_sample():
    InterdepNet=load_sample()
    params={"NUM_ITERATIONS":7,"OUTPUT_DIR":'../results/sample',"V":1,"T":3,"WINDOW_LENGTH":3,"ALGORITHM":"INFO_SHARE"}
    params["N"]=InterdepNet
    params["MAGNITUDE"]=0
    params["SIM_NUMBER"]=0
    run_info_share(params,layers=[1,2],T=params["T"],suffix="")

def run_inrg_sample():
    InterdepNet=load_sample()
    params={"NUM_ITERATIONS":4,"OUTPUT_DIR":'../results/sample_inrg_1-2_samp2',"V":1,"T":1,"ALGORITHM":"INRG"}
    #params={"NUM_ITERATIONS":7,"OUTPUT_DIR":'../results/sample',"V":1,"T":3,"WINDOW_LENGTH":3,"ALGORITHM":"INFO_SHARE"}
    params["N"]=InterdepNet
    params["MAGNITUDE"]=0
    params["SIM_NUMBER"]=0
    run_inrg(params,layers=[1,2],player_ordering=[1,2],suffix="")

def run_indp_sample():
    InterdepNet=load_sample()
    params={"NUM_ITERATIONS":4, "OUTPUT_DIR":'../results/sample_indp_2-1_samp2',"V":2,"T":4,"WINDOW_LENGTH":4,"ALGORITHM":"INDP"}
    params["N"]=InterdepNet
    params["MAGNITUDE"]=0
    params["SIM_NUMBER"]=0
    run_indp(params,layers=[1,2],T=params["T"],suffix="")

def run_percolation_model():
    #import percisland.supplynet as sn
    import percisland.percisland as pi
    import percisland.powernetgen as png
    p_S=0.3
    p_C=0.7
    sample=1
    N=1000
    T=1
    demand_distr="EXPONWEIB"
    network=png.load_power_grid(N,sample,indir="../../percisland/networks")
    outfile=None
    loadfile="../../percisland/networks/power_grid_"+`sample`+"_N"+`N`+"_loads.csv"
    print "not creating thing."
    if not os.path.isfile(loadfile):
        print "creating thing."
        outfile=loadfile
        loadfile=None
    supply_net=pi.generate_supply_network(p_S,p_C,n=len(network),demand_distr=demand_distr,N_c=network,outfile=outfile,loadfile=loadfile)
    for n,d in supply_net.N.nodes_iter(data=True):
        print n,d['data'].supply
    InterdepNet=load_percolation_model(supply_net)
    print "Nubmer of iterations=",len(supply_net.N)*3
    params={"NUM_ITERATIONS":len(supply_net.N)*3,"OUTPUT_DIR":'../results/percmodel_T'+`T`+'_N'+`len(supply_net.N)`+"_S"+`p_S`+"_C"+`p_C`+"_"+demand_distr,"ALGORITHM":'INDP',"V":1}
    if T > 1:
        params["NUM_ITERATIONS"]=1
        params["T"]=3000
        params["WINDOW_LENGTH"]=T
    else:
        params["NUM_ITERATIONS"]=len(supply_net.N)*3
        params["T"]=T
    params["N"]=InterdepNet
    params["MAGNITUDE"]=0
    params["SIM_NUMBER"]=0
    run_indp(params,layers=[0],T=params["T"],suffix="")
    
def run_indp_L3_V3(mags):
    for m in mags:
        params={"NUM_ITERATIONS":30,"OUTPUT_DIR":'../results/indp_results_L3',"MAGNITUDE":m,"V":3,"T":1,"ALGORITHM":"INDP"}
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
    
def run_infoshare_L2_V2(mags):
    for m in mags:
        params={"NUM_ITERATIONS":6,"OUTPUT_DIR":'../results/infoshare_results_L2_forced_',"MAGNITUDE":m,"V":1,"T":50,"WINDOW_LENGTH":5,"ALGORITHM":"INFO_SHARE"}
        batch_run(params,layers=[1,3])

def run_infoshare_L2_V2_inf(mags):
    params={"SIM_NUMBER":"INF","NUM_ITERATIONS":7,"OUTPUT_DIR":'../results/infoshare_results_L2_inf',"MAGNITUDE":0,"V":1,"T":212,"WINDOW_LENGTH":3,"ALGORITHM":"INFO_SHARE"}
    single_scenario_run(params,layers=[1,3])

def run_inrg_L2_V2(mags):
    for m in mags:
        params={"NUM_ITERATIONS":50,"OUTPUT_DIR":'../results/inrg_results_L2_3-1',"MAGNITUDE":m,"V":1,"T":1,"ALGORITHM":"INRG"}
        batch_run(params,layers=[1,3],player_ordering=[3,1])

def run_inrg_L2_V2_inf(mags):
    params={"SIM_NUMBER":"INF","NUM_ITERATIONS":212,"OUTPUT_DIR":'../results/inrg_results_L2_random_inf',"MAGNITUDE":0,"V":1,"T":1,"ALGORITHM":"INRG"}
    single_scenario_run(params,layers=[1,3],player_ordering="RANDOM",num_samples=100)

def run_backwards_induction_L2_V2(mags):
    for m in mags:
        params={"T":20,"OUTPUT_DIR":"../results/bi_results_L2_3-1_m"+`m`+"_v1","MAGNITUDE":m,"V":1,"ALGORITHM":"BACKWARDS_INDUCTION"}
        batch_run(params,layers=[1,3],player_ordering=[3,1])

def run_inrg_L2_V2_random_flip1(mags):
    num_flips=1
    # Max damage Mag 8=110
    # Max damage Mag 9=446
    for m in mags:
        params = {}
        if m == 8:
            params={"NUM_ITERATIONS":50,"OUTPUT_DIR":'../results/inrg_flip_'+`num_flips`+'_results_L2',"MAGNITUDE":m,"V":1,"T":1,"ALGORITHM":"INRG","SIM_NUMBER":110,"NUM_FLIPS":num_flips}
        else:
            params={"NUM_ITERATIONS":50,"OUTPUT_DIR":'../results/inrg_flip_'+`num_flips`+'_results_L2',"MAGNITUDE":m,"V":1,"T":1,"ALGORITHM":"INRG","SIM_NUMBER":446,"NUM_FLIPS":num_flips}
        one_flip_run(params,layers=[1,3],player_ordering=[3,1])

def run_infoshare_L2_V2_random_flip1(mags):
    num_flips=1
    # Max damage Mag 8=110
    # Max damage Mag 9=446
    for m in mags:
        params = {}
        if m == 8:
            params={"NUM_ITERATIONS":10,"OUTPUT_DIR":'../results/infoshare_flip_'+`num_flips`+'_results_L2',"MAGNITUDE":m,"V":1,"T":50,"WINDOW_LENGTH":3,"ALGORITHM":"INFO_SHARE","SIM_NUMBER":110,"NUM_FLIPS":num_flips}
        else:
            params={"NUM_ITERATIONS":10,"OUTPUT_DIR":'../results/infoshare_flip_'+`num_flips`+'_results_L2',"MAGNITUDE":m,"V":1,"T":50,"WINDOW_LENGTH":3,"ALGORITHM":"INFO_SHARE","SIM_NUMBER":446,"NUM_FLIPS":num_flips}
        one_flip_run(params,layers=[1,3])

def run_infoshare_L2_V2_random_flip(mags):
    prob=0.1
    # Max damage Mag 8=110
    # Max damage Mag 9=446 
    for m in mags:
        params = {}
        if m == 8:
            params={"NUM_ITERATIONS":10,"OUTPUT_DIR":'../results/infoshare_flip_'+`prob`+'_results_L2',"MAGNITUDE":m,"V":1,"T":50,"WINDOW_LENGTH":3,"ALGORITHM":"INFO_SHARE","SIM_NUMBER":110}
        else:
            params={"NUM_ITERATIONS":10,"OUTPUT_DIR":'../results/infoshare_flip_'+`prob`+'_results_L2',"MAGNITUDE":m,"V":1,"T":50,"WINDOW_LENGTH":3,"ALGORITHM":"INFO_SHARE","SIM_NUMBER":446}
        random_flip_run(params,layers=[1,3],flip_prob=prob)

def run_tdindp_L2_V2_seeded_flip(mags):
    for m in mags:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=0,magnitude=m,v=1)
        params={"N":InterdepNet,"NUM_ITERATIONS":1,"OUTPUT_DIR":'../results/tdindp_seededflip_results_L2',"MAGNITUDE":m,"V":2,"T":50,"WINDOW_LENGTH":3,"ALGORITHM":"INDP"}
        params["N"]=random_flip(params,[1,3],7463728,0.1)
        if m == 8:
            params["SIM_NUMBER"]=110
        elif m == 9:
            params["SIM_NUMBER"]=446
        single_scenario_run(params,layers=[1,3])

def run_infoshare_L2_V2_seeded_flip(mags):
    for m in mags:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=0,magnitude=m,v=1)
        params={"N":InterdepNet,"NUM_ITERATIONS":10,"OUTPUT_DIR":'../results/infoshare_seededflip_results_L2',"MAGNITUDE":m,"V":1,"T":50,"WINDOW_LENGTH":3,"ALGORITHM":"INFO_SHARE"}
        params["N"]=random_flip(params,[1,3],7463728,0.1)
        if m == 8:
            params["SIM_NUMBER"]=110
        elif m == 9:
            params["SIM_NUMBER"]=446
        single_scenario_run(params,layers=[1,3])

def run_inrg_L2_V2_random_ordering_seeded_flip(mags):
    for m in mags:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=0,magnitude=m,v=1)
        params={"N":InterdepNet,"NUM_ITERATIONS":50,"OUTPUT_DIR":'../results/inrg_randorder_seededflip_results_L2',"MAGNITUDE":m,"V":1,"T":1,"ALGORITHM":"INRG"}
        params["N"]=random_flip(params,[1,3],7463728,0.1)
        if m == 8:
            params["SIM_NUMBER"]=110
        elif m == 9:
            params["SIM_NUMBER"]=446
        single_scenario_run(params,layers=[1,3],player_ordering="RANDOM")

def main():
    """ Run as: python indpexp.py <algorithm=indp|tdindp|infoshare> <num_layers=1|2|3> <num_resources=1|2|3> <magnitude=6|8|9> """
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
    #run_inrg_sample()
    #run_indp_sample()
    run_percolation_model()
    #run_backwards_induction_L2_V2([6])
    #main()
