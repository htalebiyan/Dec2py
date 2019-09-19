import sys
import random
from indp import *
from Dindp import *
import gametree
import os.path
import networkx as nx
import numpy as np

def batch_run(params,failSce_param,layers,player_ordering=[3,1]):
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

                
#    if "N" not in params:
#        print("Initializing network...")
#        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",external_interdependency_dir=None,sim_number=0,magnitude=6,v=params["V"]) #"../data/INDP_4-12-2016"
#        print(InterdepNet)
#    else:
#        InterdepNet=params["N"] 
                            
    # Use fake paramter name for the case of Wu's scenarios to fit to Andres's scenarios' nomenclature
    if failSce_param['type']=='WU':
        failSce_param['sample_range']=failSce_param['set_range']
        failSce_param['mags']=failSce_param['sce_range']
        if failSce_param['filtered_List']!=None:
            listHD = pd.read_csv(failSce_param['filtered_List'])
            
    for m in failSce_param['mags']:    
        for i in failSce_param['sample_range']:
            if failSce_param['filtered_List']==None or len(listHD.loc[(listHD.set == i) & (listHD.sce == m)].index):
            
                print '\n---Running Magnitude '+`m`+' sample '+`i`+'...'
                
                print("Initializing network...")
                InterdepNet=initialize_network(BASE_DIR="../data/Extended_Shelby_County/",external_interdependency_dir=None,sim_number=0,magnitude=6,v=params["V"])  # #"../data/INDP_7-20-2015/" "../data/INDP_4-12-2016"
                    
                params["N"]=InterdepNet
                params["SIM_NUMBER"]=i
                params["MAGNITUDE"]=m
                
                if failSce_param['type']=='WU':
                    add_Wu_failure_scenario(InterdepNet,BASE_DIR="../data/Wu_Scenarios/",noSet=i,noSce=m,noNet=3)
                elif failSce_param['type']=='ANDRES':
                    add_failure_scenario(InterdepNet,BASE_DIR="../data/INDP_7-20-2015/",magnitude=m,v=params["V"],sim_number=i)
                
                if params["ALGORITHM"]=="INDP":
                    run_indp(params,validate=False,T=params["T"],layers=layers,controlled_layers=layers,saveModel=False,print_cmd_line=False)
                elif params["ALGORITHM"]=="INFO_SHARE":
                    run_info_share(params,layers=layers,T=params["T"])
                elif params["ALGORITHM"]=="INRG":
                    run_inrg(params,layers=layers,player_ordering=player_ordering)
                elif params["ALGORITHM"]=="BACKWARDS_INDUCTION":
                    gametree.run_backwards_induction(InterdepNet,i,players=layers,player_ordering=player_ordering,T=params["T"],outdir=params["OUTPUT_DIR"])
                elif params["ALGORITHM"]=="JUDGMENT_CALL":
                    run_judgment_call(params,layers=layers,T=params["T"],saveJCModel=False,print_cmd=False)

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
    import warnings
    warnings.filterwarnings("ignore")
    InterdepNet=initialize_sample_network()
    params={"NUM_ITERATIONS":7, "OUTPUT_DIR":'../results/sample_indp_12Node_results_L2',"V":2,"T":1,"WINDOW_LENGTH":1,"ALGORITHM":"INDP"}
    params["N"]=InterdepNet
    params["MAGNITUDE"]=0
    params["SIM_NUMBER"]=0
#    run_indp(params,layers=[1,2],T=params["T"],suffix="",saveModel=True,print_cmd_line=True)
##    plot_indp_sample(params)
    auction_type = ["MCA","MDAA","MDDA"]
    valuation_type = ["DTC","MDDN"]
#    for at,vt in itertools.product(auction_type,valuation_type):
#        for jc in ["PESSIMISTIC","OPTIMISTIC"]:
#            InterdepNet=initialize_sample_network()
#            params["N"]=InterdepNet
#            params["NUM_ITERATIONS"]=7
#            params["ALGORITHM"]="JUDGMENT_CALL"
#            params["JUDGMENT_TYPE"]=jc
#            params["OUTPUT_DIR"]='../results/sample_judgeCall_12Node_'+jc+'_results_L2'
#            params["V"]=[1,1]
#            params["T"]=1
#            params["AUCTION_TYPE"]= at
#            params["VALUATION_TYPE"]= vt
#            run_judgment_call(params,layers=[1,2],T=params["T"],saveJCModel=True,print_cmd=True)
##            plot_indp_sample(params,
##                folderSuffix='_auction_'+params["AUCTION_TYPE"]+'_'+params["VALUATION_TYPE"],suffix="sum")
    
    method_name=['sample_indp_12Node','sample_judgeCall_12Node_PESSIMISTIC']
    suffixes=['','Real_sum']
    
    df = read_and_aggregate_results(mags=[0],method_name=method_name,
                    auction_type=auction_type, valuation_type=valuation_type,
                    suffixes=suffixes,L=2,sample_range=[0],no_resources=[2]) 
    lambda_df = relative_performance(df,sample_range=[0],ref_method='sample_indp_12Node')
    resource_allocation=read_resourcec_allocation(df,sample_range=[0],L=2,T=5,
                                    layers=[1,2],ref_method='sample_indp_12Node')

    plot_performance_curves(df,cost_type='Total',decision_names=method_name,
                            auction_type=auction_type, valuation_type=valuation_type,ci=None)
    plot_relative_performance(lambda_df)   
    plot_auction_allocation(resource_allocation,ci=None)
    plot_relative_allocation(resource_allocation)
    
def run_indp_L3(failSce_param,v_r):
    """
    This function runs iterativ indp for different numbers of resources
    
    Args:
        failSce_param (dict): informaton of damage scenrios
        v_r (list, float or list of float): number of resources, 
        if this is a list of floats, each float is interpreted as a different total number of resources, and indp is run given the total number of resources. 
        If this is a list of lists of floats, each list is interpreted as fixed upper bounds on the number of resources each layer can use (same for all time step).
        
    Returns:
    """
    for v in v_r:
        params={"NUM_ITERATIONS":10,"OUTPUT_DIR":'../results/indp_results_L3',"V":v,"T":1,"ALGORITHM":"INDP"}
        batch_run(params,failSce_param,layers=[1,2,3])
        
def run_tdindp_L3(failSce_param,v_r):
    """
    This function runs time-dependent indp for different numbers of resources
    Args:
        failSce_param (dict): informaton of damage scenrios
        v_r (list, float or list of float): number of resources, 
        if this is a list of floats, each float is interpreted as a different total number of resources, and indp is run given the total number of resources. 
        If this is a list of lists of floats, each list is interpreted as fixed upper bounds on the number of resources each layer can use (same for all time step).
        
    Returns:
    """
    for v in v_r:
        params={"NUM_ITERATIONS":1,"OUTPUT_DIR":'../results/tdindp_results_L3',"V":v,"T":10,"WINDOW_LENGTH":3,"ALGORITHM":"INDP"}
        batch_run(params,failSce_param,layers=[1,2,3])  
        
def run_dindp_L3(failSce_param,v_r,judgment_type="OPTIMISTIC",auction_type=None,valuation_type='DTC'):
    """
    This function runs Judfment Call Method for different numbers of resources, and a given judge, auction, and valuation type
    Args:
        failSce_param (dict): informaton of damage scenrios
        v_r (list, float or list of float): number of resources, 
        if this is a list of floats, each float is interpreted as a different total number of resources, and indp is run given the total number of resources. It only works when auction_type!=None.
        If this is a list of lists of floats, each list is interpreted as fixed upper bounds on the number of resources each layer can use (same for all time step). 
        judgment_type (str): Type of Judgments in Judfment Call Method. 
        auction_type (str): Type of auction for resource allocation. If None, fixed number of resources is allocated based on v_r, which MUST be a list of float when auction_type==None.
        valuation_type (str): Type of valuation in auction.
    Returns:
    """
    for v in v_r:
        params={"NUM_ITERATIONS":10,"OUTPUT_DIR":'../results/judgeCall_'+judgment_type+'_results_L3',
                "V":[v],"T":1,"ALGORITHM":"JUDGMENT_CALL",
                "JUDGMENT_TYPE":judgment_type,"AUCTION_TYPE":auction_type,
                "VALUATION_TYPE":valuation_type}
        batch_run(params,failSce_param,layers=[1,2,3])
    
                
if __name__ == "__main__":  
    plt.close('all')
    
    ''' Run a toy example for different methods '''
#    run_indp_sample()

    ''' Decide the failure scenario'''
    listFilteredSce = 'damagedElements_sliceQuantile_0.95.csv'
    failSce_param = {"type":"WU","set_range":range(24,25),"sce_range":range(5,6),
                     'filtered_List':listFilteredSce}
#    failSce_param = {"type":"WU","set_range":range(1,51),"sce_range":range(0,96),
#                     'filtered_List':listFilteredSce}
#    failSce_param = {"type":"ANDRES","sample_range":range(1,1001),"mags":[6,7,8,9]}
#    failSce = read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/",magnitude=8)

    ''' Run different methods'''
    v_r=[12] #[3,6,8,12]  # No restriction on nuber of resources for each layer
#    v_r=[[1,1,1],[2,2,2],[3,3,3]] # Prescribed number of resources for each layer
    judge_types = ["OPTIMISTIC"] #["PESSIMISTIC","OPTIMISTIC","DEMAND","DET-DEMAND","RANDOM"]
    auction_types =  ["MDD","MDA","MCA"] #["MDDA","MDAA","MCA"] #!!!
    valuation_types = ['DTC'] #['DTC','DTC_uniform','MDDN']    
    
    run_indp_L3(failSce_param,v_r)
#    run_tdindp_L3(failSce_param, v_r)
#    for jc in judge_types:
##        run_dindp_L3_V3(failSce_param,judgment_type=jc,auction_type=None)
#        for at in auction_types:
#            for vt in valuation_types:
#                run_dindp_L3(failSce_param,v_r=v_r, judgment_type=jc,
#                                auction_type=at,valuation_type=vt)


 
  
    """ Compute metrics """ 
    method_name = ['indp']
    for jc in judge_types:
        method_name.append('judgeCall_'+jc)
    suffixes = ['Real_sum','']
    sample_range=failSce_param["set_range"]
    mags=failSce_param['sce_range']
    
#    df = read_and_aggregate_results(mags,method_name,auction_types,valuation_types,suffixes,L=3,
#                                    sample_range=sample_range,no_resources=v_r,
#                                    listHDadd=failSce_param['filtered_List'])
###    df = correct_tdindp_results(df,mags,method_name,sample_range)
#   
#    lambda_df = relative_performance(df,sample_range=sample_range,ref_method='indp',
#                                     listHDadd=failSce_param['filtered_List'])
#    resource_allocation=read_resourcec_allocation(df,sample_range=sample_range,
#                            T=10,layers=[1,2,3],ci=None,
#                            listHDadd=failSce_param['filtered_List'])    

    
    """ Plot results """    
#    plot_performance_curves(df,cost_type='Total',decision_names=method_name,ci=None)
#    plot_relative_performance(lambda_df)
#    plot_auction_allocation(resource_allocation,ci=None)
#    plot_relative_allocation(resource_allocation)