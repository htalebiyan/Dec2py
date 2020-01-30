from indp import *
from Dindp import *
#from plots import *
#import gametree
import os.path
import networkx as nx
import numpy as np
import pickle
import multiprocessing
import pandas as pd

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
    
    # Set root directories
    base_dir = failSce_param['Base_dir']
    damage_dir = failSce_param['Damage_dir'] 
    topology = None
    shelby_data = True
    ext_interdependency = None
    if failSce_param['type']=='Andres':
        ext_interdependency = "../data/INDP_4-12-2016"
    elif failSce_param['type']=='WU':
        if failSce_param['filtered_List']!=None:
            listHD = pd.read_csv(failSce_param['filtered_List'])
    elif failSce_param['type']=='random':
        pass
    elif failSce_param['type']=='synthetic':  
        shelby_data = False  
        topology = failSce_param['topology']
        
    for m in failSce_param['mags']:    
        for i in failSce_param['sample_range']:
            if failSce_param['filtered_List']==None or len(listHD.loc[(listHD.set == i) & (listHD.sce == m)].index):
                print '\n---Running Magnitude '+`m`+' sample '+`i`+'...'
            
                # print("Initializing network...")
                if not shelby_data:  
                    InterdepNet,noResource,layers=initialize_network(BASE_DIR=base_dir,external_interdependency_dir=ext_interdependency,magnitude=m,sample=i,shelby_data=shelby_data,topology=topology) 
                    params["V"]=noResource
                else:  
                    InterdepNet,_,_=initialize_network(BASE_DIR=base_dir,external_interdependency_dir=ext_interdependency,sim_number=0,magnitude=6,sample=0,v=params["V"],shelby_data=shelby_data)                    
                params["N"]=InterdepNet
                params["SIM_NUMBER"]=i
                params["MAGNITUDE"]=m
                output_dir_full=''				
                if params["ALGORITHM"]=="JUDGMENT_CALL" and params["AUCTION_TYPE"]:
                    output_dir_full = params["OUTPUT_DIR"]+'_L'+`len(layers)`+'_m'+`params["MAGNITUDE"]`+"_v"+`params["V"]`+'_auction_'+params["AUCTION_TYPE"]+'_'+params["VALUATION_TYPE"]+'/actions_'+`i`+'_sum.csv'
                elif params["ALGORITHM"]=="JUDGMENT_CALL" and not params["AUCTION_TYPE"]:
                    output_dir_full= params["OUTPUT_DIR"]+'_L'+`len(layers)`+'_m'+`params["MAGNITUDE"]`+"_v"+`params["V"]`+'_uniform_alloc/actions_'+`i`+'_sum.csv'
                else:	
                    output_dir_full=params["OUTPUT_DIR"]+'_L'+`len(layers)`+'_m'+`params["MAGNITUDE"]`+"_v"+`params["V"]`+'/actions_'+`i`+'_.csv'
                if os.path.exists(output_dir_full):
                    print 'results are already there\n'
                    continue
                if failSce_param['type']=='WU':
                    add_Wu_failure_scenario(InterdepNet,DAM_DIR=damage_dir,noSet=i,noSce=m)
                elif failSce_param['type']=='ANDRES':
                    add_failure_scenario(InterdepNet,DAM_DIR=damage_dir,magnitude=m,v=params["V"],sim_number=i)
                elif failSce_param['type']=='random':
                    add_random_failure_scenario(InterdepNet,DAM_DIR=damage_dir,sample=i)
                elif failSce_param['type']=='synthetic':
                    add_synthetic_failure_scenario(InterdepNet,DAM_DIR=base_dir,topology=topology,config=m,sample=i)            
                    
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
    auction_type = ["MCA","MAA","MDA"]
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
    
def run_indp_batch(failSce_param,v_r,layers,output_dir='../'):
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
        params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'results/indp_results',
                "V":v,"T":1,"ALGORITHM":"INDP"}
        batch_run(params,failSce_param,layers=layers)
        
def run_tdindp_batch(failSce_param,v_r,layers,output_dir='../'):
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
        params={"NUM_ITERATIONS":1,"OUTPUT_DIR":output_dir+'results/tdindp_results',
                "V":v,"T":10,"WINDOW_LENGTH":3,"ALGORITHM":"INDP"}
        batch_run(params,failSce_param,layers=layers)  
        
def run_dindp_batch(failSce_param,v_r,layers,judgment_type="OPTIMISTIC",auction_type=None,valuation_type='DTC',output_dir='../'):
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
        params={"NUM_ITERATIONS":10,
                "OUTPUT_DIR":output_dir+'results/judgeCall_'+judgment_type+'_results',
                "V":v,"T":1,"ALGORITHM":"JUDGMENT_CALL",
                "JUDGMENT_TYPE":judgment_type,"AUCTION_TYPE":auction_type,
                "VALUATION_TYPE":valuation_type}
        batch_run(params,failSce_param,layers=layers)
    
def run_parallel(i):
    ''' Decide the failure scenario (Andres or Wu) and network dataset (shelby or synthetic)
    Help:
    For Andres scenario: sample range: failSce_param["sample_range"], magnitudes: failSce_param['mags']
    For Wu scenario: set range: failSce_param["sample_range"], sce range: failSce_param['mags']
    For Synthetic nets: sample range: failSce_param["sample_range"], configurations: failSce_param['mags']  
    '''
    listFilteredSce = None
    base_dir = "/scratch/ht20/Extended_Shelby_County/"    #'../../data/Extended_Shelby_County/' #
    damage_dir = "/scratch/ht20/random_disruption_shelby/"   #'../../data/random_disruption_shelby/' #
    output_dir = '/scratch/ht20/' #'../../' #
    
    # failSce = read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/",magnitude=8)
    # failSce_param = {"type":"ANDRES","sample_range":range(1,1001),"mags":[6,7,8,9],
    #                  'Base_dir':base_dir,'Damage_dir':damage_dir}
    # failSce_param = {"type":"WU","sample_range":range(23,24),"mags":range(5,6),
    #                 'filtered_List':listFilteredSce,'Base_dir':base_dir,'Damage_dir':damage_dir}
    failSce_param = {"type":"random","sample_range":range(10,12),"mags":range(0,1),
                    'filtered_List':None,'Base_dir':base_dir,'Damage_dir':damage_dir}
    # failSce_param = {"type":"synthetic","sample_range":range(0,5),"mags":range(0,100),
    #                   'filtered_List':None,'topology':'Grid',
    #                   'Base_dir':base_dir,'Damage_dir':damage_dir}

    sampleNo = i
    magNo = 0
    
    # with open('output.txt', 'a') as f:
        # f.write('Sce: '+`magNo`+', Sample: '+`sampleNo`+'\n')
    # f.close
    failSce_param = {"type":"random","sample_range":range(sampleNo,sampleNo+1),
                     "mags":range(magNo,magNo+1),'filtered_List':listFilteredSce,
                     'Base_dir':base_dir,'Damage_dir':damage_dir}


    ''' Run different methods'''
    # No restriction on number of resources for each layer # Not necessary for synthetic nets
    v_r=[60,70,80,90,100]
#    v_r=[[1,1,1,1],[2,2,2,2],[3,3,3,3]]              # Prescribed number of resources for each layer
    judge_types = [""]    #["PESSIMISTIC","OPTIMISTIC","DEMAND","DET-DEMAND","RANDOM"]
    auction_types =  []      #["MDA","MAA","MCA"] 
    valuation_types = ['']       #['DTC','DTC_uniform','MDDN']    
    layers=[1,2,3,4] # List of layers of the net # Not necessary for synthetic nets


    run_indp_batch(failSce_param,v_r,layers,output_dir=output_dir)
   # run_tdindp_batch(failSce_param, v_r,layers,output_dir=output_dir)
    # for jc in judge_types:
    #     run_dindp_batch(failSce_param,v_r,layers,judgment_type=jc,auction_type=None,valuation_type=None,output_dir=output_dir)
    #     for at in auction_types:
    #         for vt in valuation_types:
    #             run_dindp_batch(failSce_param,v_r,layers,
    #                judgment_type=jc,auction_type=at,valuation_type=vt,output_dir=output_dir)
    return True
	
if __name__ == "__main__":  
#    plt.close('all')
    ''' Run a toy example for different methods '''
#    run_indp_sample()

    ''' Decide the failure scenario (Andres or Wu) and network dataset (shelby or synthetic)
    Help:
    For Andres scenario: sample range: failSce_param["sample_range"], magnitudes: failSce_param['mags']
    For Wu scenario: set range: failSce_param["sample_range"], sce range: failSce_param['mags']
    For Synthetic nets: sample range: failSce_param["sample_range"], configurations: failSce_param['mags']  
    '''
    listFilteredSce = None #'../data/damagedElements_sliceQuantile_0.95.csv'
    base_dir = "/scratch/ht20/Extended_Shelby_County/"    #'../../data/Extended_Shelby_County/' #
    damage_dir = "/scratch/ht20/random_disruption_shelby/"   #'../../data/random_disruption_shelby/' #
    output_dir = '/scratch/ht20/' #'../../' #
    
    # failSce = read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/",magnitude=8)
    # failSce_param = {"type":"ANDRES","sample_range":range(1,1001),"mags":[6,7,8,9],
    #                  'Base_dir':base_dir,'Damage_dir':damage_dir}
    # failSce_param = {"type":"WU","sample_range":range(23,24),"mags":range(5,6),
    #                 'filtered_List':listFilteredSce,'Base_dir':base_dir,'Damage_dir':damage_dir}
    failSce_param = {"type":"random","sample_range":range(50,500),"mags":range(0,1),
                    'filtered_List':None,'Base_dir':base_dir,'Damage_dir':damage_dir}
    # failSce_param = {"type":"synthetic","sample_range":range(0,5),"mags":range(0,100),
    #                   'filtered_List':None,'topology':'Grid',
    #                   'Base_dir':base_dir,'Damage_dir':damage_dir}



    ''' Run different methods'''
    # No restriction on number of resources for each layer # Not necessary for synthetic nets
    v_r=[60,70,80,90,100] 
#    v_r=[[1,1,1,1],[2,2,2,2],[3,3,3,3]]              # Prescribed number of resources for each layer
    judge_types = []    #["PESSIMISTIC","OPTIMISTIC","DEMAND","DET-DEMAND","RANDOM"]
    auction_types =  ['']      #["MDA","MAA","MCA"] 
    valuation_types = ['']       #['DTC','DTC_uniform','MDDN']    
    layers=[1,2,3,4] # List of layers of the net # Not necessary for synthetic nets
	
    num_cores = multiprocessing.cpu_count()
    print 'number of cores:'+`num_cores`+'\n'
    pool = multiprocessing.Pool(num_cores-1)  
    resuls1 = pool.map(run_parallel,failSce_param["sample_range"])

    ''' Compute metrics ''' 
    cost_type = 'Total'
    ref_method = 'indp'
    method_name = ['indp']
    for jc in judge_types:
        method_name.append('judgeCall_'+jc)
    # auction_types.append('Uniform')
    suffixes = ['Real_sum','']
    sample_range=failSce_param["sample_range"]
    mags=failSce_param['mags']
    
    synthetic_dir=None #base_dir+failSce_param['topology']+'Networks/'
    combinations,optimal_combinations=generate_combinations('shelby',mags,sample_range,
                layers,v_r,method_name,auction_types,valuation_types,listHDadd=listFilteredSce,synthetic_dir=synthetic_dir)
    
    root=output_dir+'results/' 
    df = read_and_aggregate_results(combinations,optimal_combinations,suffixes,root_result_dir=root)
##    df = correct_tdindp_results(df,optimal_combinations)
    lambda_df = relative_performance(df,combinations,optimal_combinations,ref_method=ref_method,cost_type=cost_type)
    resource_allocation,res_alloc_rel=read_resourcec_allocation(df,combinations,
                optimal_combinations,root_result_dir=root,ref_method=ref_method)   
    run_time_df = read_run_time(combinations,optimal_combinations,suffixes,root_result_dir=root)

    ''' Save Variables to file '''
    object_list = [combinations,optimal_combinations,df,method_name,lambda_df,resource_allocation,res_alloc_rel,cost_type,run_time_df]
    # Saving the objects:
    with open(output_dir+'objs.pkl', 'w') as f: 
        pickle.dump(object_list, f)

    # # Getting back the objects:
    # with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
        # obj0, obj1, obj2 = pickle.load(f)

    """ Plot results """    
#    plot_performance_curves_shelby(df,cost_type='Total',decision_names=method_name,ci=None,normalize=True)
#    plot_relative_performance_shelby(lambda_df)
#    plot_auction_allocation_shelby(resource_allocation,ci=None)
#    plot_relative_allocation_shelby(res_alloc_rel)
    
    
    # plot_performance_curves_synthetic(df,ci=None,x='t',y='cost',cost_type=cost_type)  
    # plot_performance_curves_synthetic(df,ci=None,x='t',y='cost',cost_type='Under Supply Perc')
    # plot_relative_performance_synthetic(lambda_df,cost_type=cost_type,lambda_type='U')  
    # plot_auction_allocation_synthetic(resource_allocation,ci=None,resource_type='normalized_resource')
    # plot_relative_allocation_synthetic(res_alloc_rel)
    # plot_run_time_synthetic(run_time_df,ci=None)