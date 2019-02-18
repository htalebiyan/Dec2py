import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from indp import *
import os.path

def run_judgment_call(params,layers=[1,2,3],T=1,saveJC=True,print_cmd=True,saveJCModel=False):
    """ Solves an INDP problem with specified parameters using a decentralized hueristic called Judgment Call . Outputs to directory specified in params['OUTPUT_DIR'].
    :param params: Global parameters.
    :param layers: Layers to consider in the infrastructure network.
    """
    # Initialize failure scenario.
    InterdepNet=None
    if "N" not in params:
        InterdepNet=initialize_network(BASE_DIR="../data/INDP_7-20-2015/",sim_number=params['SIM_NUMBER'],magnitude=params["MAGNITUDE"])
    else:
        InterdepNet=params["N"]
        
    if "NUM_ITERATIONS" not in params:
        params["NUM_ITERATIONS"]=1
 
    v_r = params["V"]
    num_Timesteps = params["NUM_ITERATIONS"]
    m = params["MAGNITUDE"]
    output_dir = params["OUTPUT_DIR"]+'_m'+`params["MAGNITUDE"]`+"_v"+`sum(v_r)`+'_Layer_Res_Cap'
    
    Dindp_results={l:[] for l in layers}    
    if T == 1:        
        for l in layers:
            # Initial calculations.
            indp_results=indp(InterdepNet,0,1,layers=layers,controlled_layers=[l])
            Dindp_results[l] = indp_results[1]
            Dindp_results[l].add_components(0,INDPComponents.calculate_components(indp_results[0],InterdepNet,layers=[l]))

        functionality = {0:{}}  
        for t in range(num_Timesteps):
            if print_cmd:
                print "Timestep ",t,"/",num_Timesteps
                
            for l in layers:
                if print_cmd:
                    print "Layer ",l
                    
                for u,v,a in InterdepNet.G.edges_iter(data=True):
                    if a['data']['inf_data'].is_interdep and v[1]==l:
                        if v[0]<0:
                            functionality[0][u] = 0.0
                        else:
                            functionality[0][u] = 1.0
                        
                indp_results = indp(InterdepNet,v_r[l-1],1,layers=layers,
                                    controlled_layers=[l],
                                    functionality=functionality)
                Dindp_results[l].extend(indp_results[1],t_offset=t+1)
                if saveJCModel:
                    save_INDP_model_to_file(indp_results[0],output_dir+"/Model",t,l)
                # Modify network to account for recovery and calculate components.
                apply_recovery(InterdepNet,Dindp_results[l],t+1)
                Dindp_results[l].add_components(t+1,INDPComponents.calculate_components(indp_results[0],InterdepNet,layers=[l]))
                
        Dindp_results_sum = INDPResults()
        cost_types = Dindp_results[1][0]['costs'].keys()
        for t in range(num_Timesteps+1):
            for cost_type in cost_types:
                sumTemp = 0.0
                for l in layers:
                     sumTemp += Dindp_results[l][t]['costs'][cost_type]
                Dindp_results_sum.add_cost(t,cost_type,sumTemp)

        # Save results of D-iINDP run.
        if saveJC:        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for l in layers:
                Dindp_results[l].to_csv(output_dir,params["SIM_NUMBER"],suffix=`l`)
            Dindp_results_sum.to_csv(output_dir,params["SIM_NUMBER"],suffix='sum')
    else:
        # td-INDP formulations. Includes "DELTA_T" parameter for sliding windows to increase
        # efficiency.
        # Edit 2/8/16: "Sliding window" now overlaps.
        print 'hahahaha'

def aggregate_results(mags,method_name,resource_cap,suffixes,L,v,sample_range):
    columns = ['t','Magnitude','cost_type','method','resource_cap','sample','cost']
    agg_results = pd.DataFrame(columns=columns)
    for m in mags:
        for i in range(len(method_name)):
            full_suffix = '_L'+`L`+'_m'+`m`+'_v'+`v`+resource_cap[i]
            result_dir = '../results/'+method_name[i]+full_suffix
            
            # Save average values to file
            results_average = INDPResults()
            results_average = results_average.from_results_dir(outdir=result_dir,
                                sample_range=sample_range,suffix=suffixes[i])
            
            outdir = '../results/average_cost_all/'
            costs_file =outdir+method_name[i]+full_suffix+"_average_costs.csv"
            with open(costs_file,'w') as f:
                f.write("t,Space Prep,Arc,Node,Over Supply,Under Supply,Flow,Total\n")
                for t in results_average.results:
                    costs=results_average.results[t]['costs']
                    f.write(`t`+","+`costs["Space Prep"]`+","+`costs["Arc"]`+","
                            +`costs["Node"]`+","+`costs["Over Supply"]`+","+
                            `costs["Under Supply"]`+","+`costs["Flow"]`+","+
                            `costs["Total"]`+"\n")            
                    
            # Save all results to Pandas dataframe
            sample_result = INDPResults()
            for s in sample_range:
                sample_result=sample_result.from_csv(result_dir,s,suffix=suffixes[i])
                for t in sample_result.results:
                    for c in sample_result.cost_types:
                        values = [t,m,c,method_name[i],resource_cap[i],s,
                                float(sample_result[t]['costs'][c])]
#                        df2 = pd.DataFrame(values, columns=columns)
                        agg_results = agg_results.append(dict(zip(columns,values)), ignore_index=True)
    
    return agg_results
    
def plot_results(df,x='t',y='cost',cost_type='Total'):
    sns.set()
    
    df['resource_cap'] = df['resource_cap'].replace('', 'Network Cap')
    df['resource_cap'] = df['resource_cap'].replace('_SepRes', 'Layer Cap')
    
    colors = ["windows blue", "amber", "red"]
    with sns.xkcd_palette(colors):
        ax = sns.lineplot(x=x, y=y, hue="method", style='resource_cap',
                          markers=True,
                      data=df[df['cost_type'] == cost_type])
        ax.set(xticks=np.arange(0,21,2))

    