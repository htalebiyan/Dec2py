import STAR_utils
import matplotlib.pyplot as plt 
import networkx as nx 

if __name__ == "__main__":  
    plt.close('all')
    ''' Decide the failure scenario (Andres or Wu) and network dataset (shelby or synthetic)
    Help:
    For Andres scenario: sample range: failSce_param["sample_range"], magnitudes: failSce_param['mags']
    For Wu scenario: set range: failSce_param["sample_range"], sce range: failSce_param['mags']
    For Synthetic nets: sample range: failSce_param["sample_range"], configurations: failSce_param['mags']  
    '''
    listFilteredSce = '../../data/damagedElements_sliceQuantile_0.95.csv'
    base_dir = "../../" #'C:/Users/ht20/Documents/Files/Generated_Network_Dataset_v3.1/'
    output_dir = 'C:/Users/ht20/Documents/Files/Auction_Extended_Shelby_County_Data/' #'C:/Users/ht20/Documents/Files/Auction_synthetic_networks_v3.1/'

#    failSce = read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/",magnitude=8)
#    failSce_param = {"type":"ANDRES","sample_range":range(1,1001),"mags":[6,7,8,9]}
    failSce_param = {"type":"WU","sample_range":range(0,50),"mags":range(0,96),
                     'filtered_List':listFilteredSce,'Base_dir':base_dir}
#    failSce_param = {"type":"synthetic","sample_range":range(0,5),"mags":range(0,100),
#                     'filtered_List':None,'topology':'Grid','Base_dir':base_dir}
    v = 3
    layers=[1,2,3,4]
    params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'results/indp_results',
            "V":v,"ALGORITHM":"INDP"}
    
    # samples,network_objects,initial_net,_,_=STAR_utils.importData(params,failSce_param,layers) 
    
    comp_graph = initial_net.G.copy()
    decomp_graph = initial_net.G.copy()
    for u,v,a in comp_graph.edges_iter(data=True):
        if a['data']['inf_data'].is_interdep:
            decomp_graph.remove_edge(u, v)
    graphs = list(nx.connected_component_subgraphs(decomp_graph.to_undirected())) 
    graphs=[graphs[0],graphs[1],graphs[2],graphs[4]]
    
    ''' repair time and repair probability '''
    T=10
    import pandas as pd 
    samplesDiffTime = {}
    samplesRepProb = {}
    for key, val in samples.items(): 
        if key not in samplesDiffTime.keys():
            samplesDiffTime[key]=[]
        if key not in samplesRepProb.keys():
            samplesRepProb[key]=[]
        for s in range(val.shape[1]):
            samplesDiffTime[key].append(0)
            for t in range(1,T+1):
                if val[t,s]==1:
                    if val[t-1,s] == 0:
                        samplesDiffTime[key][-1] = t
                        samplesRepProb[key].append(1.0)
                elif val[t,s]==0 and val[t-1,s] == 0:
                        samplesRepProb[key].append(0.0)
                elif val[t,s]==0 and t==T and samplesDiffTime[key][-1]==0: 
                    samplesDiffTime[key][-1] = 2*T #!!!assumption
    samplesDiffTimeMean = {}
    for key, val in samplesDiffTime.items():
        samplesDiffTimeMean[key]=sum(val)/float(len(val))
    feature_dict=pd.DataFrame.from_dict(samplesDiffTimeMean, orient='index', columns=['repair_time'])    
    samplesRepProbMean = {}
    for key, val in samplesRepProb.items():
        if len(val)==0:
            samplesRepProbMean[key]=1.0 #!!!assumption
        else:
            samplesRepProbMean[key]=sum(val)/float(len(val))
    temp=pd.DataFrame.from_dict(samplesRepProbMean, orient='index', columns=['repair_prob']) 
    feature_dict=pd.concat([feature_dict,temp],axis=1)
    
    ''' neighbor repair time '''
    neighborDiffTime = {}
    dependeeDiffTime = {}
    neighborRepProb = {}
    dependeeRepProb = {}
    for u,v,d in comp_graph.edges(data=True):
        if d['data']['inf_data'].is_interdep:
            if v not in dependeeDiffTime.keys():
                dependeeDiffTime[v]=[]
                dependeeRepProb[v]=[]
            dependeeDiffTime[v].append(samplesDiffTimeMean[u])
            dependeeRepProb[v].append(samplesRepProbMean[u])
        else:
            if v not in neighborDiffTime.keys():
                neighborDiffTime[v]=[]
                neighborRepProb[v]=[]
            if u not in neighborDiffTime.keys():
                neighborDiffTime[u]=[]
                neighborRepProb[u]=[]
            neighborDiffTime[v].append(samplesDiffTimeMean[u])
            neighborDiffTime[u].append(samplesDiffTimeMean[v]) 
            neighborRepProb[v].append(samplesRepProbMean[u])
            neighborRepProb[u].append(samplesRepProbMean[v])
    dependeeDiffTimeMean = {}
    dependeeRepProbMean = {}
    for key, val in samplesDiffTimeMean.items():
        if key in dependeeDiffTime.keys():
            val = dependeeDiffTime[key]
            dependeeDiffTimeMean[key]=sum(val)/float(len(val))
            val = dependeeRepProb[key]
            dependeeRepProbMean[key]=sum(val)/float(len(val))
        else:
            dependeeDiffTimeMean[key]= 0.0 #!!!assumption
            dependeeRepProbMean[key]= 1.0 #!!!assumption
    temp=pd.DataFrame.from_dict(dependeeDiffTimeMean, orient='index', columns=['dependee_repair_time'])    
    feature_dict=pd.concat([feature_dict,temp],axis=1)
    temp=pd.DataFrame.from_dict(dependeeRepProbMean, orient='index', columns=['dependee_repair_prob'])    
    feature_dict=pd.concat([feature_dict,temp],axis=1)
    
    neighborDiffTimeMean = {}
    neighborRepProbMean = {}
    for key, val in samplesDiffTimeMean.items():
        if key in neighborDiffTime.keys():
            val = neighborDiffTime[key]
            neighborDiffTimeMean[key]=sum(val)/float(len(val))
            val = neighborRepProb[key]
            neighborRepProbMean[key]=sum(val)/float(len(val))
        else:
            neighborDiffTimeMean[key]= 0.0   #!!!assumption
            neighborRepProbMean[key] = 1.0  #!!!assumption
    temp=pd.DataFrame.from_dict(neighborDiffTimeMean, orient='index', columns=['neighbor_repair_time'])     
    feature_dict=pd.concat([feature_dict,temp],axis=1)
    temp=pd.DataFrame.from_dict(neighborRepProbMean, orient='index', columns=['neighbor_repair_prob'])     
    feature_dict=pd.concat([feature_dict,temp],axis=1)
    ''' centrality '''        
    centrality_list=[nx.degree_centrality,nx.closeness_centrality,nx.betweenness_centrality,
                     nx.current_flow_closeness_centrality,nx.eigenvector_centrality_numpy,
                     nx.katz_centrality,nx.communicability_centrality,
                     nx.communicability_betweenness_centrality,nx.load_centrality]
    for cent_name in centrality_list:
        cent = {}
        for gr in graphs:
            if cent_name in [nx.current_flow_closeness_centrality]:
                cent.update(cent_name(gr.to_undirected())) 
            else:
                cent.update(cent_name(gr))   
        temp=pd.DataFrame.from_dict(cent, orient='index', columns=[cent_name.func_name])
        feature_dict=pd.concat([feature_dict,temp],axis=1)

    demand={}
    for gr in graphs:
        for n,d in gr.nodes_iter(data=True):
            value=d['data']['inf_data'].demand
            demand[n]=value         
    temp=pd.DataFrame.from_dict(demand, orient='index', columns=['demand'])
    feature_dict=pd.concat([feature_dict,temp],axis=1)

    feature_dict=feature_dict.dropna()
    feature_dict.drop(feature_dict[feature_dict['repair_time']==0].index , inplace=True)
    
    import seaborn as sns
    sns.set(style="white")
    g = sns.jointplot(x='communicability_centrality',y='communicability_betweenness_centrality',data=feature_dict,kind="reg",height=7,space=0)
#    g = sns.pairplot(feature_dict, kind="reg")
    
    sns.clustermap(feature_dict.corr(), center=0, cmap="vlag",linewidths=.75, figsize=(13, 13))