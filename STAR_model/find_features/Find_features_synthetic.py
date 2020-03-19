import STAR_utils
import matplotlib.pyplot as plt 
import networkx as nx 

if __name__ == "__main__":  
    plt.close('all')

    base_dir = "C:/Users/ht20/Documents/Files/Generated_Network_Dataset_v3.1" #'C:/Users/ht20/Documents/Files/Generated_Network_Dataset_v3.1/'
    output_dir = 'C:/Users/ht20/Documents/Files/Auction_synthetic_networks_v3.1' #'C:/Users/ht20/Documents/Files/Auction_synthetic_networks_v3.1/'
    sce_range =range(0,100)
    sam_range=range(0,5)
    topo = 'Grid' #'Grid'|'ScaleFree'|'Random'
    feature_dict = {}
    for sce in sce_range:
        for sam in sam_range:
            v = 3
            layers=[1,2,3,4]

            failSce_param = {"type":"synthetic","sample_range":range(sam,sam+1),"mags":range(sce,sce+1),
                            'filtered_List':None,'topology':topo,'Base_dir':base_dir}            
            params={"NUM_ITERATIONS":10,"OUTPUT_DIR":output_dir+'/'+failSce_param['topology']+'/results/indp_results',
                    "V":v,"ALGORITHM":"INDP"}
            
            samples,network_objects,initial_net,v,layers=STAR_utils.importData(params,failSce_param,layers)   
            
            comp_graph = initial_net.G.copy()
            decomp_graph = initial_net.G.copy()
            for u,v,a in comp_graph.edges_iter(data=True):
                if a['data']['inf_data'].is_interdep:
                    decomp_graph.remove_edge(u, v)
            graphs = list(nx.connected_component_subgraphs(decomp_graph.to_undirected())) 
            if len(graphs)!=len(layers):
                print '****disconnected graph\n'
            
            ''' repair time '''
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
            feature_dict[sce,sam]=pd.DataFrame.from_dict(samplesDiffTime, orient='index', columns=['repair_time'])  
            
            samplesRepProbMean = {}
            for key, val in samplesRepProb.items():
                if len(val)==0:
                    samplesRepProbMean[key]=1.0 #!!!assumption
                else:
                    samplesRepProbMean[key]=sum(val)/float(len(val))
            temp=pd.DataFrame.from_dict(samplesRepProbMean, orient='index', columns=['repair_prob']) 
            feature_dict[sce,sam]=pd.concat([feature_dict[sce,sam],temp],axis=1)  
            
            ''' neighbor and dependee repair time '''
            neighborDiffTime = {}
            dependeeDiffTime = {}
            neighborRepProb = {}
            dependeeRepProb = {}
            for u,v,d in comp_graph.edges(data=True):
                if d['data']['inf_data'].is_interdep:
                    if v not in dependeeDiffTime.keys():
                        dependeeDiffTime[v]=[]
                        dependeeRepProb[v]=[]
                    dependeeDiffTime[v].append(samplesDiffTime[u][0])
                    dependeeRepProb[v].append(samplesRepProbMean[u])
                else:
                    if v not in neighborDiffTime.keys():
                        neighborDiffTime[v]=[]
                        neighborRepProb[v]=[]
                    if u not in neighborDiffTime.keys():
                        neighborDiffTime[u]=[]
                        neighborRepProb[u]=[]
                    neighborDiffTime[v].append(samplesDiffTime[u][0])
                    neighborDiffTime[u].append(samplesDiffTime[v][0]) 
                    neighborRepProb[v].append(samplesRepProbMean[u])
                    neighborRepProb[u].append(samplesRepProbMean[v])
            dependeeDiffTimeMean = {}
            dependeeRepProbMean = {}
            for key, val in samplesDiffTime.items():
                if key in dependeeDiffTime.keys():
                    val = dependeeDiffTime[key]
                    dependeeDiffTimeMean[key]=sum(val)/float(len(val))
                    val = dependeeRepProb[key]
                    dependeeRepProbMean[key]=sum(val)/float(len(val))
                else:
                    dependeeDiffTimeMean[key]=0.0 #!!!assumption   
                    dependeeRepProbMean[key]= 1.0 #!!!assumption
            temp=pd.DataFrame.from_dict(dependeeDiffTimeMean, orient='index', columns=['dependee_repair_time'])    
            feature_dict[sce,sam]=pd.concat([feature_dict[sce,sam],temp],axis=1)
            temp=pd.DataFrame.from_dict(dependeeRepProbMean, orient='index', columns=['dependee_repair_prob'])    
            feature_dict[sce,sam]=pd.concat([feature_dict[sce,sam],temp],axis=1)
            
            neighborDiffTimeMean = {}
            neighborRepProbMean = {}
            for key, val in samplesDiffTime.items():
                if key in neighborDiffTime.keys():
                    val = neighborDiffTime[key]
                    neighborDiffTimeMean[key]=sum(val)/float(len(val))
                    val = neighborRepProb[key]
                    neighborRepProbMean[key]=sum(val)/float(len(val))
                else:
                    neighborDiffTimeMean[key]=0.0 #!!!assumption
                    neighborRepProbMean[key] = 1.0  #!!!assumption
            temp=pd.DataFrame.from_dict(neighborDiffTimeMean, orient='index', columns=['neighbor_repair_time'])     
            feature_dict[sce,sam]=pd.concat([feature_dict[sce,sam],temp],axis=1)
            temp=pd.DataFrame.from_dict(neighborRepProbMean, orient='index', columns=['neighbor_repair_prob'])     
            feature_dict[sce,sam]=pd.concat([feature_dict[sce,sam],temp],axis=1)
            
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
                feature_dict[sce,sam]=pd.concat([feature_dict[sce,sam],temp],axis=1)
                
            ''' Demand '''   
            demand={}
            for gr in graphs:
                for n,d in gr.nodes_iter(data=True):
                    value=d['data']['inf_data'].demand
                    demand[n]=abs(value)
            temp=pd.DataFrame.from_dict(demand, orient='index', columns=['demand'])
            feature_dict[sce,sam]=pd.concat([feature_dict[sce,sam],temp],axis=1)
            
    feature_df = pd.DataFrame(columns=list(feature_dict[0,0].columns.values))
    for sce in sce_range:
        for sam in sam_range:  
            feature_df=feature_df.append(feature_dict[sce,sam],ignore_index=True)
            
    # feature_df=feature_dict.dropna()
    feature_df_backup = feature_df.copy()
    feature_df["repair_time"] = pd.to_numeric(feature_df["repair_time"])
    feature_df.drop(feature_df[feature_df['repair_time']==0].index , inplace=True)
    import seaborn as sns
    sns.set(style="white")
    g = sns.jointplot(x='repair_prob',y='repair_time',data=feature_df,
                      kind="reg",height=7,space=0)
#    g = sns.pairplot(feature_dict, kind="reg")
    correlation_feature = feature_df.corr()
    sns.clustermap(correlation_feature, center=0, cmap="vlag",linewidths=.75, figsize=(13, 13))