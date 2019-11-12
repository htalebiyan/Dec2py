import networkx as nx
import os 
import pandas as pd
import matplotlib.pyplot as plt
'''
This function plots two interdependent networks in 3D
Pn,Pa,Ppos = List of nodes, List of arcs, and positions of nodes of network 1 (power network)
Gn,Ga,Gpos = List of nodes, List of arcs, and positions of nodes of network 2 (gas network)
intdpnPairs = set of interdependent nodes
Note: Here, the naming of nodes is different Plot.py and nodes in each network
is named from '0' on. Other details are the same so to understand this 
function look at Plot.py
'''  


def load_array_format_extended(BASE_DIR,topo='RN',config=0,sample=0,cost_scale=1.0):
    file_dir = BASE_DIR+topo+'Config_'+str(config)+'\\Sample_'+str(sample)+'\\'
    with open(BASE_DIR+'List_of_Configurations.txt') as f:
            data = pd.read_csv(f, delimiter='\t')    
    config_param = data.iloc[config]
    noLayers = int(config_param.loc[' No. Layers'])
    G=nx.Graph()
    
    files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
    pos = {}
    dam_nodes = []
    dam_arcs = []
    z_offsetx = 0.25
    z_offsety = 3
    for k in range(1,noLayers+1):
        for file in files: 
            if file=='N'+str(k)+'_Nodes.txt':
                with open(file_dir+file) as f:
    #                print "Opened",file,"."
                    data = pd.read_csv(f, delimiter='\t',header=None)
                    for v in data.iterrows():               
                        G.add_node((int(v[1][0]),k))
                        pos[(int(v[1][0]),k)] = (v[1][1]+v[1][3]*z_offsetx,v[1][2]+v[1][3]*z_offsety)
        for file in files:
            if file=='N'+str(k)+'_Arcs.txt':
                with open(file_dir+file) as f:
    #                print "Opened",file,"."
                    data = pd.read_csv(f, delimiter='\t',header=None)
                    for v in data.iterrows():   
                        G.add_edge((int(v[1][0]),k),(int(v[1][1]),k))
        for kt in range(noLayers):
            if k!=kt:
                for file in files:
                    if file=='Interdependent_Arcs_'+str(k)+'_'+str(kt)+'.txt':
                        with open(file_dir+file) as f:
                #                print "Opened",file,"."
                            data = pd.read_csv(f, delimiter='\t',header=None)
                            for v in data.iterrows():   
                                G.add_edge((int(v[1][0]),k),(int(v[1][1]),kt)) 
        for file in files: 
            if file=='N'+str(k)+'_Damaged_Nodes.txt':
                with open(file_dir+file) as f:
                    try:
                        data = pd.read_csv(f, delimiter='\t',header=None)
                        for v in data.iterrows():               
                            dam_nodes.append((int(v[1][0]),k))
                    except:
                        print('Empty file: '+ file)
                                
        for file in files:
            if file=='N'+str(k)+'_Damaged_Arcs.txt':
                with open(file_dir+file) as f:
                    try:
                        data = pd.read_csv(f, delimiter='\t',header=None)
                        for v in data.iterrows():   
                            dam_arcs.append(((int(v[1][0]),k),(int(v[1][1]),k)))
                    except:
                        print('Empty file: '+ file)
#    pos=nx.spring_layout(G)   
##    for key,value in pos.items():                     
##        pos[key][0] += key[1]*z_offsetx
##        pos[key][1] += key[1]*z_offsety   
    return G,pos,noLayers,dam_nodes,dam_arcs

plt.close('all')
plt.figure(figsize=(10,8))  
BASE_DIR="C:\\Users\ht20\Documents\\Files\\Generated_Network_Dataset_v3\\ScaleFreeNetworks\\"
G,pos,noLayers,dam_nodes,dam_arcs = load_array_format_extended(BASE_DIR,topo='SFN',config=75,sample=0)  
labels = {}
for n,d in G.nodes(data=True):
    labels[n]= "%d" % (n[0])
pos_moved={}
for key,value in pos.items():
    pos_moved[key] = [0,0]
    pos_moved[key][0] = pos[key][0]-0.2
    pos_moved[key][1] = pos[key][1]+0.2
#nx.draw(G, pos,node_color='w')
#nx.draw_networkx_labels(G,labels=labels,pos=pos,
#                        font_color='w',font_family='CMU Serif',font_weight='bold')

clr=['r','b','g','m']
for k in range(noLayers):
    node_list = [x for x in G.nodes() if x[1]==k+1]
    nx.draw_networkx_nodes(G,pos,nodelist=node_list,node_color=clr[k],node_size=70,alpha=0.9)
for k in range(noLayers):
    arc_dict = [x for x in G.edges() if x[0][1]==k+1 and x[1][1]==k+1]
    nx.draw_networkx_edges(G,pos,edgelist=arc_dict,width=1,alpha=0.25,edge_color=clr[k])
    interarc_dict = [x for x in G.edges() if x[0][1]==k+1 and x[1][1]!=k+1]
    nx.draw_networkx_edges(G,pos,edgelist=interarc_dict,width=1,alpha=0.25,edge_color='k')
#nx.draw_networkx_nodes(G,pos,nodelist=dam_nodes,node_color='w',node_shape="x",node_size=35)
#nx.draw_networkx_edges(G,pos,edgelist=dam_arcs,width=1,alpha=1,edge_color='w',style='dashed')
plt.tight_layout()   
plt.axis('off')
#plt.savefig(output_dir+'/plot_net'+folderSuffix+'.png',dpi=600)