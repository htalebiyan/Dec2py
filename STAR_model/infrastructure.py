import networkx as nx
import indputils
import re
import string
import numpy as np
import os
import sys
import pandas as pd

class InfrastructureNode(object):
    def __init__(self,id,net_id,local_id=""):
        self.id=id
        self.net_id=net_id
        if local_id == "":
            self.local_id=id
        self.local_id=local_id
        self.failure_probability=0.0
        self.functionality=1.0
        self.repaired=1.0
        self.reconstruction_cost=0.0
        self.oversupply_penalty=0.0
        self.undersupply_penalty=0.0
        self.demand=0.0
        self.space=0
        self.resource_usage=0
    def set_failure_probability(self,failure_probability):
        self.failure_probability=failure_probability
    def in_space(self,space_id):
        if self.space ==space_id:
            return 1
        else:
            return 0
    
class InfrastructureArc(object):
    def __init__(self,source,dest,layer,is_interdep=False):
        self.source=source
        self.dest=dest
        self.layer=layer
        self.failure_probability=0.0
        self.functionality=1.0
        self.repaired=1.0
        self.flow_cost=0.0
        self.reconstruction_cost=0.0
        self.resource_usage=0.0
        self.capacity = 0.0
        self.space=0
        self.is_interdep=is_interdep
    def in_space(self,space_id):
        if self.space == space_id:
            return 1
        else:
            return 0

class InfrastructureInterdepArc(InfrastructureArc):
    def __init__(self,source,dest,source_layer,dest_layer,gamma):
        super(InfrastructureInterdepArc,self).__init__(source,dest,source_layer,True)
        self.source_layer=source_layer
        self.dest_layer=dest_layer
        self.gamma=gamma

class InfrastructureSpace(object):
    def __init__(self,id,cost):
        self.id=id
        self.cost=cost

class InfrastructureNetwork(object):
    '''
    Stores information of the infrastructure network
    '''
    def __init__(self,id):
        self.G=nx.DiGraph()
        self.S=[]
        self.id=id
    def copy(self):
        newNet=InfrastructureNetwork(self.id)
        newNet.G=self.G.copy()
        newNet.S=[s for s in self.S]
        return newNet
    def update_with_strategy(self,player_strategy):
        for q in player_strategy[0]:
            node=q
            strat=player_strategy[0][q]
            self.G.node[q]['data']['inf_data'].repaired=round(strat['repair'])
            self.G.node[q]['data']['inf_data'].functionality=round(strat['w'])
        for q in player_strategy[1]:
            src=q[0]
            dst=q[1]
            strat = player_strategy[1][q]
            self.G[src][dst]['data']['inf_data'].repaired=round(strat['repair'])
            self.G[src][dst]['data']['inf_data'].functionality=round(strat['y'])
    def get_clusters(self,layer):
        G_prime_nodes=[n[0] for n in self.G.nodes(data=True) if n[1]['data']['inf_data'].net_id == layer and n[1]['data']['inf_data'].functionality==1.0]
        G_prime=nx.DiGraph(self.G.subgraph(G_prime_nodes).copy())
        G_prime.remove_edges_from([(u,v) for u,v,a in G_prime.edges(data=True) if a['data']['inf_data'].functionality==0.0])
        #print nx.connected_components(G_prime.to_undirected())
        return list(nx.connected_components(G_prime.to_undirected()))
    def gc_size(self,layer):
        G_prime_nodes=[n[0] for n in self.G.nodes(data=True) if n[1]['data']['inf_data'].net_id == layer and n[1]['data']['inf_data'].functionality==1.0]
        G_prime=nx.Graph(self.G.subgraph(G_prime_nodes))
        G_prime.remove_edges_from([(u,v) for u,v,a in G_prime.edges(data=True) if a['data']['inf_data'].functionality==0.0])
        cc=nx.connected_components(G_prime.to_undirected())
        if cc:
            #if len(list(cc)) == 1:
            #    print "I'm here"
            #    return len(list(cc)[0])
            #cc_list=list(cc)
            #print "Length",len(cc_list)
            #if len(cc_list) == 1:
            #    return len(cc_list[0])
            return len(max(cc, key=len))
        else:
            return 0
    def to_gamefile(self,layers=[1,3]):
        """ Used for multidefender security games."""
        with open("../results/indp_gamefile.game") as f:
            num_players=len(layers)
            num_targets=len([n for n in self.G.nodes(data=True) if n[1]['data']['inf_data'].net_id in layers])
            f.write(str(num_players)+","+str(num_targets)+",2\n")
            for layer in range(len(layers)):
                layer_nodes=[n for n in self.G.nodes(data=True) if n[1]['data']['inf_data'].net_id==layers[layer]]
                for node in layer_nodes:
                    def_values=[0.0]*len(layers)
                    def_values[layer]=abs(node[1]['data']['inf_data'].demand)
                    atk_values=sum(def_values)
                    f.write(str(layer)+"\n")
                    # f.write("0.0,1.0")
                    for v in def_values:
                        f.write(str(v)+"\n")
                    f.write(str(atk_values)+"\n")
                    
    def to_csv(self,layers=[1,3],filename="infrastructure_adj.csv"):
        with open(filename,'w') as f:
            for u,v,a in self.G.edges(data=True):
                f.write(str(u[0])+"."+str(u[1])+","+str(v[0])+"."+str(v[1])+"\n")
                            

def load_sample():
    G=InfrastructureNetwork("Sample")
    node_dict={1: {1: [3,0.0], 2: [-1,0.0], 3: [4,1.0], 4: [-6,0.0]}, 
               2: {5: [4,1.0], 6: [-6,0.0], 7: [3,0.0], 8: [-1,0.0]}}
    arc_dict= {1: [(1,2),(1,4),(3,4)], 2: [(5,6),(7,6),(7,8)]}
    for key,nodes in node_dict.iteritems():
        for node,vals in nodes.iteritems():
            n=InfrastructureNode(node,key,node)
            n.functionality=vals[1]
            n.repaired=vals[1]
            n.reconstruction_cost=abs(vals[0])
            n.oversupply_penalty=100
            n.undersupply_penalty=100
            n.demand=vals[0]
            n.space=((node-1) % 4)+1
            n.resource_usage=1 
            G.G.add_node((node,key), data={'inf_data':n})
    for key,arcs in arc_dict.iteritems():
        for arc in arcs:
            a=InfrastructureArc(arc[0],arc[1],key)
            a.flow_cost=1.0
            a.capacity=20.0
            a.space=1
            G.G.add_edge((arc[0],key),(arc[1],key),data={'inf_data':a})
    a1=InfrastructureInterdepArc(5,1,2,1,1.0)
    G.G.add_edge((5,2),(1,1),data={'inf_data':a1})
    #a2=InfrastructureInterdepArc(8,4,2,1,1.0)
    #G.G.add_edge((8,2),(4,1),data={'inf_data':a2})
    #a3=InfrastructureInterdepArc(2,6,1,2,1.0)
    #G.G.add_edge((2,1),(6,2),data={'inf_data':a3})
    a2=InfrastructureInterdepArc(4,8,1,2,1.0)
    G.G.add_edge((4,1),(8,2),data={'inf_data':a2})
    a3=InfrastructureInterdepArc(6,2,2,1,1.0)
    G.G.add_edge((6,2),(2,1),data={'inf_data':a3})
    a4=InfrastructureInterdepArc(3,7,1,2,1.0)
    G.G.add_edge((3,1),(7,2),data={'inf_data':a4})
    G.S.append(InfrastructureSpace(1,0.0))
    G.S.append(InfrastructureSpace(2,0.0))
    G.S.append(InfrastructureSpace(3,0.0))
    G.S.append(InfrastructureSpace(4,0.0))
    return G

def load_percolation_model(supply_net):
    G=InfrastructureNetwork("PercolationModel")
    for x,d in supply_net.N.nodes(data=True):
        n=InfrastructureNode(x,0,x)
        n.functionality=1.0
        n.repaired=1.0
        n.reconstruction_cost=abs(d['data'].supply)
        n.oversupply_penalty=100
        n.undersupply_penalty=100
        n.demand=d['data'].supply
        n.space=1
        n.resource_usage=1
        G.G.add_node((x,0), data={'inf_data':n})
    for u,v,x in supply_net.N_c.edges(data=True):
        a=InfrastructureArc(u,v,0)
        a.flow_cost=0.0
        a.capacity=100000.0
        a.space=1
        a.functionality=0.0
        a.repaired=0.0
        a.resource_usage=1.0
        a.reconstruction_cost=0.0
        G.G.add_edge((u,0),(v,0),data={'inf_data':a})
        G.G.add_edge((v,0),(u,0),data={'inf_data':a})
    return G

def load_infrastructure_data(BASE_DIR="../data/INDP_7-20-2015/", external_interdependency_dir=None,
                             magnitude=6, v=3, sim_number=1, cost_scale=1.0, shelby_data='shelby_extended'):
    if shelby_data == 'shelby_old':
#        print "Loading a network.." #!!!
        G = load_infrastructure_array_format(BASE_DIR=BASE_DIR,external_interdependency_dir=external_interdependency_dir,magnitude=magnitude,v=v,sim_number=sim_number,cost_scale=cost_scale)
#        print G #!!!
        return G
    elif shelby_data == 'shelby_extended':
#        print "Loading a network.." #!!!
        G = load_infrastructure_array_format_extended(BASE_DIR=BASE_DIR,v=v,sim_number=sim_number,cost_scale=cost_scale)
#        print G #!!!
        return G        
    else:
        sys.exit('Error: no interdepndent network is found')
    #elif BASE_DIR == "../data/INDP_4-12-2016":
    #    load_infrastructure_csv_format(BASE_DIR=BASE_DIR,external_interdependency_dir=external_interdependency_dir,magnitude=magnitude,v=v,sim_number=sim_number,cost_scale=cost_scale)

def load_interdependencies(netnumber,src_netnumber,BASE_DIR="../data/INDP_4-12-2016"):
    # Variables in CSV format is broken into files:
	# S<netnumber>_adj : Adjacency matrix for <netnumber>
	# S<netnumber>_bal : demand/supply of node.
	# S<netnumber>_cst : flow cost ?
	# S<netnumber>_dset : Set of distributors (but this can be calculated with *_bal)
	# S<netnumber>_gset : Set of generators (but this can be calculated with *_bal)
	# S<netnumber>_ub  : Upper bound on arc capacity.
	# S<netnumber>_lb  : Lower bound on arc capacity (typically 0).
	# S<netnumber>_imt : Interdependency matrix for network <netnumber>. <=== This method will only load interdependencies.
    arcs=[]
    with open(BASE_DIR+"/csv/S"+str(netnumber)+"_imt.csv") as f:
        rows=[[int(y) for y in string.split(x,",")] for x in f.readlines()]
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                if rows[i][j] > 0:
                    arc=InfrastructureInterdepArc(i+1,j+1,src_netnumber,netnumber,rows[i][j])
                    arcs.append(arc)
    return arcs
                    
def load_infrastructure_array_format(BASE_DIR="../data/INDP_7-20-2015/",external_interdependency_dir=None,magnitude=6,v=3,sim_number=1,cost_scale=1.0):
    variables=['v','c','f','q','Mp','Mm','g','b','u','h','p','gamma','alpha','beta','a','n']
    entry_regex=         r"\(\s*(?P<tuple>(\d+\s*,\s*)*\d+)\s*\)\s*(?P<val>-?\d+)"
    recovery_entry_regex=r"\(\s*(?P<tuple>(\d+\s+)*\d+)\s*\)\s*(?P<val>\d+)"
    var_regex=   r"(?P<varname>\D+):\s*\[(?P<entries>[^\]]*)\]"
    pattern = re.compile(var_regex,flags=re.DOTALL)
    files=[BASE_DIR+"/INDP_InputData/MURI_INDP_data.txt"]
    if sim_number > 0:
        files.append(BASE_DIR+"/Failure and recovery scenarios/recoveryM"+str(magnitude)+"v3.txt")
    vars={}
    for file in files:
        with open(file) as f:
#            print "Opened",file,"."
            lines = f.read()
            vars_strings={}
            for match in pattern.finditer(lines):
                vars_strings[string.strip(match.group('varname'))]=string.strip(match.group('entries'))
#            print "Varstrings=",vars_strings.keys()
            for v in vars_strings:
                entry_string=vars_strings[v]
                entry_pattern=re.compile(entry_regex,flags=re.DOTALL)
                is_match=False
                for match in entry_pattern.finditer(entry_string):
                    is_match=True
                    tuple_string=match.group('tuple')
                    val_string=match.group('val')
                    tuple_string=string.split(tuple_string,",")
                    tuple_string=[int(string.strip(x)) for x in tuple_string]
                    val_string=float(string.strip(val_string))
                    #print v,":",tuple_string,"=>",val_string
                    if v not in vars:
                        vars[v]=[]
                    vars[v].append((tuple_string,val_string))
                if not is_match:
                    recovery_pattern=re.compile(recovery_entry_regex,flags=re.DOTALL)
                    for match in recovery_pattern.finditer(entry_string):
                        is_match=True
                        tuple_string=match.group('tuple')
                        val_string=match.group('val')
                        tuple_string=string.split(tuple_string," ")
                        tuple_string=[int(string.strip(x)) for x in tuple_string]
                        val_string=float(string.strip(val_string))
                        #print v,":",tuple_string,"=>",val_string
                        if v not in vars:
                            vars[v]=[]
                        vars[v].append((tuple_string,val_string))
    G=InfrastructureNetwork("Test")
    global_index=1
    for node in vars['n']:
        n=InfrastructureNode(global_index,node[0][1],node[0][0])
        #print "Adding node",node[0][0],"in layer",node[0][1],"."
        G.G.add_node((n.local_id,n.net_id),data={'inf_data':n})
        global_index+=1
    for arc in vars['a']:
        a=InfrastructureArc(arc[0][0],arc[0][1],arc[0][2]) 
        G.G.add_edge((a.source,a.layer),(a.dest,a.layer),data={'inf_data':a})
    if external_interdependency_dir:
        print( "Loading external interdependencies...")
        interdep_arcs=load_interdependencies(1,3,BASE_DIR=external_interdependency_dir)
        for a in interdep_arcs:
            G.G.add_edge((a.source,a.source_layer),(a.dest,a.dest_layer),data={'inf_data':a})
        interdep_arcs=load_interdependencies(3,1,BASE_DIR=external_interdependency_dir)
        for a in interdep_arcs:
            G.G.add_edge((a.source,a.source_layer),(a.dest,a.dest_layer),data={'inf_data':a})
    else:
        for arc in vars['gamma']:
            gamma=float(arc[1])
            if gamma > 0.0:
                a=InfrastructureInterdepArc(arc[0][0],arc[0][1],arc[0][2],arc[0][3],gamma)
                G.G.add_edge((a.source,a.source_layer),(a.dest,a.dest_layer),data={'inf_data':a})
    # Arc Attributes.
    for flow_cost in vars['c']:
        G.G[(flow_cost[0][0],flow_cost[0][2])][(flow_cost[0][1],flow_cost[0][2])]['data']['inf_data'].flow_cost=float(flow_cost[1])*cost_scale
    for rec_cost in vars['f']:
        G.G[(rec_cost[0][0],rec_cost[0][2])][(rec_cost[0][1],rec_cost[0][2])]['data']['inf_data'].reconstruction_cost=float(rec_cost[1])*cost_scale
    for cap in vars['u']:
        G.G[(cap[0][0],cap[0][2])][(cap[0][1],cap[0][2])]['data']['inf_data'].capacity=cap[1]
    for res in vars['h']:
        # Assume only one kind of resource for now.
        G.G[(res[0][0],res[0][2])][(res[0][1],res[0][2])]['data']['inf_data'].resource_usage=res[1]
    # Space attributes.
    for space in vars['g']:
        G.S.append(InfrastructureSpace(int(space[0][0]),float(space[1])))
    for space in vars['beta']:
        G.G[(space[0][0],space[0][2])][(space[0][1],space[0][2])]['data']['inf_data'].space=int(space[0][3])
    for space in vars['alpha']:
        G.G.node[(space[0][0],space[0][1])]['data']['inf_data'].space=int(space[0][2])
    # Node Attributes.
    for rec_cost in vars['q']:
        G.G.node[(rec_cost[0][0],rec_cost[0][1])]['data']['inf_data'].reconstruction_cost=float(rec_cost[1])*cost_scale
    for sup in vars['Mp']:
        G.G.node[(sup[0][0],sup[0][1])]['data']['inf_data'].oversupply_penalty=float(sup[1])*cost_scale
    for sup in vars['Mm']:
        G.G.node[(sup[0][0],sup[0][1])]['data']['inf_data'].undersupply_penalty=float(sup[1])*cost_scale
    for res in vars['p']:
        # Assume only one kind of resource for now.
        G.G.node[(res[0][0],res[0][1])]['data']['inf_data'].resource_usage=res[1]
    for dem in vars['b']:
        G.G.node[(dem[0][0],dem[0][1])]['data']['inf_data'].demand=dem[1]

    # Load failure scenarios.
    if sim_number > 0:
        for func in vars['nresults']:
            if func[0][0] == sim_number:
                G.G.node[(func[0][1],func[0][2])]['data']['inf_data'].functionality=float(func[1])
                G.G.node[(func[0][1],func[0][2])]['data']['inf_data'].repaired=float(func[1])
                #if float(func[1]) == 0.0:
                #    print "Node (",`func[0][1]`+","+`func[0][2]`+") broken."
        for func in vars['aresults']:
            if func[0][0] == sim_number:
                G.G[(func[0][1],func[0][3])][(func[0][2],func[0][3])]['data']['inf_data'].functionality=float(func[1])
                G.G[(func[0][1],func[0][3])][(func[0][2],func[0][3])]['data']['inf_data'].repaired=float(func[1])
                #if float(func[1]) == 0.0:
                #    print "Arc ((",`func[0][1]`+","+`func[0][3]`+"),("+`func[0][2]`+","+`func[0][3]`+")) broken."
    return G
               
def load_infrastructure_array_format_extended(BASE_DIR="../data/Extended_Shelby_County/",v=3,sim_number=1,cost_scale=1.0):
    files = [f for f in os.listdir(BASE_DIR) if os.path.isfile(os.path.join(BASE_DIR, f))]
    netNames = {'Water':1,'Gas':2,'Power':3,'Telecommunication':4}
    G=InfrastructureNetwork("Test")
    global_index=0
    for file in files:
        fname = file[0:-4] 
        if fname[-5:]=='Nodes':
            with open(BASE_DIR+file) as f:
#                print "Opened",file,"."
                data = pd.read_csv(f, delimiter=',')
                net = netNames[fname[:-5]]
                for v in data.iterrows():               
                    n=InfrastructureNode(global_index,net,int(v[1]['ID']))
                    #print "Adding node",node[0][0],"in layer",node[0][1],"."
                    G.G.add_node((n.local_id,n.net_id),data={'inf_data':n})
                    global_index+=1                    
                    G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].reconstruction_cost=float(v[1]['q (complete DS)'])*cost_scale
                    G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].oversupply_penalty=float(v[1]['Mp'])*cost_scale
                    G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].undersupply_penalty=float(v[1]['Mm'])*cost_scale
                    # Assume only one kind of resource for now and one resource for each repaired element.
                    G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].resource_usage=1
                    G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].demand=float(v[1]['Demand'])
    for file in files:
        fname = file[0:-4] 
        if fname[-4:]=='Arcs':
            with open(BASE_DIR+file) as f:
#                print "Opened",file,"."
                data = pd.read_csv(f, delimiter=',')
                net = netNames[fname[:-4]]
                for v in data.iterrows():  
                    for duplicate in range(2):
                        if duplicate==0:
                            a=InfrastructureArc(int(v[1]['Start Node']),int(v[1]['End Node']),net) 
                        elif duplicate==1:
                            a=InfrastructureArc(int(v[1]['End Node']),int(v[1]['Start Node']),net) 
                        G.G.add_edge((a.source,a.layer),(a.dest,a.layer),data={'inf_data':a})
                        G.G[(a.source,a.layer)][(a.dest,a.layer)]['data']['inf_data'].flow_cost=float(v[1]['c'])*cost_scale
                        G.G[(a.source,a.layer)][(a.dest,a.layer)]['data']['inf_data'].reconstruction_cost=float(v[1]['f'])*cost_scale
                        G.G[(a.source,a.layer)][(a.dest,a.layer)]['data']['inf_data'].capacity=float(v[1]['u'])
                        # Assume only one kind of resource for now and one resource for each repaired element.
                        G.G[(a.source,a.layer)][(a.dest,a.layer)]['data']['inf_data'].resource_usage=1   

    for file in files:
        fname = file[0:-4]
        with open(BASE_DIR+file) as f:
#            print "Opened",file,"."
            data = pd.read_csv(f, delimiter=',')
            for v in data.iterrows():
                if fname=='beta':
                    net = netNames[v[1]['Network']]
                    G.G[(int(v[1]['Start Node']),net)][(int(v[1]['End Node']),net)]['data']['inf_data'].space=int(int(v[1]['Subspace']))
                    G.G[(int(v[1]['End Node']),net)][(int(v[1]['Start Node']),net)]['data']['inf_data'].space=int(int(v[1]['Subspace']))
                if fname=='alpha':
                    net = netNames[v[1]['Network']]
                    G.G.node[(int(v[1]['ID']),net)]['data']['inf_data'].space=int(int(v[1]['Subspace']))
                if fname=='g':
                    G.S.append(InfrastructureSpace(int(v[1]['Subspace_ID']),float(v[1]['g'])))       
                if fname=='Interdep' and v[1]['Type']=='Physical':
                    i = int(v[1]['Dependee Node'])
                    net_i = netNames[v[1]['Dependee Network']]
                    j = int(v[1]['Depender Node'])
                    net_j = netNames[v[1]['Depender Network']]
                    a=InfrastructureInterdepArc(i,j,net_i,net_j,gamma=1.0)
                    G.G.add_edge((a.source,a.source_layer),(a.dest,a.dest_layer),data={'inf_data':a})
    return G

def add_failure_scenario(G,DAM_DIR="../data/INDP_7-20-2015/",magnitude=6,v=3,sim_number=1):
    print("Initiallize Damage...")
    if sim_number == "INF":
        # Destory all nodes!!!!
        for n,d in G.G.nodes(data=True):
            G.G.node[n]['data']['inf_data'].functionality=0.0
            G.G.node[n]['data']['inf_data'].repaired=0.0
        for u,v,a in G.G.edges(data=True):
            if not a['data']['inf_data'].is_interdep:
                G.G[u][v]['data']['inf_data'].functionality=0.0
                G.G[u][v]['data']['inf_data'].repaired=0.0
    elif sim_number > 0:
        variables=['v','c','f','q','Mp','Mm','g','b','u','h','p','gamma','alpha','beta','a','n']
        entry_regex=         r"\(\s*(?P<tuple>(\d+\s*,\s*)*\d)\s*\)\s*(?P<val>-?\d+)"
        recovery_entry_regex=r"\(\s*(?P<tuple>(\d+\s+)*\d)\s*\)\s*(?P<val>\d+)"
        var_regex=   r"(?P<varname>\D+):\s*\[(?P<entries>[^\]]*)\]"
        pattern = re.compile(var_regex,flags=re.DOTALL)
        file = DAM_DIR+"/Failure and recovery scenarios/recoveryM"+str(magnitude)+"v3.txt"
        vars={}
        with open(file) as f:
            #print "Opened",file,"."                                                                                                      xs                              
            lines = f.read()
            vars_strings={}
            for match in pattern.finditer(lines):
                vars_strings[string.strip(match.group('varname'))]=string.strip(match.group('entries'))
            #print "Varstrings=",vars_strings.keys()                                                                                                            
            for v in vars_strings:
                entry_string=vars_strings[v]
                entry_pattern=re.compile(entry_regex,flags=re.DOTALL)
                is_match=False
                for match in entry_pattern.finditer(entry_string):
                    is_match=True
                    tuple_string=match.group('tuple')
                    val_string=match.group('val')
                    tuple_string=string.split(tuple_string,",")
                    tuple_string=[int(string.strip(x)) for x in tuple_string]
                    val_string=float(string.strip(val_string))
                    #print v,":",tuple_string,"=>",val_string                                                                                                         
                    if v not in vars:
                        vars[v]=[]
                    vars[v].append((tuple_string,val_string))
                if not is_match:
                    recovery_pattern=re.compile(recovery_entry_regex,flags=re.DOTALL)
                    for match in recovery_pattern.finditer(entry_string):
                        is_match=True
                        tuple_string=match.group('tuple')
                        val_string=match.group('val')
                        tuple_string=string.split(tuple_string," ")
                        tuple_string=[int(string.strip(x)) for x in tuple_string]
                        val_string=float(string.strip(val_string))
                        #print v,":",tuple_string,"=>",val_string
                        if v not in vars:
                            vars[v]=[]
                        vars[v].append((tuple_string,val_string))
        # Load failure scenarios.
        for func in vars['nresults']:
            if func[0][0] == sim_number:
                G.G.node[(func[0][1],func[0][2])]['data']['inf_data'].functionality=float(func[1])
                G.G.node[(func[0][1],func[0][2])]['data']['inf_data'].repaired=float(func[1])
                #if float(func[1]) == 0.0:
                #    print "Node (",`func[0][1]`+","+`func[0][2]`+") broken."
        for func in vars['aresults']:
            if func[0][0] == sim_number:
                G.G[(func[0][1],func[0][3])][(func[0][2],func[0][3])]['data']['inf_data'].functionality=float(func[1])
                G.G[(func[0][1],func[0][3])][(func[0][2],func[0][3])]['data']['inf_data'].repaired=float(func[1])
                #if float(func[1]) == 0.0:
                #    print "Arc ((",`func[0][1]`+","+`func[0][3]`+"),("+`func[0][2]`+","+`func[0][3]`+")) broken."

def add_random_failure_scenario(G, sample, config=0, DAM_DIR="", no_arc_damage=False):
    import csv
    print("Initiallize Random Damage...")
    if no_arc_damage:
        print("ARCS are NOT DAMAGED in this scenario")
    with open(DAM_DIR+'Initial_node.csv') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            rawN = row[0]
            rawN = rawN.split(',')
            n = (int(rawN[0].strip(' )(')),int(rawN[1].strip(' )(')))
            state = float(row[sample+1])
            G.G.nodes[n]['data']['inf_data'].functionality=state
            G.G.nodes[n]['data']['inf_data'].repaired=state
    if not no_arc_damage:       
        with open(DAM_DIR+'Initial_links.csv') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            for row in data:
                rawUV = row[0]
                rawUV = rawUV.split(',')
                u = (int(rawUV[0].strip(' )(')),int(rawUV[1].strip(' )(')))
                v = (int(rawUV[2].strip(' )(')),int(rawUV[3].strip(' )(')))
                state = float(row[sample+1])
                if state==0.0:
                    G.G[u][v]['data']['inf_data'].functionality=state
                    G.G[u][v]['data']['inf_data'].repaired=state
                
                    G.G[v][u]['data']['inf_data'].functionality=state
                    G.G[v][u]['data']['inf_data'].repaired=state  
               
def add_Wu_failure_scenario(G,DAM_DIR="../data/Wu_Scenarios/",noSet=1,noSce=1, no_arc_damage=False):
    print("Initiallize Wu failure scenarios...")
    if no_arc_damage:
        print("ARCS are NOT DAMAGED in this scenario")
    dam_nodes = {}
    dam_arcs = {}
    folderDir = DAM_DIR+'Set%d/Sce%d/' % (noSet,noSce)
    netNames = {1:'Water',2:'Gas',3:'Power',4:'Telecommunication'}
    ofst = 0 # set to 1 if node IDs start from 1
    # Load failure scenarios.
    if os.path.exists(folderDir):  
        for k in range(1,len(netNames.keys())+1):
            dam_nodes[k] = np.loadtxt(folderDir+
                                    'Net_%s_Damaged_Nodes.txt' % netNames[k]).astype('int')
            dam_arcs[k] = np.loadtxt(folderDir+
                                    'Net_%s_Damaged_Arcs.txt' % netNames[k]).astype('int')        
        for k in range(1,len(netNames.keys())+1):
            if dam_nodes[k].size!=0:
                if dam_nodes[k].size==1:
                    dam_nodes[k] = [dam_nodes[k]]
                for v in dam_nodes[k]:
                    G.G.nodes[(v+ofst,k)]['data']['inf_data'].functionality=0.0
                    G.G.nodes[(v+ofst,k)]['data']['inf_data'].repaired=0.0
#                    print "Node (",`v+ofst`+","+`k`+") broken."
            if dam_arcs[k].size!=0 and not no_arc_damage:
                if dam_arcs[k].size==2:
                    dam_arcs[k] = [dam_arcs[k]]
                for a in dam_arcs[k]:
                    G.G[(a[0]+ofst,k)][(a[1]+ofst,k)]['data']['inf_data'].functionality=0.0
                    G.G[(a[0]+ofst,k)][(a[1]+ofst,k)]['data']['inf_data'].repaired=0.0
#                    print "Arc ((",`a[0]+ofst`+","+`k`+"),("+`a[1]+ofst`+","+`k`+")) broken."
                    
                    G.G[(a[1]+ofst,k)][(a[0]+ofst,k)]['data']['inf_data'].functionality=0.0
                    G.G[(a[1]+ofst,k)][(a[0]+ofst,k)]['data']['inf_data'].repaired=0.0
#                    print "Arc ((",`a[1]+ofst`+","+`k`+"),("+`a[0]+ofst`+","+`k`+")) broken."
#                    
#        for u,v,a in G.G.edges(data=True):
#            if a['data']['inf_data'].is_interdep:
#                if G.G.node[u]['data']['inf_data'].functionality == 0.0:
#                    G.G.node[v]['data']['inf_data'].functionality = 0.0
#                else:
#                    if G.G.node[v]['data']['inf_data'].repaired == 1.0:
#                        G.G.node[v]['data']['inf_data'].functionality = 1.0
    else:
        pass #Undamaging scenrios are not presesnted with any file or folder in the datasets      
          
def load_recovery_scenario(N,T,action_file):
    with open(action_file) as f:
        lines=f.readlines()[1:]
        for line in lines:
            data=string.split(line,",")
            time=int(data[0])
            if time <= T:
                action=data[1]
                if "/" in action:
                    # Edge.
                    edge=string.split(action,"/")
                    u=(int(string.split(edge[0],".")[0]),int(string.split(edge[0],".")[1]))
                    v=(int(string.split(edge[1],".")[0]),int(string.split(edge[1],".")[1]))
                    N.G[u][v]['data']['inf_data'].functionality=1.0
                    N.G[u][v]['data']['inf_data'].repaired=1.0
                else:
                    # Node.    
                    node=string.split(action,".")
                    n=(int(node[0]),int(node[1]))
                    N.G.node[n]['data']['inf_data'].functionality=1.0
                    N.G.node[n]['data']['inf_data'].repaired=1.0
            else:
                return N

def compare_results(strategy_list_string,BASE_DIR="../data/INDP_7-20-2015/",magnitude=6,v=3,sim_number=1):
    if sim_number > 0:
        variables=['v','c','f','q','Mp','Mm','g','b','u','h','p','gamma','alpha','beta','a','n']
        entry_regex=         r"\(\s*(?P<tuple>(\d+\s*,\s*)*\d)\s*\)\s*(?P<val>-?\d+)"
        recovery_entry_regex=r"\(\s*(?P<tuple>(\d+\s+)*\d)\s*\)\s*(?P<val>\d+)"
        var_regex=   r"(?P<varname>\D+):\s*\[(?P<entries>[^\]]*)\]"
        pattern = re.compile(var_regex,flags=re.DOTALL)
        file = BASE_DIR+"/Failure and recovery scenarios/recoveryM"+str(magnitude)+"v3.txt"
        vars={}
        with open(file) as f:
            #print "Opened",file,"."                                                                                                      xs                                      
            lines = f.read()
            vars_strings={}
            for match in pattern.finditer(lines):
                vars_strings[string.strip(match.group('varname'))]=string.strip(match.group('entries'))
            #print "Varstrings=",vars_strings.keys()                                                                                                                              
            for v in vars_strings:
                entry_string=vars_strings[v]
                entry_pattern=re.compile(entry_regex,flags=re.DOTALL)
                is_match=False
                for match in entry_pattern.finditer(entry_string):
                    is_match=True
                    tuple_string=match.group('tuple')
                    val_string=match.group('val')
                    tuple_string=string.split(tuple_string,",")
                    tuple_string=[int(string.strip(x)) for x in tuple_string]
                    val_string=float(string.strip(val_string))
                    #print v,":",tuple_string,"=>",val_string                                                                                                                     
                    if v not in vars:
                        vars[v]=[]
                    vars[v].append((tuple_string,val_string))
                if not is_match:
                    recovery_pattern=re.compile(recovery_entry_regex,flags=re.DOTALL)
                    for match in recovery_pattern.finditer(entry_string):
                        is_match=True
                        tuple_string=match.group('tuple')
                        val_string=match.group('val')
                        tuple_string=string.split(tuple_string," ")
                        tuple_string=[int(string.strip(x)) for x in tuple_string]
                        val_string=float(string.strip(val_string))
                        #print v,":",tuple_string,"=>",val_string                                                                                                                 
                        if v not in vars:
                            vars[v]=[]
                        vars[v].append((tuple_string,val_string))
        repair_dict={}
        # Strategy list string ex: [(13.1)/(21.3,56.3)]-[(34.3)/(45.3)]-[(41)]-[]-[]-[]-[]-[]-[]-[]-[]-[]-[]-[]-[]-[]-[]-[]-[]-[]
        for func in vars['resultsw']:
            if func[0][0] == sim_number:
                node_num=int(func[0][1])
                net_num= int(func[0][2])
                iteration_repaired=int(func[1])
                if iteration_repaired not in repair_dict:
                    repair_dict[iteration_repaired]=[]
                #print "Adding repair:",node_num,".",net_num,"on iteration",iteration_repaired
                repair_dict[iteration_repaired].append("("+str(node_num)+"."+str(net_num)+")")
        for func in vars['resultsy']:
            if func[0][0] == sim_number:
                src_node=int(func[0][1])
                dst_node=int(func[0][2])
                net_num =int(func[0][3])
                iteration_repaired=int(func[1])
                if iteration_repaired not in repair_dict:
                    repair_dict[iteration_repaired]=[]
                #print "Adding repair",src_node,".",net_num,"=>",dst_node,".",net_num,"on iteration",iteration_repaired
                repair_dict[iteration_repaired].append("("+str(src_nodeum)+"."+str(net_num)+","+str(dst_node)+"."+str(net_num)+")")
        it_array=repair_dict.keys()
        it_array.sort()
        strat_list_array=string.split(strategy_list_string,"-")
        strat_list_array=[x[1:-1] for x in strat_list_array]
        print(repair_dict)
        print(strat_list_array)
        for s in it_array:
            if len(strat_list_array) >= s:
                real_strats=repair_dict[s]
                comp_strats=string.split(strat_list_array[s-1],"/")
                for r in real_strats:
                    if r not in comp_strats:
                        return False
                for r in comp_strats:
                    if r not in real_strats:
                        return False
    return True
                


def count_interdependencies(N):
    interdep_dict={1:{2:0,3:0},2:{1:0,3:0},3:{1:0,2:0}}
    total_interdeps=0
    for u,v,a in N.G.edges(data=True):
        if a['data']['inf_data'].is_interdep:
            total_interdeps+=1
            src_layer=u[1]
            dst_layer=v[1]
            print(str(u[0])+"."+str(u[1]),"=>",str(v[0])+"."+str(v[1]))
            interdep_dict[dst_layer][src_layer]=interdep_dict[dst_layer][src_layer]+1
            #if N.G.node[u]['data']['inf_data'].demand >= 0:
            #    print "...is supply node"
            #else:
            #    print "I'm demanding",N.G.node[u]['data']['inf_data'].demand
    print( "Total Interdependencies =",total_interdeps)
    print( "Layer 1 (Water) dependencies on Layer 2 (Gas) =  ",interdep_dict[1][2])
    print( "Layer 1 (Water) dependencies on Layer 3 (Power) =",interdep_dict[1][3])
    print( "Layer 2 (Gas) dependencies on Layer 1 (Water)   =",interdep_dict[2][1])
    print( "Layer 2 (Gas) dependencies on Layer 3 (Power)   =",interdep_dict[2][3])
    print( "Layer 3 (Power) dependencies on Layer 1 (Water) =",interdep_dict[3][1])
    print( "Layer 3 (Power) dependencies on Layer 2 (Gas)   =",interdep_dict[3][2])

def count_supply(N,layer_id=1):
    total_supply=0.0
    total_demand=0.0
    for n,d in N.G.nodes(data=True):
        if n[1] == layer_id:
            demand=d['data']['inf_data'].demand
            print( "Demand for node",n,"=",demand)
            if demand < 0:
                total_demand+=-demand
            else:
                total_supply+=demand
    print( "Total supply for Layer",layer_id,"=",total_supply)
    print( "Total demand for Layer",layer_id,"=",total_demand)
            
def count_nodes(N,layer_id=1):
    num_nodes=len([x for x,d in N.G.nodes(data=True) if x[1] == layer_id])
    num_arcs= len([u for u,v,a in N.G.edges(data=True) if u[1] == layer_id and not a['data']['inf_data'].is_interdep])
    print( "Nodes in Layer",layer_id,"=",num_nodes)
    print( "Arcs in Layer", layer_id,"=",num_arcs)
        

def debug():
    #N=load_infrastructure_data(BASE_DIR="../../../iINDP",sim_number=250,magnitude=6)
    N=load_infrastructure_data(BASE_DIR="../data/INDP_7-20-2015/",sim_number=250,magnitude=8,external_interdependency_dir="../data/INDP_4-12-2016")
    N.to_csv()
    #N=load_infrastructure_data(BASE_DIR="../../../iINDP",sim_number=250,magnitude=9)
    count_interdependencies(N)
    count_nodes(N,1)
    count_nodes(N,3)
    #count_supply(N,1)
    #count_supply(N,2)
    #count_supply(N,3)
#debug()

def read_failure_scenario(BASE_DIR="../data/INDP_7-20-2015/",magnitude=6,v=3,sim_number=1):
    if sim_number == "INF":
        return None
    elif sim_number > 0:
        variables=['v','c','f','q','Mp','Mm','g','b','u','h','p','gamma','alpha','beta','a','n']
        entry_regex=         r"\(\s*(?P<tuple>(\d+\s*,\s*)*\d)\s*\)\s*(?P<val>-?\d+)"
        recovery_entry_regex=r"\(\s*(?P<tuple>(\d+\s+)*\d)\s*\)\s*(?P<val>\d+)"
        var_regex=   r"(?P<varname>\D+):\s*\[(?P<entries>[^\]]*)\]"
        pattern = re.compile(var_regex,flags=re.DOTALL)
        file = BASE_DIR+"/Failure and recovery scenarios/recoveryM"+str(magnitude)+"v3.txt"
        vars={}
        with open(file) as f:
            #print "Opened",file,"."                                                                                                      xs                              
            lines = f.read()
            vars_strings={}
            for match in pattern.finditer(lines):
                vars_strings[string.strip(match.group('varname'))]=string.strip(match.group('entries'))
            #print "Varstrings=",vars_strings.keys()                                                                                                            
            for v in vars_strings:
                entry_string=vars_strings[v]
                entry_pattern=re.compile(entry_regex,flags=re.DOTALL)
                is_match=False
                for match in entry_pattern.finditer(entry_string):
                    is_match=True
                    tuple_string=match.group('tuple')
                    val_string=match.group('val')
                    tuple_string=string.split(tuple_string,",")
                    tuple_string=[int(string.strip(x)) for x in tuple_string]
                    val_string=float(string.strip(val_string))
                    #print v,":",tuple_string,"=>",val_string                                                                                                         
                    if v not in vars:
                        vars[v]=[]
                    vars[v].append((tuple_string,val_string))
                if not is_match:
                    recovery_pattern=re.compile(recovery_entry_regex,flags=re.DOTALL)
                    for match in recovery_pattern.finditer(entry_string):
                        is_match=True
                        tuple_string=match.group('tuple')
                        val_string=match.group('val')
                        tuple_string=string.split(tuple_string," ")
                        tuple_string=[int(string.strip(x)) for x in tuple_string]
                        val_string=float(string.strip(val_string))
                        #print v,":",tuple_string,"=>",val_string
                        if v not in vars:
                            vars[v]=[]
                        vars[v].append((tuple_string,val_string))
        return vars

def load_synthetic_network(BASE_DIR="../data/Generated_Network_Dataset_v3",topology='Random',config=6,sample=0,cost_scale=1.0):
    print("Initiallize Damage...")
    net_dir = BASE_DIR+'/'+topology+'Networks/'
    topo_initial = {'Random':'RN','ScaleFree':'SFN','Grid':'GN'}
    with open(net_dir+'List_of_Configurations.txt') as f:
        config_data = pd.read_csv(f, delimiter='\t')
    config_param = config_data.iloc[config]
    noLayers = int(config_param.loc[' No. Layers'])    
    noResource = int(config_param.loc[' Resource Cap'])  
    
    file_dir = net_dir+topo_initial[topology]+'Config_'+str(config)+'/Sample_'+str(sample)+'/'
    G=InfrastructureNetwork("Test")
    global_index=0
    files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
    for k in range(1,noLayers+1):
        for file in files: 
            if file=='N'+str(k)+'_Nodes.txt':
                with open(file_dir+file) as f:
    #                print "Opened",file,"."
                    data = pd.read_csv(f, delimiter='\t',header=None)
                    for v in data.iterrows():                        
                        n=InfrastructureNode(global_index,k,int(v[1][0]))
                        #print "Adding node",node[0][0],"in layer",node[0][1],"."
                        G.G.add_node((n.local_id,n.net_id),data={'inf_data':n})
                        global_index+=1                    
                        G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].reconstruction_cost=float(v[1][7])*cost_scale
                        G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].oversupply_penalty=float(v[1][5])*cost_scale
                        G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].undersupply_penalty=float(v[1][6])*cost_scale
                        # Assume only one kind of resource for now and one resource for each repaired element.
                        G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].resource_usage=1
                        G.G.nodes[(n.local_id,n.net_id)]['data']['inf_data'].demand=float(v[1][4])
        for file in files:
            if file=='N'+str(k)+'_Arcs.txt':
                with open(file_dir+file) as f:
    #                print "Opened",file,"."
                    data = pd.read_csv(f, delimiter='\t',header=None)
                    for v in data.iterrows(): 
                        for duplicate in range(2):
                            if duplicate==0:
                                a=InfrastructureArc(int(v[1][0]),int(v[1][1]),k) 
                            elif duplicate==1:
                                a=InfrastructureArc(int(v[1][1]),int(v[1][0]),k) 
                            G.G.add_edge((a.source,a.layer),(a.dest,a.layer),data={'inf_data':a})
                            G.G[(a.source,a.layer)][(a.dest,a.layer)]['data']['inf_data'].flow_cost=float(v[1][2])*cost_scale
                            G.G[(a.source,a.layer)][(a.dest,a.layer)]['data']['inf_data'].reconstruction_cost=float(v[1][4])*cost_scale
                            G.G[(a.source,a.layer)][(a.dest,a.layer)]['data']['inf_data'].capacity=float(v[1][2])
                            # Assume only one kind of resource for now and one resource for each repaired element.
                            G.G[(a.source,a.layer)][(a.dest,a.layer)]['data']['inf_data'].resource_usage=1   
    for k in range(1,noLayers+1):
        for kt in range(1,noLayers+1):
            if k!=kt:
                for file in files:
                    if file=='Interdependent_Arcs_'+str(k)+'_'+str(kt)+'.txt':
                        with open(file_dir+file) as f:
                #                print "Opened",file,"."
                            try:
                                data = pd.read_csv(f, delimiter='\t',header=None)
                                for v in data.iterrows():                                      
                                    i = int(v[1][0])
                                    net_i = k
                                    j = int(v[1][1])
                                    net_j = kt
                                    a=InfrastructureInterdepArc(i,j,net_i,net_j,gamma=1.0)
                                    G.G.add_edge((a.source,a.source_layer),(a.dest,a.dest_layer),data={'inf_data':a})
                            except:
                                print('Empty file: '+ file)
# add subspace data #!!!
#for v in data.iterrows():
#    if fname=='beta':
#        net = netNames[v[1]['Network']]
#        G.G[(int(v[1]['Start Node']),net)][(int(v[1]['End Node']),net)]['data']['inf_data'].space=int(int(v[1]['Subspace']))
#        G.G[(int(v[1]['End Node']),net)][(int(v[1]['Start Node']),net)]['data']['inf_data'].space=int(int(v[1]['Subspace']))
#    if fname=='alpha':
#        net = netNames[v[1]['Network']]
#        G.G.node[(int(v[1]['ID']),net)]['data']['inf_data'].space=int(int(v[1]['Subspace']))
#    if fname=='g':
#        G.S.append(InfrastructureSpace(int(v[1]['Subspace_ID']),float(v[1]['g'])))  

    return G,noResource,range(1,noLayers+1)

def add_synthetic_failure_scenario(G,DAM_DIR="../data/Generated_Network_Dataset_v3",topology='Random',config=0,sample=0):
    net_dir = DAM_DIR+'/'+topology+'Networks/'
    topo_initial = {'Random':'RN','ScaleFree':'SFN','Grid':'GN'}
    with open(net_dir+'List_of_Configurations.txt') as f:
        config_data = pd.read_csv(f, delimiter='\t')
    config_param = config_data.iloc[config] 
    noLayers = int(config_param.loc[' No. Layers']) 
    noResource = int(config_param.loc[' Resource Cap'])  
    file_dir = net_dir+topo_initial[topology]+'Config_'+str(config)+'/Sample_'+str(sample)+'/'
    files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
    for k in range(1,noLayers+1):
        for file in files: 
            if file=='N'+str(k)+'_Damaged_Nodes.txt':
                with open(file_dir+file) as f:
                    try:
                        data = pd.read_csv(f, delimiter='\t',header=None)
                        for v in data.iterrows():    
                            G.G.nodes[(v[1][0],k)]['data']['inf_data'].functionality=0.0
                            G.G.nodes[(v[1][0],k)]['data']['inf_data'].repaired=0.0
#                            print "Node (",`int(v[1][0])`+","+`k`+") broken."
                    except:
                        print('Empty file: '+ file)
                                
        for file in files:
            if file=='N'+str(k)+'_Damaged_Arcs.txt':
                with open(file_dir+file) as f:
                    try:
                        data = pd.read_csv(f, delimiter='\t',header=None)
                        for a in data.iterrows():   
                            G.G[(a[1][0],k)][(a[1][1],k)]['data']['inf_data'].functionality=0.0
                            G.G[(a[1][0],k)][(a[1][1],k)]['data']['inf_data'].repaired=0.0
#                            print "Arc ((",`int(a[1][0])`+","+`k`+"),("+`int(a[1][1])`+","+`k`+")) broken."
                            
                            G.G[(a[1][1],k)][(a[1][0],k)]['data']['inf_data'].functionality=0.0
                            G.G[(a[1][1],k)][(a[1][0],k)]['data']['inf_data'].repaired=0.0
#                            print "Arc ((",`int(a[1][1])`+","+`k`+"),("+`int(a[1][0])`+","+`k`+")) broken."
                    except:
                        print('Empty file: '+ file)    