import copy
import indp
from indputils import *

class INRGChoiceNode:
    def __init__(self,owner,pred,iset_id,recovery_action=None):
        self.owner=owner
        self.pred=pred
        self.succs=[]
        self.iset_id=iset_id
        self.best_utility=None
        self.best_strategy=None
        self.recovery_action=recovery_action
    def add_successor(self,succ_node):
        self.succs.append(succ_node)

class INRGTerminalNode:
    def __init__(self,pred,recovery_path):
        self.pred=pred
        self.utility={}
        self.recovery_action=recovery_path[-1]
        self.recovery_path=recovery_path
        
class INRGGameTree:
    def __init__(self,InterdepNet,players=[1,3],player_ordering=[1,3],T=1):
        self.terminal_nodes=[]
        self.players=players
        self.player_ordering=player_ordering
        self.network=InterdepNet
        self.iset_counter=0
        # Get available recovery actions for lead player.
        damaged_nodes=[n for n,d in InterdepNet.G.nodes_iter(data=True) if d['data']['inf_data'].functionality == 0.0 and n[1] == player_ordering[0]]
        damaged_arcs_all =[(u,v) for u,v,a in InterdepNet.G.edges_iter(data=True) if a['data']['inf_data'].functionality == 0.0 and not a['data']['inf_data'].is_interdep and u[1] == player_ordering[0]]
        damaged_arcs=[]
        for u,v in damaged_arcs_all:
            if not (v,u) in damaged_arcs:
                damaged_arcs.append((u,v))
        damaged_components=damaged_nodes+damaged_arcs
        # Create root node, and construct tree via DFS.
        self.root=INRGChoiceNode(player_ordering[0],None,self.iset_counter)
        self.iset_counter+=1
        T=T-1
        while damaged_components:
            action=damaged_components.pop()
            self.root.add_successor(self.construct_tree(self.root,copy.copy(self.player_ordering[1:]),[action],T))
            
    def construct_tree(self,pred,player_ordering,recovery_path,T):
        """ Recursive algorithm to create game tree using DFS. """
        created_node=None
        if len(player_ordering) == 0 and T == 0:
            # create INRG terminal node and evaluate utility.
            created_node=INRGTerminalNode(pred,recovery_path)
            self.terminal_nodes.append(created_node)
            for p in self.player_ordering:
                utility=calculate_utility(self.network,recovery_path,self.player_ordering,p)
                created_node.utility[p]=utility
            return created_node
        if len(player_ordering) == 0 and T > 0:
            # Reset player ordering, and do more timesteps!
            player_ordering=self.player_ordering
            T=T-1
        damaged_nodes=[n for n,d in self.network.G.nodes_iter(data=True) if d['data']['inf_data'].functionality == 0.0 and n[1] == player_ordering[0] and n not in recovery_path]
        # Only want one direction to be repaired.
        damaged_arcs_all =[(u,v) for u,v,a in self.network.G.edges_iter(data=True) if a['data']['inf_data'].functionality == 0.0 and not a['data']['inf_data'].is_interdep and u[1] == player_ordering[0] and (u,v) not in recovery_path]
        damaged_arcs=[]
        for u,v in damaged_arcs_all:
            if (v,u) not in damaged_arcs:
                damaged_arcs.append((u,v))
        damaged_components=damaged_nodes+damaged_arcs
        if len(damaged_components) == 0:
            # No more recovery actions to take, create terminal node.
            created_node=INRGTerminalNode(pred,recovery_path)
            self.terminal_nodes.append(created_node)
            for p in self.player_ordering:
                utility=calculate_utility(self.network,recovery_path,self.player_ordering,p)
                created_node.utility[p]=utility
            return created_node
        # Still players, time, and recovery actions... recurse!
        created_node=INRGChoiceNode(player_ordering[0],pred,self.iset_counter,recovery_path[-1])
        self.iset_counter+=1
        while damaged_components:
            action=damaged_components.pop()
            created_node.add_successor(self.construct_tree(created_node,player_ordering[1:],recovery_path+[action],T))
        return created_node

    def backwards_induction(self,current_node):
        if isinstance(current_node,INRGTerminalNode):
            return (current_node.utility,[current_node.recovery_action])
        else:
            owner=current_node.owner
            best_utility=None
            best_strategy=None
            for child in current_node.succs:
                result=self.backwards_induction(child)
                utility=result[0]
                action =result[1]
                if not best_utility:
                    best_utility=utility
                    best_strategy=action
                if utility[owner].results[0]["costs"]["Total"] < best_utility[owner].results[0]["costs"]["Total"]:
                    best_utility=utility
                    best_strategy=action
            current_node.best_utility=best_utility
            current_node.best_strategy=best_strategy
            return (current_node.best_utility,current_node.best_strategy+[current_node.recovery_action])
        
# Experiment functions.
def run_backwards_induction(N,sample=1,players=[1,3],player_ordering=[1,3],T=50,outdir="../results/test"):
    indp_results={}
    for p in players:
        indp_results[p]=INDPResults()
    for t in range(1,T+1):
        gt=INRGGameTree(N,players=players,player_ordering=player_ordering)
        print( "T=",t,":",len(gt.terminal_nodes))
        if len(gt.root.succs) > 0:
            strategy=gt.backwards_induction(gt.root)
            utility=strategy[0]
            repairs=strategy[1]
            for p in players:
                indp_results[p].extend(utility[p],t)
                for repair in repairs:
                    if repair:
                        action=""
                        owner=None
                        if not isinstance(repair[0],tuple):
                            owner = repair[1]
                            if owner == p:
                                action=str(repair[0])+"."+str(repair[1])
                                indp_results[p].add_action(t,action)
                        else:
                            owner= repair[0][1]
                            if owner == p:
                                action=str(repair[0][0])+"."+str(repair[0][1])+"/"+str(repair[1][0])+"."+str(repair[1][1])
                                indp_results[p].add_action(t,action)
                                action=str(repair[1][0])+"."+str(repair[1][1])+"/"+str(repair[0][0])+"."+str(repair[0][1])
                                indp_results[p].add_action(t,action)
                        apply_repairs(N,[r for r in repairs if r])
            for p in players:
                indp_results[p].to_csv(outdir,1,suffix=str(p))
        else:
            break
        
# Helper functions.
def apply_repairs(N,repairs,p=None):
    repair_cost={}
    repair_cost["Node"]=0.0
    repair_cost["Arc"]=0.0
    for repair in repairs:
        if not isinstance(repair[0],tuple):
            dependency_functionality=True
            if p == repair[1]:
                repair_cost["Node"]=repair_cost["Node"]+N.G.node[repair]['data']['inf_data'].reconstruction_cost
            for n in N.G.predecessors_iter(repair):
                if n[1] != repair[1] and N.G.node[n]['data']['inf_data'].functionality==0.0 and n not in repairs:
                    dependency_functionality=False
            if dependency_functionality:
                N.G.node[repair]['data']['inf_data'].functionality=1.0
                N.G.node[repair]['data']['inf_data'].repaired=1.0
        else:
            N.G[repair[0]][repair[1]]['data']['inf_data'].functionality=1.0
            N.G[repair[0]][repair[1]]['data']['inf_data'].repaired=1.0
            N.G[repair[1]][repair[0]]['data']['inf_data'].functionality=1.0
            N.G[repair[1]][repair[0]]['data']['inf_data'].repaired=1.0
            if p == repair[0][1]:
                repair_cost["Arc"]=repair_cost["Arc"]+N.G[repair[0]][repair[1]]['data']['inf_data'].reconstruction_cost
    return repair_cost

def calculate_utility(N,recovery_path,active_layers,player):
    repaired_network=N.copy()
    repair_cost=apply_repairs(repaired_network,recovery_path,player)
    result=indp.indp(repaired_network,0,layers=active_layers,controlled_layers=[player])
    # Should save all costs here?
    result[1].add_components(0,INDPComponents.calculate_components(result[0],repaired_network,layers=[player]))
    result[1].results[0]['costs']['Node']=repair_cost['Node']
    result[1].results[0]['costs']['Arc'] =repair_cost['Arc']
    result[1].results[0]['costs']['Total']=result[1].results[0]['costs']['Total']+repair_cost['Node']+repair_cost['Arc']
    utility=result[1]
    return utility
