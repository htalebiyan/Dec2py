import string 
import networkx as nx
import os
from gurobipy import *

class INDPComponents:
    def __init__(self):
        self.components=[]
        self.num_components=0
        self.gc_size=0
    def add_component(self,members,excess_supply):
        self.components.append((members,excess_supply))
    def to_csv_string(self):
        comp_strings=[]
        for c in self.components:
            comp=c[0]
            supp=c[1]
            comp_string="/".join(c[0])
            comp_string+=":"+str(c[1])
            comp_strings.append(comp_string)
        return ",".join(comp_strings)
    @classmethod
    def calculate_components(clss,m,net,t=0,layers=[1,2,3]):
        indp_components=INDPComponents()
        components=net.get_clusters(layers[0])
        indp_components.num_components=len(components)
        indp_components.gc_size=net.gc_size(layers[0])
        for c in components:
            total_excess_supply=0.0
            members=[]
            for n in c:
                members.append(str(n[0])+"."+str(n[1]))
                excess_supply=0.0
                excess_supply+=m.getVarByName('delta+_'+str(n)+","+str(t)).x
                excess_supply+=-m.getVarByName('delta-_'+str(n)+","+str(t)).x
                total_excess_supply+=excess_supply
            indp_components.add_component(members,total_excess_supply)
        return indp_components
    @classmethod
    def from_csv_string(clss,csv_string):
        indp_components=INDPComponents()
        comps=csv_string
        for comp in comps:
            data=string.split(comp,":")
            members=string.split(data[0],"/")
            supp=str.strip(data[1])
            indp_components.add_component(members,float(supp))
        return indp_components

class INDPResults:
    cost_types=["Space Prep","Arc","Node","Over Supply","Under Supply","Flow","Total","Under Supply Perc"]
    def __init__(self,layers=[]):
        self.results={}
        self.layers=layers
        self.results_layer={l:{} for l in layers}
    def __len__(self):
        return len(self.results)
    def __getitem__(self,index):
        return self.results[index]
    def extend(self,indp_result,t_offset=0,t_start=0,t_end=0):
        if t_end == 0: 
            t_end = len(indp_result)
        for new_t,t in zip([x+t_offset for x in range(t_end-t_start)],[y+t_start for y in range(t_end-t_start)]):
            self.results[new_t]=indp_result.results[t]
        if self.layers:
            if t_end == 0: 
                t_end = len(indp_result[self.layers[0]])
            for l in indp_result.results_layer.keys():
                for new_t,t in zip([x+t_offset for x in range(t_end-t_start)],[y+t_start for y in range(t_end-t_start)]):
                    self.results_layer[l][new_t]=indp_result.results_layer[l][t]
    def add_cost(self,t,cost_type,cost,cost_layer={}):
        if t not in self.results:
            self.results[t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Under Supply Perc":0.0,"Flow":0.0,"Total":0.0},'actions':[],'gc_size':0,'num_components':0,'components':INDPComponents(),'run_time':0.0}
        self.results[t]['costs'][cost_type]=cost
        if self.layers:
            for l in cost_layer.keys():
                if t not in self.results_layer[l]:
                    self.results_layer[l][t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Under Supply Perc":0.0,"Flow":0.0,"Total":0.0},'actions':[],'gc_size':0,'num_components':0,'components':INDPComponents(),'run_time':0.0}
                self.results_layer[l][t]['costs'][cost_type]=cost_layer[l]
    def add_run_time(self,t,run_time):
        if t not in self.results:
            self.results[t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Under Supply Perc":0.0,"Flow":0.0,"Total":0.0},'actions':[],'gc_size':0,'num_components':0,'components':INDPComponents(),'run_time':0.0}
        self.results[t]['run_time']=run_time
        if self.layers:
            for l in self.layers:
                if t not in self.results_layer[l]:
                    self.results_layer[l][t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Under Supply Perc":0.0,"Flow":0.0,"Total":0.0},'actions':[],'gc_size':0,'num_components':0,'components':INDPComponents(),'run_time':0.0}
                self.results_layer[l][t]['run_time']=run_time
    def add_action(self, t, action, save_layer=True):
        if t not in self.results:
            self.results[t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Under Supply Perc":0.0,"Flow":0.0,"Total":0.0},'actions':[],'gc_size':0,'num_components':0,'components':INDPComponents(),'run_time':0.0}
        self.results[t]['actions'].append(action)
        if self.layers and save_layer:
            action_layer = int(action[-1])
            if t not in self.results_layer[action_layer]:
                self.results_layer[action_layer][t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Under Supply Perc":0.0,"Flow":0.0,"Total":0.0},'actions':[],'gc_size':0,'num_components':0,'components':INDPComponents(),'run_time':0.0}
            self.results_layer[action_layer][t]['actions'].append(action)
    def add_gc_size(self,t,gc_size):
        if t not in self.results:
            self.results[t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Under Supply Perc":0.0,"Flow":0.0,"Total":0.0},'actions':[],'gc_size':0,'num_components':0,'components':INDPComponents(),'run_time':0.0}
        self.results[t]['gc_size']=gc_size
    def add_num_components(self,t,num_components):
        if t not in self.results:
            self.results[t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Under Supply Perc":0.0,"Flow":0.0,"Total":0.0},'actions':[],'gc_size':0,'num_components':0,'components':INDPComponents(),'run_time':0.0}
        self.results[t]['num_components']=num_components
    def add_components(self,t,components):
        if t not in self.results:
            self.results[t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Under Supply Perc":0.0,"Flow":0.0,"Total":0.0},'actions':[],'gc_size':0,'num_components':0,'components':INDPComponents(),'run_time':0.0}
        self.results[t]['components']=components
        self.add_num_components(t,components.num_components)
        self.add_gc_size(t,components.gc_size)
    def to_csv(self,outdir,sample_num=1,suffix=""):
        action_file =outdir+"/actions_"+str(sample_num)+"_"+suffix+".csv"
        costs_file =outdir+"/costs_"+str(sample_num)+"_"+suffix+".csv"
        run_time_file =outdir+"/run_time_"+str(sample_num)+"_"+suffix+".csv"
        perc_file  =outdir+"/percolation_"+str(sample_num)+"_"+suffix+".csv"
        comp_file  =outdir+"/components_"+str(sample_num)+"_"+suffix+".csv"
        with open(action_file,'w') as f:
            f.write("t,action\n")
            for t in self.results:
                for a in self.results[t]['actions']:
                    f.write(str(t)+","+a+"\n")
        with open(run_time_file,'w') as f:
            f.write("t,run_time\n")
            for t in self.results:
                f.write(str(t)+","+str(self.results[t]['run_time'])+"\n")
        with open(costs_file,'w') as f:
            f.write("t,Space Prep,Arc,Node,Over Supply,Under Supply,Flow,Total,Under Supply Perc\n")
            for t in self.results:
                costs=self.results[t]['costs']
                f.write(str(t)+","+str(costs["Space Prep"])+","+str(costs["Arc"])+","+str(costs["Node"])+","+str(costs["Over Supply"])+","+str(costs["Under Supply"])+","+str(costs["Flow"])+","+str(costs["Total"])+","+str(costs["Under Supply Perc"])+"\n")
#        with open(perc_file,'w') as f:
#            f.write("t,gc_size,num_components\n")
#            for t in self.results:
#                f.write(str(t)+","+`self.results[t]['gc_size']`+","+`self.results[t]['num_components']`+"\n")
#        with open(comp_file,'w') as f:
#            f.write("t,components\n")
#            for t in self.results:
#                f.write(str(t)+","+self.results[t]['components'].to_csv_string()+"\n")
    def to_csv_layer(self,outdir,sample_num=1,suffix=""):
        for l in self.layers:
            action_file =outdir+"/actions_"+str(sample_num)+"_L"+str(l)+"_"+suffix+".csv"
            costs_file =outdir+"/costs_"+str(sample_num)+"_L"+str(l)+"_"+suffix+".csv"
            run_time_file =outdir+"/run_time_"+str(sample_num)+"_L"+str(l)+"_"+suffix+".csv"
            with open(action_file,'w') as f:
                f.write("t,action\n")
                for t in self.results_layer[l]:
                    for a in self.results_layer[l][t]['actions']:
                        f.write(str(t)+","+a+"\n")
            with open(run_time_file,'w') as f:
                f.write("t,run_time\n")
                for t in self.results_layer[l]:
                    f.write(str(t)+","+str(self.results_layer[l][t]['run_time'])+"\n")
            with open(costs_file,'w') as f:
                f.write("t,Space Prep,Arc,Node,Over Supply,Under Supply,Flow,Total,Under Supply Perc\n")
                for t in self.results_layer[l]:
                    costs=self.results_layer[l][t]['costs']
                    f.write(str(t)+","+str(costs["Space Prep"])+","+str(costs["Arc"])+","+str(costs["Node"])+","+str(costs["Over Supply"])+","+str(costs["Under Supply"])+","+str(costs["Flow"])+","+str(costs["Total"])+","+str(costs["Under Supply Perc"])+"\n")
    @classmethod
    def from_csv(clss,outdir,sample_num=1,suffix=""):
        action_file=outdir+"/actions_"+str(sample_num)+"_"+suffix+".csv"
        costs_file =outdir+"/costs_"  +str(sample_num)+"_"+suffix+".csv"
        perc_file  =outdir+"/percolation_"+str(sample_num)+"_"+suffix+".csv"
        comp_file  =outdir+"/components_" +str(sample_num)+"_"+suffix+".csv"
        run_time_file  =outdir+"/run_time_" +str(sample_num)+"_"+suffix+".csv"
        indp_result=INDPResults()
        if os.path.isfile(action_file): #!!!
            with open(action_file) as f:
                lines=f.readlines()[1:]
                for line in lines:
                    data=line.strip().split(',')
                    t=int(data[0])
                    action=str.strip(data[1])
                    indp_result.add_action(t,action)
            with open(costs_file) as f:
                lines=f.readlines()
                cost_types=lines[0].strip().split(',')[1:]
                for line in lines[1:]:
                    data=line.strip().split(',')
                    t=int(data[0])
                    costs=data[1:]
                    for ct in range(len(cost_types)):
                        indp_result.add_cost(t,cost_types[ct],float(costs[ct]))
            with open(run_time_file) as f:
                lines=f.readlines()
                for line in lines[1:]:
                    data=line.strip().split(',')
                    t=int(data[0])
                    run_time=data[1]
                    indp_result.add_run_time(t,run_time)
#            with open(perc_file) as f:
#                lines=f.readlines()[1:]
#                for line in lines:
#                    data=string.split(str.strip(line),",")
#                    t=int(data[0])
#                    indp_result.add_gc_size(t,int(data[1]))
#                    indp_result.add_num_components(t,int(data[2]))
#            with open(comp_file) as f:
#                lines=f.readlines()[1:]
#                for line in lines:
#                    data=string.split(str.strip(line),",")
#                    t=int(data[0])
#                    comps=data[1:]
#                    if comps[0]!='':
#                        indp_result.add_components(t,INDPComponents.from_csv_string(comps))
#                    else:
##                        print "Caution: No component."
#                        pass
        return indp_result

    @classmethod
    def from_results_dir(clss,outdir,sample_range,player=-1,iteration=-1,suffix="",flip=False):
        avg_results=INDPResults()
        orig_suffix=suffix
        suffix=suffix
        if player > -1 and iteration > -1:
            suffix="P"+str(player)+"_i"+str(iteration)
        elif player > -1:
            suffix="P"+str(player)
        else:
            suffix=""
        if not flip:
            suffix+=orig_suffix
        # Sum cost and percolation values over all samples.
        for s in sample_range:
            sample_result=None
            if not flip:
                sample_result=clss.from_csv(outdir,s,suffix=suffix)
            else:
                if suffix == "":
                    this_suffix=str(s)
                else:
                    this_suffix=suffix+"_"+str(s)
                sample_result=clss.from_csv(outdir,int(orig_suffix),suffix=this_suffix)
            for t in sample_result.results:
                if t not in avg_results.results:
                    avg_results.results[t]={'costs':{"Space Prep":0.0,"Arc":0.0,"Node":0.0,"Over Supply":0.0,"Under Supply":0.0,"Flow":0.0,"Total":0.0},'gc_size':0.0,'num_components':0.0}
                for c in clss.cost_types:
                    avg_results[t]['costs'][c]+=float(sample_result[t]['costs'][c])
                avg_results[t]['gc_size']+=float(sample_result[t]['gc_size'])
                avg_results[t]['num_components']+=float(sample_result[t]['num_components'])
        # Average results over number of samples.
        for t in avg_results.results:
            for c in clss.cost_types:
                avg_results[t]['costs'][c]=avg_results[t]['costs'][c]/float(len(sample_range))
            avg_results[t]['gc_size']=avg_results[t]['gc_size']/float(len(sample_range))
            avg_results[t]['num_components']=avg_results[t]['num_components']/float(len(sample_range))
        return avg_results
