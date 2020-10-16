import random
import cPickle as pickle
import pandas as pd
import sys
import string
import os            
def damage_key(key):
    if key[0]=='w':
        raw = key[2:].split(',')
        new = '('+raw[0].strip(' )(')+', '+raw[1].strip(' )(')+')'
        new2 = None
    if key[0]=='y':
        raw = key[2:].split(',')
        u = '('+raw[0].strip(' )(')+', '+raw[1].strip(' )(')+')'
        v = '('+raw[2].strip(' )(')+', '+raw[3].strip(' )(')+')'
        new = '('+u+', '+v+')'
        new2 = '('+v+', '+u+')'
    return new,new2

def read_actions(results_dir,i,suffix=''):
    actions={}
    results_dir+='/results/indp_results_L'+`4`+'_m'+`0`+'_v'+`4`
    action_file=results_dir+"/actions_"+`i`+"_"+suffix+".csv" 
    if os.path.isfile(action_file):
        with open(action_file) as f:
            lines=f.readlines()[1:]
            for line in lines:
                data=string.split(str.strip(line),",")
                t=int(data[0])
                action=str.strip(data[1])
                k = int(action[-1])
                if '/' in action:
                    act_data=string.split(str.strip(action),"/")
                    u = string.split(str.strip(act_data[0]),".")
                    v = string.split(str.strip(act_data[1]),".")
                    if u[1]!=v[1]:
                        sys.exit('Interdepndency '+act_data+' is counted as an arc')
                    arc_id = `(int(u[0]),k)`+','+`(int(v[0]),k)`
                    actions['y_'+arc_id] = t
                else:
                    act_data=string.split(str.strip(action),".")
                    node_id=int(act_data[0])
                    actions['w_'+`(node_id,k)`] = t
    else:
        sys.exit('No results dir: '+action_file)
    return actions

if __name__ == "__main__": 
    # t_suf = '20200410'
    # samples,costs,costs_local,initial_net = pickle.load(open('data'+t_suf+'/initial_data.pkl', "rb" ))     
    # train_data,test_data = pickle.load(open('data'+t_suf+'/train_test_data.pkl', "rb" )) 
    
    damage_dir = "../data/random_disruption_shelby/"
    with open(damage_dir+'Initial_node.csv') as csvfile:
        node_damage_data = pd.read_csv(csvfile, delimiter=',',index_col=0,header=None)
    with open(damage_dir+'Initial_links.csv') as csvfile:
        arc_damage_data = pd.read_csv(csvfile, delimiter=',',index_col=0,header=None)
    output_dir = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/'
    sample_range = range(50,550)
    rnd_samples = [random.choice(sample_range) for x in range(100)]
    elements = samples.keys()
    
    # for s in rnd_samples:
    #     rnd_elems = [random.choice(elements) for x in range(100)]
    #     for e in rnd_elems:
    #         dam_key,dam_key2 = damage_key(e)
    #         if e[0]=='w':
    #             initial_state = node_damage_data[s+1][dam_key]
    #         if e[0]=='y':
    #             if dam_key in arc_damage_data.index:
    #                 initial_state = arc_damage_data[s+1][dam_key]
    #             else:
    #                 initial_state = arc_damage_data[s+1][dam_key2]
                    
    #         if initial_state!=int(samples[e][0,s-sample_range[0]]):
    #             sys.exit('Discrepancy in initial state in samples: '+`(e,s)`)
    # print 'No Discrepancy in initial state in samples'
    
    output_dir = 'C:/Users/ht20/Documents/Files/STAR_training_data/INDP_random_disruption/'
    T=samples[samples.keys()[0]].shape[0]
    for s in rnd_samples:
        actions=read_actions(output_dir,s)
        for key in actions.keys():
            if sum(samples[key][:,s-sample_range[0]])!=T-actions[key]:
                sys.exit('Discrepancy in repair time in samples: '+`(key,s)`)
    print 'No Discrepancy in repair time in samples'