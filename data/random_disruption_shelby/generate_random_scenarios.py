'''
Each node or arc is this database is damages with probability p where p increases 
from 1% to 99% with sample number. For each value, 10 samples are generated, 
which makes a total number of 990 samples. Samples are stacked in an 
increasing order in terms of p. For example, the first 10 samples have p=1%, 
the second 10 samples have p=2%, and so on and so forth.
'''
import numpy as np

p_values = np.linspace(0.01,0.99,99)
node_samples=[]
for p in p_values:
    temp=np.random.choice([0,1],(167,10),p=[p,1-p])
    if p==0.01:
        node_samples=temp
    else:
        node_samples=np.hstack([node_samples,temp])

arc_samples=[]
for p in p_values:
    temp=np.random.choice([0,1],(426,10),p=[p,1-p])
    if p==0.01:
        arc_samples=temp
    else:
        arc_samples=np.hstack([arc_samples,temp]) 
