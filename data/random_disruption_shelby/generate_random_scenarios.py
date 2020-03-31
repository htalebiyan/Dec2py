'''
Each node or arc is this database is damages with probability p where p increases 
from 1% to 99% with sample number. For each value, 10 samples are generated, 
which makes a total number of 990 samples. Samples are stacked in an 
increasing order in terms of p. For example, the first 10 samples have p=1%, 
the second 10 samples have p=2%, and so on and so forth.
'''
import numpy as np
import matplotlib.pyplot as plt

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
    temp=np.random.choice([0,1],(213,10),p=[p,1-p])
    if p==0.01:
        arc_samples=temp
    else:
        arc_samples=np.hstack([arc_samples,temp]) 

# initial=initial_net.G.edges()
# edges = []
# inter = []
# for i in initial:
# 	u=i[0]
# 	v=i[1]
# 	if u[1]!=v[1]:
#         inter.append(i)
#         if (v,u) in edges:
#             print((v,u))
#     elif (v,u) not in edges:       
#         edges.append(i)
        
ax=plt.imshow(node_samples)      
plt.xlabel('scenario')
plt.ylabel('element')  
plt.xlim(0,500)