%Created by Andres D. Gonzalez & Leonardo Dueñas-Osorio - 2015
%email: andres.gonzalez@rice.edu & leonardo.duenas-osorio@rice.edu
%SISRRA (prof. Leonardo Dueñas-Osorio Research Group) http://duenas-osorio.rice.edu/Content.aspx?id=45

README:

!!!!!!!!!!!!!!!!!!Failure probabilities:

The probabilities of failure of each component (probM*.txt) are related to the magnitude of the disaster. 
File probM*.txt has the probability of failure of each component for the gas, power and water networks in Shelby County, TN, 
given an earthquake with magnitude * (6,7,8, and 9)

In each file there is a "proba" variable, to indicate the probability of failure of each arc(for the given magnitude).
The format for each line in "proba" is: 
(s , t, k) p -> s is the starting node, t is the ending node, k is the network number (1 for water, 2 for gas, and 3 for power), and p is the failure probability for that arc.

In each file there is a "probn" variable, to indicate the probability of failure of each node (for the given magnitude).
The format for each line in "probn" is: 
(i , k) p -> i is the node number, k is the network number (1 for water, 2 for gas, and 3 for power), and p is the failure probability for that node.



!!!!!!!!!!!!!!!!!!Failure and recovery scenarios:

The files named recoveryM*v3.txt (where M indicates the simulated earthquake magnitude) contain the following varaibles:
resultsy -> format: (simulation, starting node, ending node, network number) iteration , where iteration indicates when the arc was fixed
resultsw -> format: (simulation, node number, network number) iteration, where iteration indicates when the arc was fixed
aresults -> format: (simulation, starting node, ending node, network number) functionality, where functionality=0 indicates the arc was destroyed in that earthquake simulation
nresults -> format: (simulation, node number, network number) iteration in which the node was fixed, where functionality=0 indicates the node was destroyed in that earthquake simulation

In this case, we performed 1000 independent simulations for each earthquake magnitude.


!!!!!!!!!!!!!!!!!!Recovery costs:
There are four files P*Recovery.mat, where the * indicates the analyzed earthquake magnitude.
Each file has variables with the format:
R# <simulations x RecoveryIterations x v>, 
v is related to the different values of limited resource used (position 1 is associated to v=3, position 2 with v=6, position 3 with v=9, position 4 with v=12)
RecoveryIterations indicates the maximum number of INDP iterations observed to recover the system after an earthquake of the given magnitude.
simulations indicates the number of independent simulations performed (destruction-reconstruction process). In this case, we performed 1000 simulations.

R1-> Flow cost
R2-> Arc + Node reconstruction costs 
R3-> Geographical preparation costs 
R4-> Flow cost + reconstruction costs + geographical preparation costs 
R5-> Cumulative reconstruction costs
R6-> Unbalance costs
R7-> Unbalance costs + Flow cost + reconstruction costs + geographical preparation costs (i.e., the full objective function of the INDP)
R8-> Arcs recovered
R9-> Nodes recovered

On the other hand, variables mR# are the average of R# over the performed simulations, and dR# are the standard deviation of R# over the performed simulations.


!!!!!!!!!!!!!!!!!!INDP Input data

c ->unitary flow costs
f ->reconstruction cost of link
q ->reconstruction cost of node
Mp ->oversupply penalty
Mm ->undersupply penalty
g ->cost of space preparation
b ->demand
u ->link capacity
h ->resource usage when reconstructing link
p ->resource usage when reconstructing node
v ->resource availability
gamma ->physical interdependence between components
alpha ->nodes belonging to each space
beta ->links belonging to each space
a ->arcs in the system
n ->nodes in the system

!Format:
c:array(N,N,K,L) of real
f:array(N,N,K) of real
q:array(N,K) of real
Mp:array(N,K,L) of real
Mm:array(N,K,L) of real
g:array(S) of real
b:array(N,K,L) of real
u:array(N,N,K) of real
h:array(N,N,K,R) of real
p:array(N,K,R) of real
v:array(R) of real
gamma:array(N,N,K,K) of real
alpha:array(N,K,S) of real
beta:array(N,N,K,S) of real
a:array(N,N,K) of real
n:array(N,K) of real
