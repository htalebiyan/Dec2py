from gurobipy import *

m=Model('test')
m.setParam('OutputFlag',False)
noP = 5
R = 6


for p in range(noP):
    m.addVar(name='v_,'+`p`,vtype=GRB.INTEGER)
m.update()
LHS=LinExpr()
RHS=R
for p in range(noP):
    LHS+=m.getVarByName('v_,'+`p`)
#    m.getVarByName('v_,'+`s`).LB=0
m.addConstr(LHS,GRB.LESS_EQUAL,RHS,'constr')

# Populate objective function.
objFunc=1#LHS
m.setObjective(objFunc,GRB.MAXIMIZE)
m.update()
m.setParam('MIPFocus', 2)
noSol = (R+1)**noP
m.setParam('PoolSolutions', noSol)
m.setParam('PoolSearchMode', 2)

m.optimize()
uniqueSol = []
for i in range(m.solcount):
    m.setParam('SolutionNumber', i)
    sol = []
    for p in range(noP):
        sol.append(int(m.getVars()[p].Xn))
    print sol
    if sol not in uniqueSol:
        uniqueSol.append(sol)


 
