from mdptoolbox.InnerMethod import InnerMethod
from scipy.stats import wasserstein_distance
from gurobipy import *
from numpy import *


class Wasserstein(InnerMethod):
    # Initialize Wasserstein
    def __init__(self, beta):
        InnerMethod.__init__(self)
        self.beta = beta

    # see if a transition kernel p is in sample
    def inSample(self, p) -> bool:
        max_distance = 0;
        for a in range(self.problem.A):
            for s in range(self.problem.S):
                max_distance = max(max_distance, wasserstein_distance(self.problem.P[a][s], p[a][s]))
        return max_distance < self.beta

    # calculate update scalar for inner method
    def run(self, state, action):
        model = Model('SigmaEMD')
        pGurobi = model.addVars(self.problem.S, vtype=GRB.CONTINUOUS, name="p")
        p = transpose(array(pGurobi.items()))[1]
        emdGurobi = model.addVars(self.problem.S, vtype=GRB.CONTINUOUS, name="emd")
        emd = transpose(array(emdGurobi.items()))[1]
        emdAbsGurobi = model.addVars(self.problem.S, vtype=GRB.CONTINUOUS, name="emd_abs")
        emdAbs = transpose(array(emdAbsGurobi.items()))[1]
        objective = LinExpr()
        objective += dot(p, self.problem.V)
        model.setObjective(objective, GRB.MINIMIZE)
        for i in range(self.problem.S):
            if i == 0:
                model.addConstr(emd[i] == p[i] - self.problem.P[action][state][i])
            else:
                model.addConstr(emd[i] == p[i] - self.problem.P[action][state][i] + emd[i - 1])
            model.addConstr(emd[i] <= emdAbs[i])
            model.addConstr(-emd[i] <= emdAbs[i])
        model.addConstr((-sum(emdAbs)) <= self.beta)
        model.addConstr(sum(emdAbs) <= self.beta)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal
