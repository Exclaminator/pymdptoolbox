from mdptoolbox.InnerMethod.InnerMethod import InnerMethod
from gurobipy import *
from numpy import *


class Interval(InnerMethod):
    # Initialize Interval
    def __init__(self, p_upper, p_lower):
        InnerMethod.__init__(self)
        self.p_upper = p_upper
        self.p_lower = p_lower

    def attachProblem(self, problem):
        InnerMethod.attachProblem(self, problem)
        if self.p_lower.shape == (self.problem.S,):
            self.p_lower = repeat([repeat([self.p_lower], self.problem.S, axis=0)], self.problem.A, axis=0)
        if self.p_upper.shape == (self.problem.S,):
            self.p_upper = repeat([repeat([self.p_upper], self.problem.S, axis=0)], self.problem.A, axis=0)

        assert self.p_lower.shape == (self.problem.A, self.problem.S, self.problem.S),\
            "p_lower must be in the shape A*S*S or S*1."
        assert self.p_upper.shape == (self.problem.A, self.problem.S, self.problem.S),\
            "p_upper must be in the shape A*S*S or S*1."

        self.p_lower = maximum(self.p_lower, 0)
        self.p_upper = minimum(self.p_upper, 1)

    # see if a transition kernel p is in sample
    def inSample(self, p) -> bool:
        for a in range(self.problem.A):
            for s in range(self.problem.S):
                for s2 in range(self.problem.S):
                    if p[a][s][s2] < self.p_lower[a][s][s2] or p[a][s][s2] > self.p_upper[a][s][s2]:
                        return False
        return True

    # calculate update scalar for inner method
    def run(self, state, action):
        model = Model('IntervalModel')
        mu = model.addVar(vtype=GRB.CONTINUOUS, name="mu")
        index = range(len(self.problem.V))
        lu = model.addVars(index, name="lu", vtype=GRB.CONTINUOUS)
        ll = model.addVars(index, name="ll", vtype=GRB.CONTINUOUS)
        for i in index:
            model.addConstr(mu - lu[i] + ll[i] == self.problem.V[i])
            model.addConstr(lu[i] >= 0)
            model.addConstr(ll[i] >= 0)

        objective = LinExpr()
        objective += mu

        for i in index:
            objective += -(self.p_upper[action][state][i] * lu[i])
            objective += (self.p_lower[action][state][i] * ll[i])

        model.setObjective(objective, GRB.MAXIMIZE)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        # todo: debug -> AttributeError: b"Unable to retrieve attribute 'objVal'"
        return model.objVal
