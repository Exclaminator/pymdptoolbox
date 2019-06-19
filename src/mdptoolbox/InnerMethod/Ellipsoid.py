from mdptoolbox.InnerMethod.InnerMethod import InnerMethod
from gurobipy import *
from numpy import *


class Ellipsoid(InnerMethod):
    # Initialize Ellipsoid
    def __init__(self, beta):
        InnerMethod.__init__(self)
        self.beta = beta

    # see if a transition kernel p is in sample
    def inSample(self, p) -> bool:
        distances = []
        for a in range(self.problem.A):
            for s in range(self.problem.S):
                # I replaced self.problem.P[a][s] with len(self.problem.P[a][s]), which I think makes more sense
                f_minus_p = subtract(p[a][s], self.problem.P[a][s])

                # distances.append(sum(divide(power(
                #             f_minus_p, 2),
                #         self.problem.P[a][s] + sys.float_info.epsilon)))
                distances.append(sum(power(
                    f_minus_p, 2)))
        #k2 = 2*(self.beta - self.betaMax)
        return max(distances) < self.beta

    # calculate update scalar for inner method
    def run(self, state, action):
        model = Model('EllipsoidModel')
        pGurobi = model.addVars(self.problem.S, vtype=GRB.CONTINUOUS, name="p")
        p = transpose(array(pGurobi.items()))[1]
        objective = LinExpr()
        objective += dot(p, self.problem.V)
        model.setObjective(objective, GRB.MINIMIZE)
        model.addConstr(sum(
            divide(
                multiply(
                    subtract(p, self.problem.P[action][state]),
                    subtract(p, self.problem.P[action][state])),
                self.problem.P[action][state] + sys.float_info.epsilon
            )) <= self.beta)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal
