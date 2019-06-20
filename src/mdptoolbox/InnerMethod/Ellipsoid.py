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
        for action in range(self.problem.A):
            for state in range(self.problem.S):
                # consider euclidean distance away
                # I replaced self.problem.P[a][s] with len(self.problem.P[a][s]), which I think makes more sense
                out_of_bounds = divide(sum(power(
                            subtract(p, self.problem.P[action][state]),
                            2)
                    ),  len(self.problem.P[action][state])
                ) >= self.beta
                if out_of_bounds:
                    return False

                    # k2 = 2*(self.beta - self.betaMax) is the real constraint for the ellipsoid model

        return True

    # calculate update scalar for inner method
    def run(self, state, action):
        model = Model('EllipsoidModel')
        pGurobi = model.addVars(self.problem.S, vtype=GRB.CONTINUOUS, name="p")
        p = transpose(array(pGurobi.items()))[1]
        objective = LinExpr()
        objective += dot(p, self.problem.V)
        model.setObjective(objective, GRB.MINIMIZE)
        # maybe this constraint is wrong?
        model.addConstr(divide(sum(
                multiply(
                    subtract(p, self.problem.P[action][state]),
                    subtract(p, self.problem.P[action][state]))
            ),  len(self.problem.P[action][state])
        ) <= self.beta)

        # model.addConstr(sum(
        #     divide(
        #         multiply(
        #             subtract(p, self.problem.P[action][state]),
        #             subtract(p, self.problem.P[action][state])),
        #         self.problem.P[action][state] + sys.float_info.epsilon
        #     )) <= self.beta)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal
