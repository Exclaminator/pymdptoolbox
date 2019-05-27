from mdptoolbox.mdp import MDP, _printVerbosity
from gurobipy import *
import mdptoolbox.util as _util

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp


class RobustIntervalModel(MDP):
    def __init__(self, transitions, reward, discount, p_lower, p_upper, epsilon=0.01,
                 max_iter=10, skip_check=False):
        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter, skip_check)
        # In the robust interval model, each p is given a lower and upper bound
        self.p_lower = p_lower
        self.p_upper = p_upper

    def run(self):
        # Run the modified policy iteration algorithm.
        self._startRun()
        # TODO perhaps there can be a better initial guess. (v > 0)
        self.V = _np.ones(self.S)
        self.sigma = 0

        while True:
            self.iter += 1
            self.sigma = self.computeSigma()
            self.v_next = _np.full(self.V.shape, _np.inf)

            for s in range(self.S):
                self.v_next[s] = _np.min(_np.transpose(self.R)[s])+self.discount*self.sigma

            if _np.linalg.norm(self.V - self.v_next) < (1 - self.discount) * self.epsilon / (2.0 * self.discount):
                break
            if self.iter >= self.max_iter:
                break

            self.V = self.v_next

        # make policy
        self.policy = _np.zeros(self.S, dtype=_np.int)
        for s in range(self.S):
            self.policy[s] = _np.argmin(_np.transpose(self.R)[s])
        #return policy
        self._endRun()

    def computeValue(self):
        # todo: implement this method

        # action
        # v = min(cost(i,a) + \vega \sigma^hat[s,a])

        value = -1

        return value

    def computeSigma(self):
        model = Model('SigmaIntervalMatrix')
        mu = model.addVar(vtype=GRB.CONTINUOUS, name="mu")
        objective = LinExpr()
        objective += _np.dot(
                        _np.subtract(self.p_upper, self.p_lower),
                        _np.maximum(
                            _np.subtract(
                                _np.multiply(
                                    mu,
                                    _np.ones(self.S, dtype=_np.float)
                                ),
                                self.V
                            ),
                            _np.zeros(self.S)
                        )
                    )

        objective += _np.dot(
                            self.V,
                            self.p_upper)

        objective += _np.multiply(
                            mu,
                            (1 - _np.dot(self.p_upper,_np.ones(self.S, dtype=_np.float))))

        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        return model.objVal
        # for v in model.getVars():
        #     if v.X != 0:
        #         print("aaaaaa %s %f" % (v.Varname, v.X))
