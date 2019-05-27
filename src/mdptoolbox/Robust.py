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
        self.v = _np.ones(self.S)
        self.v_next = _np.full(self.v.shape, _np.inf)
        self.sigma = 0

        while True:
            self.iter += 1
            self.sigma = self.computeSigma()

            for s in range(self.S):
                self.v_next[s] = _np.min(self.R[s,:])+self.discount*self.sigma

            if _np.linalg.norm(self.v - self.v_next) < (1 - self.discount) * self.epsilon / (2.0 * self.discount):
                break
            if self.iter >= self.max_iter:
                break

            self.v = self.v_next

        # make policy
        policy = _np.zeros(self.S)
        for s in range(self.S):
            policy[s] = _np.argmin(self.R[s, :])

        self._endRun()
        return policy

    def computeValue(self):
        # todo: implement this method

        # action
        # v = min(cost(i,a) + \vega \sigma^hat[s,a])

        value = -1

        return value

    def computeSigma(self):
        model = Model('SigmaIntervalMatrix')
        mu = model.addVar(vtype=GRB.CONTINUOUS, name="mu")
        objective = _np.add(
                        _np.multiply(
                            _np.transpose(
                                _np.subtract(
                                    self.p_upper,
                                    self.p_lower
                                )
                            ),
                            #_np.positive(
                            _np.subtract(
                                _np.multiply(
                                    mu,
                                    _np.ones(self.S, dtype=_np.int)
                                ),
                                self.v
                            )
                        ),
                        _np.add(
                            _np.multiply(
                                _np.transpose(self.v),
                                self.p_upper
                            ),
                            _np.multiply(
                                mu,
                                (
                                    1 - _np.multiply(
                                        _np.transpose(self.p_upper),
                                        _np.ones(self.S, dtype=_np.int)
                                    )
                                )
                            )
                        )
                    )

        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        for v in model.getVars():
            if v.X != 0:
                print("%s %f" % (v.Varname, v.X))
