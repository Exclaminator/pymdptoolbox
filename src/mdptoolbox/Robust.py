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

        while True:
            self.iter += 1

            # TODO not sure if this works
            self.v_next = _np.full(self.v.shape, _np.inf)
            self.computeSigma()
            for s in range(self.S):
                for a in range(self.A):
                    # TODO not sure about dimension of reward
                    # Sy: reward corresponds with the cost function
                    value = self.computeValue()
                    #value = self.reward[a, s] + self.discount * self.sigma[s,a]
                    if self.v_next[s] > value:
                        self.v_next[s] = value

            self.v = self.v_next.copy()

            if _np.linalg.norm(self.v-self.v_next) < (1 - self.discount) * self.epsilon / (2.0 * self.discount):
                break
            if self.iter >= self.max_iter:
                break


        # TODO make policy

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
        objective = _np.add(
                        _np.multiply(
                            _np.transpose(_np.subtract(self.p_upper, self.p_lower)),
                            _np.positive(
                                _np.subtract(
                                    _np.multiply(mu, _np.ones(self.S)),
                                    self.v
                        ))),
                        _np.add(
                            _np.multiply(_np.transpose(self.v), self.p_upper),
                            mu * (1 - _np.multiply(_np.transpose(self.p_upper), _np.ones(self.S)))
                        )
                    )

        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()
        
        for v in model.getVars():
            if v.X != 0:
                print("%s %f" % (v.Varname, v.X))
