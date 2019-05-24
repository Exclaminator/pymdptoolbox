from mdptoolbox.mdp import MDP, _printVerbosity
import mdptoolbox.util as _util

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp


class RobustPolicyImprovement(MDP):
    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=10, skip_check=False):
        MDP.__init__(self, transitions, reward, discount,epsilon, max_iter, skip_check)


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
                    value = self.reward[a, s] +  self.discount * self.sigma[s,a]
                    if self.v_next[s] > value:
                        self.v_next[s] = value

            self.v = self.v_next.copy()

            if _np.linalg.norm(self.v-self.v_next) < (1 - self.discount) * self.epsilon / (2.0 * self.discount):
                break
            if self.iter >= self.max_iter:
                break


        # TODO make policy

        self._endRun()

    def computeSigma(self):
        # TODO
        self.sigma = self.v.copt()
        pass