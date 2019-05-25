from mdptoolbox.mdp import MDP, _printVerbosity
import mdptoolbox.util as _util

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp


class RobustIntervalModel(MDP):
    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=10, skip_check=False, P_interval):
        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter, skip_check)
        # In the robust interval model, each p is given a lower and upper bound
        self.P_interval = P_interval


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
                    value = self.reward[a, s] + self.discount * self.sigma[s,a]
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
        # TODO: implement this method
        # n <- amount of states
        # T <- time horizon
        #
        # decision variables :
        # mu: policy (one action per time horizon, thus t_max amound of actions) T x 1
        #
        # hyperparameters :
        # p_up, p_low: (nonnegative n-vectors, so per state one value) S x 1
        # v_t(i): (worst case optimal value function in state i at stage t)
        #
        # the formula is
        # sigma = minimize{
        #   transpose(p_up - p_low) x positive_part(mu x 1-vector - v)
        #   + transpose(v) x p_up
        #   + mu (1 - transpose(p_up) x 1-vector)
        # }
        ## dimension analysis, todo: get rid of the inconsistencies
        # sigma has one value per state-action combination: S x A (or A x S ?)
        # v has a value per state: S x 1
        # (a) transpose(p_up - p_low) -> 1 x S
        # (b) positive_part(mu x 1-vector - v) -> T x 1 * S x 1 = ??? but probably S x T (not sure how mu * 1-vector works)
        # (a) x (b) -> 1 x S * S * T -> 1 x T
        # transpose(v) x p_up -> 1 x S * S x 1 -> 1 x 1
        # mu (1 - transpose(p_up) x 1-vector) ->  T x 1 * (1 - 1 x S * S x 1) = T x 1 * 1 x 1 = T x 1
        # sigma = 1 x T + 1 x 1 + 1 x T = 1 x T <- (not S x A or A x S)
        
        self.sigma = self.v.copt()
        pass
