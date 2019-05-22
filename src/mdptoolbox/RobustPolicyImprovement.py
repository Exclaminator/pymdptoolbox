from mdptoolbox.mdp import PolicyIteration, _printVerbosity
import mdptoolbox.util as _util

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp


class RobustPolicyImprovement(PolicyIteration):

    """A MDP solved using the Robust Policy Improvement algorithm.

   Parameters
   ----------
   transitions : array
       Transition probability matrices. See the documentation for the ``MDP``
       class for details.
   reward : array
       Reward matrices or vectors. See the documentation for the ``MDP`` class
       for details.
   discount : float
       Discount factor. See the documentation for the ``MDP`` class for
       details.
   ambiguitySet : array
       Ambiguity parameter Xi
   max_iter : int, optional
       Maximum number of iterations. See the documentation for the ``MDP``
       class for details. Default is 1000.
   skip_check : bool
       By default we run a check on the ``transitions`` and ``rewards``
       arguments to make sure they describe a valid MDP. You can set this
       argument to True in order to skip this check.

   Data Attributes
   ---------------
   V : array
       Optimal value function. Shape = (S, N+1). ``V[:, n]`` = optimal value
       function at stage ``n`` with stage in {0, 1...N-1}. ``V[:, N]`` value
       function for terminal stage.
   policy : array
       Optimal policy. ``policy[:, n]`` = optimal policy at stage ``n`` with
       stage in {0, 1...N}. ``policy[:, N]`` = policy for stage ``N``.
   time : float
       used CPU time

   Notes
   -----
   In verbose mode, displays the current stage and policy transpose.

   Examples
   --------
   %>>> import mdptoolbox, mdptoolbox.example
   %>>> P, R = mdptoolbox.example.forest()
   %>>> fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.9, 3)
   %>>> fh.run()
   %>>> fh.V
   %array([[2.6973, 0.81  , 0.    , 0.    ],
   %       [5.9373, 3.24  , 1.    , 0.    ],
   %       [9.9373, 7.24  , 4.    , 0.    ]])
   %>>> fh.policy
   %array([[0, 0, 0],
   %       [0, 0, 1],
   %       [0, 0, 0]])

   Translation between paper notation and code
   --------
   %k_{sa} \in R^S = vector of transition probabilities to s' from state s with action a
   %K_{sa} \in R^{S*q}

   %O_l \in S^q, O_l is PSD

   %\xi \in R^q
   %\Xi = {\xi: \xi^T O_l \xi + o_l^T \xi + \omega \geq 0}
   """

    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=10, skip_check=False):
        # Initialise a (modified) policy iteration MDP.

        # Maybe its better not to subclass from PolicyIteration, because the
        # initialisation of the two are quite different. eg there is policy0
        # being calculated here which doesn't need to be. The only thing that
        # is needed from the PolicyIteration class is the _evalPolicyIterative
        # function. Perhaps there is a better way to do it?
        PolicyIteration.__init__(self, transitions, reward, discount, None,
                                 max_iter, 1, skip_check=skip_check)

        # PolicyIteration doesn't pass epsilon to MDP.__init__() so we will
        # check it here
        self.epsilon = float(epsilon)
        assert epsilon > 0, "'epsilon' must be greater than 0."

        # computation of threshold of variation for V for an epsilon-optimal
        # policy
        if self.discount != 1:
            self.thresh = self.epsilon * (1 - self.discount) / self.discount
        else:
            self.thresh = self.epsilon

        if self.discount == 1:
            self.V = _np.zeros(self.S)
        else:
            Rmin = min(R.min() for R in self.R)
            self.V = 1 / (1 - self.discount) * Rmin * _np.ones((self.S,))

    def run(self):
        # Run the modified policy iteration algorithm.

        self._startRun()

        while True:
            self.iter += 1

            self.policy, Vnext = self._bellmanOperator()
            # [Ppolicy, PRpolicy] = mdp_computePpolicyPRpolicy(P, PR, policy);

            variation = _util.getSpan(Vnext - self.V)
            if self.verbose:
                _printVerbosity(self.iter, variation)

            self.V = Vnext
            if variation < self.thresh:
                break
            else:
                is_verbose = False
                if self.verbose:
                    self.setSilent()
                    is_verbose = True

                self._evalPolicyIterative(self.V, self.epsilon, self.max_iter)

                if is_verbose:
                    self.setVerbose()

        self._endRun()