from mdptoolbox.mdp import MDP, ValueIteration
import numpy as np
from gurobipy.gurobipy import *
"""
very simple MDP, just to check whether we are able to extend.
returns a random policy.
We should be able to extract a policy value.
"""




class RectangularMdp(ValueIteration):

    def __init__(self, transitions, k, reward, discount, epsilon=0.01,
                 max_iter=1000, initial_value=0, skip_check=False):
        # Initialise a value iteration MDP.
        ValueIteration.__init__(self, transitions, reward, discount, epsilon,
                 max_iter, initial_value, skip_check)
        # TODO dim checks vor KM kv ....
        self.k = k
        self.q = np.shape(k)[1]
        self.L = 10 #todo

    def run(self):
        # Run the value iteration algorithm.
        # we should Initalize self.V and self.policy

        self._startRun()

     #   self.policy = np.random.randint(0, self.A, self.S)
     #   self.V = np.random.rand(self.S)

        self._endRun()

    def computeW(self):
        w = np.zeros(self.S)
        for i in range(1, 10):
            wnext = w
            for s in range(self.S):
                wnext[s] = self.computeNextW(w)
            w = wnext
        return w

    def computeNextW(self, w, s):
        m = Model("mip1")
        t = m.addVar(1, name="t")
        p = m.addVar(self.actions, name="p")
        Y = m.addVar((self.q, self.y), name="Y")
        z = m.addVar(self.y, name="z")
        t = m.addVar(self.y, name="t")

        m.addConstr(p.sum() == 1, "c27c1")
        m.addConstr(p.min() >= 1, "c27c2")
        m.addConstr(t - quicksum([t[a] * np.transpose(self.transitions[s,a]) * (self.r[s,a] + self.discount * w) for a in range(self.A)])
                    <= 
                    - quicksum([(1 - w[l] / 2 * z[l] + (w[l] + 1 / 2 * t[l])) for l in 1..L])
                    , "c27b1")

      # todo what is O?  m.addConstr(                  , "c27b2")


        # Set objective
        m.setObjective(t, GRB.MAXIMIZE)

        m.optimize()

        return t # todo is this correct?

    def _bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.S,), (1, self.S)), "V is not the " \
                    "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = np.empty((self.A, self.S))
        for aa in range(self.A):
            Q[aa] = self.R[aa] + self.discount * self.P[aa].dot(V)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        return (Q.argmax(axis=0), Q.max(axis=0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)


