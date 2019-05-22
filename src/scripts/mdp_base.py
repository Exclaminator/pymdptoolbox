from mdptoolbox.mdp import MDP
import numpy as np
"""
very simple MDP, just to check whether we are able to extend.
returns a random policy.
We should be able to extract a policy value.
"""


class RandomMdp(MDP):

    def __init__(self, transitions, reward, discount, epsilon,
                 max_iter, skip_check=False):
        # Initialise a value iteration MDP.

        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter,
                     skip_check=skip_check)

    def run(self):
        # Run the value iteration algorithm.
        # we should Initalize self.V and self.policy

        self._startRun()

        self.policy = np.random.randint(0, self.A, self.S)
        self.V = np.random.rand(self.S)

        self._endRun()


