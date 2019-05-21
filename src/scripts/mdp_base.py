from mdptoolbox.mdp import MDP

"""
very simple MDP, just to check whether we can run it
"""


class CustomMdpSkeleton(MDP):

    def __init__(self, transitions, reward, discount, epsilon,
                 max_iter, initial_value, skip_check=False):
        # Initialise a value iteration MDP.

        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter,
                     skip_check=skip_check)

        # todo: add initialization

    def run(self):
        # Run the value iteration algorithm.
        self._startRun()

        # todo: add method for running

        self._endRun()


