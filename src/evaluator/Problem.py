import numpy as _np
from scipy.stats import wasserstein_distance

import mdptoolbox.example
import scipy as sp
from evaluator.Evaluator import Sampling

"""
creates a problem from a dict.
Selects the corresponding type
"""


class Problem(object):
    """
    Creates a problem object. Uses the example module.
    We can initialize it based on some parameters.
    Next, we can extract P, P_var and R from it, which can be put into a MDP object.

    Types of ambiguities
    - interval based (p_low <= p <= p_up)
    - variance based (p = P +/- sqrt(p_var))
    - distance based (d(p,P) <= beta)
    """

    def __init__(self, transition_kernel, reward_matrix, discount_factor=0.9, name="undefined", distance=0):
        self.transition_kernel = transition_kernel
        self.reward_matrix = reward_matrix
        self.discount_factor = discount_factor
        self.name = name
        self.distance = distance

    def getName(self):
        return self.name

    def getProblemSet(self, options):
        variance = options.sample_var
        sample_amount = options.sample_amount
        ttk = self.transition_kernel

        # draw from normal
        # index 1 and 2 are kernel indices, 3 is the sample index
        non_normalized_tks = _np.random.normal(_np.repeat(ttk[:, :, :, _np.newaxis], sample_amount, axis=3),
                                               variance)
        problems_out = []
        for i in range(sample_amount):
            tk = self.normalize_tk(non_normalized_tks[:, :, :, i])
            distance = 0
            for a in range(self.transition_kernel.shape[0]):
                for s in range(self.transition_kernel.shape[1]):
                    distance += wasserstein_distance(tk[a][s], self.transition_kernel[a][s])
            new_problem = Problem(tk, self.reward_matrix, self.discount_factor, self.name, distance)
            # new_problem.transition_kernel = tk
            problems_out.append(new_problem)

        return ProblemSet(problems_out, self, options, Sampling.ALL)

    def computeMDP(self, mdp):
        policy = mdp.policy
        # P and R are A x S x S' shaped
        state_amount = len(policy)

        def compute_tk_policy(state):
            return self.transition_kernel[policy[state], state, :]

        def compute_rm_policy(state):
            return self.reward_matrix[policy[state], state, :]

        # hacky conversion using list (otherwise it will return non-numeric objects)
        tk_arr = _np.array(list(map(compute_tk_policy, range(state_amount))))
        rm_arr = _np.array(list(map(compute_rm_policy, range(state_amount))))

        rm_vector = _np.zeros(state_amount)
        for i in range(state_amount):
            for j in range(state_amount):
                rm_vector[i] += rm_arr[i][j] * tk_arr[i][j]

        V = _np.linalg.solve(
            sp.eye(state_amount) - self.discount_factor * tk_arr,
            rm_vector)

        return V[0]

    def simulateMDP(self, mdp, options):
        policy = mdp.policy

        results = []
        for i in range(options.number_of_sims):
            s_current = 0
            total_reward = 0

            for t in range(options.t_max):
                action = policy[s_current]
                tk_a = self.transition_kernel[action, s_current]
                s_new = _np.random.choice(a=len(tk_a), p=tk_a)
                # R is in format A x S x S'
                rm_3d = self.reward_matrix[:, s_current, s_new]
                total_reward += rm_3d[action] * _np.power(self.discount_factor, t)
                s_current = s_new

            results.append(total_reward)

        return _np.mean(results)

    @staticmethod
    def normalize_tk(tk_in):
        tk_in = _np.abs(tk_in)
        # todo: check if we want to do this, 1 is ensured by normaliztion,
        #  clipping it here will reduce its value in normaliztion
        # tk_in = _np.minimum(tk_in, 1)
        tk_in = _np.maximum(tk_in, 0)

        tk_out = _np.zeros(tk_in.shape)
        for i in range(tk_in.shape[0]):
            for ii in range(tk_in.shape[1]):
                tk_out[i, ii, :] = tk_in[i, ii, :] / _np.sum(tk_in[i, ii, :])
        return tk_out

    @staticmethod
    def get_forest_problem(S=10, discount_factor=0.9, r1=40, r2=20, p=0.05):
        tk, reward_matrix = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)

        # S x A -> A x S x S'
        reward_matrix = _np.transpose(reward_matrix)
        reward_matrix = _np.repeat(reward_matrix[:, :, _np.newaxis], reward_matrix.shape[1], axis=2)

        # create problem
        return Problem(tk, reward_matrix, discount_factor, "forest")

    @staticmethod
    def get_random_problem(state_amount=10, A=5, discount_factor=0.9):
        tk, reward_matrix = mdptoolbox.example.rand(state_amount, A, is_sparse=False)
        reward_matrix = _np.maximum(reward_matrix, 0)

        return Problem(tk, reward_matrix, discount_factor, "random")


class ProblemSet(object):
    """
    representation of a collection of problems.
    """
    def __init__(self, samples, problem, options, sampling=Sampling.ALL):
        self.samples = samples
        self.problem = problem
        self.options = options
        self.sampling = sampling
        self.resultsComputed = []
        self.resultsSimulated = []
        self.distances = []

        # for problem in self.samples[1:min(len(self.samples), self.options.number_of_paths)]:
        for problem in self.samples:
            self.distances.append(problem.distance)

    def split(self, mdp):
        # all_problems is a list instead of a ProblemSet object
        if hasattr(mdp, "innerfunction"):
            in_samples = ProblemSet(
                [x for x in self.samples if mdp.innerfunction.inSample(x.transition_kernel)],
                self.problem,
                self.options,
                Sampling.IN)
            out_samples = ProblemSet(
                [x for x in self.samples if not mdp.innerfunction.inSample(x.transition_kernel)],
                self.problem,
                self.options,
                Sampling.OUT)
            return in_samples, out_samples

        else:
            # non robust model
            return self, self

    def computeMDP(self, mdp):
        # for problem in self.samples[1:min(len(self.samples), self.options.number_of_paths)]:
        for problem in self.samples:
            self.resultsComputed.append(problem.computeMDP(mdp))
        return self.resultsComputed

    #  if self.options.do_simulation:
    def simulateMDP(self, mdp):
        # for problem in self.samples[1:min(len(self.samples), self.options.number_of_paths)]:
        for problem in self.samples:
            self.resultsSimulated.append(problem.simulateMDP(mdp, self.options))
        return self.resultsSimulated
