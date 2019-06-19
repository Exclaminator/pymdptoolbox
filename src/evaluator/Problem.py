import numpy as _np
import mdptoolbox.example


"""
creates a problem from a dict.
Selects the corresponding type
"""


class Problem(object):

    discount_def = 0.9

    """
    Creates a problem object. Uses the example module.
    We can initialize it based on some parameters.
    Next, we can extract P, P_var and R from it, which can be put into a MDP object.

    Types of ambiguities
    - interval based (p_low <= p <= p_up)
    - variance based (p = P +/- sqrt(p_var))
    - distance based (d(p,P) <= beta)
    """

    def __init__(self, transition_kernel, reward_matrix, discount_factor=discount_def, name="undefined"):
        self.transition_kernel = transition_kernel
        self.reward_matrix = reward_matrix
        self.discount_factor = discount_factor
        self.name = name

    def getName(self):
        return self.name

    def getProblemSet(self, options):
        

    @staticmethod
    def get_forest_problem(S=10, discount_factor=discount_def, r1=40, r2=20, p=0.05):
        tk, reward_matrix = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)

        # S x A -> A x S x S'
        reward_matrix = _np.transpose(reward_matrix)
        reward_matrix = _np.repeat(reward_matrix[:, :, _np.newaxis], reward_matrix.shape[1], axis=2)

        # create problem
        return Problem(tk, reward_matrix, discount_factor, "forest")

    @staticmethod
    def get_random_problem(state_amount=10, A=5, discount_factor=discount_def):
        tk, reward_matrix = mdptoolbox.example.rand(state_amount, A, is_sparse=False)
        reward_matrix = _np.maximum(reward_matrix, 0)

        return Problem(tk, reward_matrix, discount_factor, "random")


class ProblemSet(object):
    """
    representation of a collection of problems.
    """
    def __init__(self, samples, problem, options):
        self.samples = samples
        self.problem = problem
        self.options = options

    def filter(self, mdp):
        # all_problems is a list instead of a ProblemSet object
        if hasattr(mdp, "innerfunction"):
            return ProblemSet(
                [x for x in self.samples if mdp.innerfunction.inSample(x.transition_kernel)],
                self.problem,
                self.options)
        else:
            # non robust model
            return self

    def evaluate(self, mdp):
        # store results
        result = {}
        filter_ratio = 0

        if self.options.evaluate_all:
            result[ALL_KEY, COMPUTED_KEY], result[ALL_KEY, SIMULATED_KEY] = \
                self.evaluate_policy_on_problem_list(mdp.policy, self.samples)

        if self.options.evaluate_inner:
            inner_samples = self.filter(mdp)

            # store the ratio of filtered samples in results
            # result[INNER_KEY, FILTER_RATIO_KEY]\
            filter_ratio = len(inner_samples) / len(self.samples)

            result[INNER_KEY, COMPUTED_KEY], result[INNER_KEY, SIMULATED_KEY] = \
                self.evaluate_policy_on_problem_list(mdp.policy)

        # maybe only the outer samples are interesting.
        # If you think so, uncomment and implement something for _np.difference that works
        # if self.options.get(Options.EVALUATE_OUTER):
        #     outer_samples = _np.difference(self.samples, inner_samples)
        #     result[OUTER_KEY, COMPUTED_KEY], result[OUTER_KEY, SIMULATED_KEY] = \
        #         self.evaluate_policy_on_problem_list(policy, outer_samples)

        return result, filter_ratio

    def evaluate_policy_on_problem_list(self, policy, problem_list):
        # limit on the number of paths
        number_of_paths = self.options.number_of_paths
        if len(problem_list) > number_of_paths:
            problem_list = problem_list[0:number_of_paths]
        else:
            warnings.warn(
                "number_of_paths ({}) is larger than the number of filtered policies ({})".format(number_of_paths,
                                                                                                  len(
                                                                                                      problem_list)))

        # use problem set to filter all problems
        results_computed = []
        results_simulated = []

        for problem in problem_list:
            # do this both for simulation and computation
            results_computed.append(self.compute_policy_on_problem(policy, problem))
            if self.options.do_simulation:
                results_simulated.append(self.simulate_policy_on_problem(policy, problem))

        return results_computed, results_simulated

    @staticmethod
    def compute_policy_on_problem(policy, problem):
        reward_matrix = problem.reward_matrix
        transition_kernel = problem.transition_kernel

        # P and R are A x S x S' shaped
        state_amount = len(policy)
        discount_factor = problem.discount_factor

        def compute_tk_policy(state):
            return transition_kernel[policy[state], state, :]

        def compute_rm_policy(state):
            return reward_matrix[policy[state], state, :]

        # hacky conversion using list (otherwise it will return non-numeric objects)
        tk_arr = _np.array(list(map(compute_tk_policy, range(state_amount))))
        rm_arr = _np.array(list(map(compute_rm_policy, range(state_amount))))

        rm_vector = _np.zeros(state_amount)
        for i in range(state_amount):
            for j in range(state_amount):
                rm_vector[i] += rm_arr[i][j] * tk_arr[i][j]

        V = _np.linalg.solve(
            sp.eye(state_amount) - discount_factor * tk_arr,
            rm_vector)

        return V[0]

    def simulate_policy_on_problem(self, policy, problem):
        reward_matrix = problem.reward_matrix
        discount_factor = problem.discount_factor
        tk = problem.transition_kernel

        results = []
        for i in range(self.options.number_of_sims):
            s_current = 0
            total_reward = 0

            for t in range(self.options.t_max):
                action = policy[s_current]
                tk_a = tk[action, s_current]
                s_new = _np.random.choice(a=len(tk_a), p=tk_a)
                # R is in format A x S x S'
                rm_3d = reward_matrix[:, s_current, s_new]
                total_reward += rm_3d[action] * _np.power(discount_factor, t)
                s_current = s_new

            results.append(total_reward)

        return _np.mean(results)
