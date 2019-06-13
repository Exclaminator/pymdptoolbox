import mdptoolbox.example
from mdptoolbox.Wasserstein import Wasserstein
from mdptoolbox.mdp import ValueIteration
import numpy as _np


def Robust(innerfunction):
    class RobustModel(ValueIteration):
        """A discounted Robust MDP solved using the robust interval model.

        Description
        -----------
        RobustIntervalModel applies the robust interval model to solve a
        discounted RMDP. The algorithm consists of solving linear programs
        iteratively.
        Iteration is stopped when an epsilon-optimal policy is found or after a
        specified number (``max_iter``) of iterations.
        This function uses verbose and silent modes. In verbose mode, the function
        displays the variation of ``V`` (the value function) for each iteration and
        the condition which stopped the iteration: epsilon-policy found or maximum
        number of iterations reached.

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
        innerfunction : innerfunction
            Determines the ambiguity set. Can be found inside RobustModel.innerMethod,
            avaliable: Interval, Elipsoid, Wasserstein, Likelihood
        epsilon : float, optional
            Stopping criterion. See the documentation for the ``MDP`` class for
            details.  Default: 0.01.
        max_iter : int, optional
            Maximum number of iterations. If the value given is greater than a
            computed bound, a warning informs that the computed bound will be used
            instead. By default, if ``discount`` is not equal to 1, a bound for
            ``max_iter`` is computed, otherwise ``max_iter`` = 1000. See the
            documentation for the ``MDP`` class for further details.
        initial_value : array, optional
            The starting value function. Default: a vector of zeros.
        skip_check : bool
            By default we run a check on the ``transitions`` and ``rewards``
            arguments to make sure they describe a valid MDP. You can set this
            argument to True in order to skip this check.

        Data Attributes
        ---------------
        V : tuple
            The optimal value function.
        policy : tuple
            The optimal policy function. Each element is an integer corresponding
            to an action which maximises the value function in that state.
        iter : int
            The number of iterations taken to complete the computation.
        time : float
            The amount of CPU time used to run the algorithm.

        Methods
        -------
        run()
            Do the algorithm iteration.
        setSilent()
            Sets the instance to silent mode.
        setVerbose()
            Sets the instance to verbose mode.

        Notes
        -----
        In verbose mode, at each iteration, displays the variation of V
        and the condition which stopped iterations: epsilon-optimum policy found
        or maximum number of iterations reached.

        Examples
        --------
        >>> import mdptoolbox, mdptoolbox.example
        >>> P, R = mdptoolbox.example.forest()
        >>> vi = mdptoolbox.Robust.RobustModel(P, R, 0.96, mdptoolbox.Robust.RobustModel.innerMethod.Elipsoid(1.5))
        >>> vi.verbose
        False
        >>> vi.run()
        Academic license - for non-commercial use only
        >>> expected = (5.573706829021013e-08, 1.0000000000592792, 4.000000222896506)
        >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
        True
        >>> vi.policy
        (0, 1, 0)
        >>> vi.iter
        2
        """
        def __init__(self, true_transition_kernel, reward, discount, epsilon=0.01, max_iter=1000, initial_value=0,
                     skip_check=False):
            # call parent constructor
            ValueIteration.__init__(self, true_transition_kernel, reward, discount, epsilon, max_iter, initial_value, skip_check)
            innerfunction.attachProblem(self)
            # bind context of inner function and make it accessable
            self.innerfunction = innerfunction

        def run(self):
            # Run the modified policy iteration algorithm.
            self._startRun()
            # TODO perhaps there can be a better initial guess. (v > 0)
            self.V = _np.ones(self.S)
            self.sigma = 0

            # Itterate
            while True:
                self.iter += 1
                self.v_next = _np.full(self.V.shape, -_np.inf)

                # update value
                for s in range(self.S):
                    for a in range(self.A):
                        self.sigma = self.innerfunction.run(s, a)
                        self.v_next[s] = max(self.v_next[s], self.R[a][s]+self.discount*self.sigma)
                if self.verbose:
                    print("iter {}/{}".format(self.iter, self.max_iter))
                # see if there is no more improvement
                if _np.linalg.norm(self.V - self.v_next) < (1 - self.discount) * self.epsilon / (2.0 * self.discount):
                    self.V = self.v_next
                    break

                self.V = self.v_next
                if self.iter >= self.max_iter:
                    break

            # make policy
            self.policy = _np.zeros(self.S, dtype=_np.int)
            v_next = _np.full(self.V.shape, -_np.inf)
            for s in range(self.S):
                self.policy[s] = 0
                for a in range(self.A):
                    # choose a corresponding sigma
                    self.sigma = self.innerfunction.run(s, a)
                    v_a = self.R[a][s] + self.discount * self.sigma
                    if v_a > v_next[s]:
                        v_next[s] = v_a
                        self.policy[s] = a
            #return policy
            self._endRun()
    return RobustModel


if __name__ == "__main__":
    P, R = mdptoolbox.example.forest()
    m = Robust(Wasserstein(0.12))(P, R, 0.94);
    m.run()
    print(m.policy)
    print(m.V)

    # example of how to use it:
    # mdp_list = {
    #     "wasserstein b=0.12": Robust(Wasserstein(0.12)),
    # }