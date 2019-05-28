from mdptoolbox.mdp import ValueIteration, _printVerbosity
from gurobipy import *
import numpy as _np

class RobustIntervalModel(ValueIteration):
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
    p_lower : array
        Lowerbound on transition probabilty matrix.
    p_upper : array
        Upperbound on transition probabilty matrix.
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
    # >>> import mdptoolbox, mdptoolbox.example
    # >>> P, R = mdptoolbox.example.forest()
    # >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
    # >>> vi.verbose
    # False
    # >>> vi.run()
    # >>> expected = (5.93215488, 9.38815488, 13.38815488)
    # >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    # True
    # >>> vi.policy
    # (0, 0, 0)
    # >>> vi.iter
    # 4
    #
    # >>> import mdptoolbox
    # >>> import numpy as np
    # >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    # >>> R = np.array([[5, 10], [-1, 2]])
    # >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    # >>> vi.run()
    # >>> expected = (40.048625392716815, 33.65371175967546)
    # >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    # True
    # >>> vi.policy
    # (1, 0)
    # >>> vi.iter
    # 26
    #
    # >>> import mdptoolbox
    # >>> import numpy as np
    # >>> from scipy.sparse import csr_matrix as sparse
    # >>> P = [None] * 2
    # >>> P[0] = sparse([[0.5, 0.5],[0.8, 0.2]])
    # >>> P[1] = sparse([[0, 1],[0.1, 0.9]])
    # >>> R = np.array([[5, 10], [-1, 2]])
    # >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    # >>> vi.run()
    # >>> expected = (40.048625392716815, 33.65371175967546)
    # >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    # True
    # >>> vi.policy
    # (1, 0)

    """

    def __init__(self, transitions, reward, discount, p_lower, p_upper, epsilon=0.01,
                 max_iter=10, initial_value=0, skip_check=False):
        ValueIteration.__init__(self, transitions, reward, discount, epsilon, max_iter, initial_value, skip_check)

        # In the robust interval model, each p is given a lower and upper bound
        # TODO add errors for wrong upper and lower bounds
        self.p_lower = p_lower
        self.p_upper = p_upper

    def run(self):
        # Run the modified policy iteration algorithm.
        self._startRun()
        # TODO perhaps there can be a better initial guess. (v > 0)
        self.V = _np.ones(self.S)
        self.sigma = 0

        # Itterate
        while True:
            self.iter += 1
            self.sigma = self.computeSigma()
            self.v_next = _np.full(self.V.shape, _np.inf)

            # update value
            for s in range(self.S):
                self.v_next[s] = _np.min(_np.transpose(self.R)[s])+self.discount*self.sigma

            # see if there is no more improvement
            if _np.linalg.norm(self.V - self.v_next) < (1 - self.discount) * self.epsilon / (2.0 * self.discount):
                break
            if self.iter >= self.max_iter:
                break

            # update V
            self.V = self.v_next

            # notify user
            if self.verbose:
                _printVerbosity(self.iter, self.V)

        # make policy
        self.policy = _np.zeros(self.S, dtype=_np.int)
        for s in range(self.S):
            self.policy[s] = _np.argmin(_np.transpose(self.R)[s])

        #return policy
        self._endRun()

    def computeSigma(self):
        model = Model('SigmaIntervalMatrix')
        mu = model.addVar(vtype=GRB.CONTINUOUS, name="mu")
        objective = LinExpr()
        objective += _np.dot(
                        _np.subtract(self.p_upper, self.p_lower),
                        _np.maximum(
                            _np.subtract(
                                _np.multiply(mu, _np.ones(self.S, dtype=_np.float)),
                                self.V),
                            _np.zeros(self.S)))
        objective += _np.dot(self.V, self.p_upper)
        objective += _np.multiply(mu, (1 - _np.dot(self.p_upper,_np.ones(self.S, dtype=_np.float))))
        model.setObjective(objective, GRB.MINIMIZE)
        
        # stay silent if requested
        if (not self.verbose):
            model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal
