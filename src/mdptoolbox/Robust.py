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
                 max_iter=10, initial_value=0, beta = 0.1, skip_check=False):
        ValueIteration.__init__(self, transitions, reward, discount, epsilon, max_iter, initial_value, skip_check)

        # In the robust interval model, each p is given a lower and upper bound
        # TODO add errors for wrong upper and lower bounds
        if p_lower.shape == (self.S,):
            p_lower = _np.repeat([_np.repeat([p_lower], self.S, axis=0)], self.A, axis=0)
        if p_upper.shape == (self.S,):
            p_upper = _np.repeat([_np.repeat([p_upper], self.S, axis=0)], self.A, axis=0)

        assert p_lower.shape == (self.A, self.S, self.S), "p_lower must be in the shape A*S*S or S*1."
        assert p_upper.shape == (self.A, self.S, self.S), "p_upper must be in the shape A*S*S or S*1."

        self.p_lower = p_lower
        self.p_upper = p_upper
        self.beta = beta

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
                    self.sigma = self.computeSigmaElipsoidal(s, a)
                    # notify user
                    if self.verbose:
                        _printVerbosity(self.iter, self.sigma)
                    self.v_next[s] = max(self.v_next[s], self.R[a][s]+self.discount*self.sigma)

            # see if there is no more improvement
            if _np.linalg.norm(self.V - self.v_next) < (1 - self.discount) * self.epsilon / (2.0 * self.discount):
                self.V = self.v_next
                break

            self.V = self.v_next
            if self.iter >= self.max_iter:
                break



        # make policy
        self.policy = _np.zeros(self.S, dtype=_np.int)
        self.v_next = _np.full(self.V.shape, -_np.inf)
        for s in range(self.S):
            self.policy[s] = 0
            for a in range(self.A):
                self.sigma = self.computeSigmaDualReductionGreg2(s, a)
                v_a = self.R[a][s] + self.discount * self.sigma
                if v_a > self.v_next[s]:
                    self.v_next[s] = v_a
                    self.policy[s] = a

        #return policy
        self._endRun()

    def computeSigmaIntervalModel(self, state, action):
        model = Model('SigmaIntervalMatrix')
        mu = model.addVar(vtype=GRB.CONTINUOUS, name="mu")
        objective = LinExpr()
        objective += _np.dot(
                        _np.subtract(self.p_upper[action][state], self.p_lower[action][state]),
                        # use _np.maximum(v,0) to implement positive part of vector v
                        _np.maximum(
                            _np.subtract(_np.multiply(mu, _np.ones(self.S, dtype=_np.float)), self.V),
                            _np.zeros(self.S)))
        objective += _np.dot(self.V, self.p_lower[action][state])
        objective += _np.multiply(mu, (1 - _np.dot(self.p_lower[action][state], _np.ones(self.S, dtype=_np.float))))
        model.setObjective(objective, GRB.MINIMIZE)
        
        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal

    def computeSigmaDualReductionGreg2(self, state, action):
        model = Model('SigmaReductionGreg')
        mu = model.addVar(vtype=GRB.CONTINUOUS, name="mu")
        index = range(len(self.V))
        lu = model.addVars(index, name="lu", vtype=GRB.CONTINUOUS)
        ll = model.addVars(index, name="ll", vtype=GRB.CONTINUOUS)
        for i in index:
            model.addConstr(mu - lu[i] + ll[i] == self.V[i])
            model.addConstr(lu[i] >= 0)
            model.addConstr(ll[i] >= 0)

        objective = LinExpr()
        objective += mu

        for i in index:
            objective += -(self.p_upper[action][state][i] * lu[i])
            objective += (self.p_lower[action][state][i] * ll[i])

        model.setObjective(objective, GRB.MAXIMIZE)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        result =  model.objVal
        return model.objVal



    def computeSigmaDualReductionGreg(self, state, action):
        model = Model('SigmaReductionGreg')
        mu = model.addVar(vtype=GRB.CONTINUOUS, name="mu")
        index = range(len(self.V))
        z = model.addVars(index, name="z", vtype=GRB.CONTINUOUS)
        zt = model.addVars(index, name="zt", vtype=GRB.CONTINUOUS)
        for i in index:
            model.addConstr(zt[i] == mu - self.V[i])
            model.addConstr(z[i] == max_(zt[i], 0))


        objective = LinExpr()
        objective += mu
        #objective += -_np.dot(
        #                _np.subtract(self.p_upper[action][state], self.p_lower[action][state]),
        #                _np.maximum(
        #                    _np.subtract(_np.multiply(mu, _np.ones(self.S, dtype=_np.float)), self.V),
        #                    _np.zeros(self.S)))
        for i in index:
            objective += -(self.p_upper[action][state][i] - self.p_lower[action][state][i]) * z[i]

        for i in index:
            objective += -self.p_lower[action][state][i] * (mu - self.V[i])
        #objective += -_np.dot(
        #                self.p_lower[action][state],
        #                _np.subtract(_np.multiply(mu, _np.ones(self.S, dtype=_np.float)), self.V)
        #            )

        model.setObjective(objective, GRB.MAXIMIZE)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal

    # Chi squared distance
    def computeSigmaElipsoidal(self, state, action):
        model = Model('SigmaElipsoidal')
        pGurobi = model.addVars(self.S, vtype=GRB.CONTINUOUS, name="p")
        p = _np.transpose(_np.array(pGurobi.items()))[1]
        # objective = LinExpr()
        # for key, val in p.items():
        #     objective += val*self.V[key]
        objective = LinExpr()
        objective += _np.dot(p, self.V)
        model.setObjective(objective, GRB.MAXIMIZE)
        model.addConstr(_np.sum(
                _np.divide(
                    _np.multiply(
                        _np.subtract(p, self.P[action][state]),
                        _np.subtract(p, self.P[action][state])),
                    self.P[action][state] + sys.float_info.epsilon
                )) <= self.beta)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal


    # non linear model, cannot be solved by Gurobi
    def computeSigmaMaximumLikelihoodModel(self, state, action):
        model = Model('SigmaMaximumLikelihood')
        mu = model.addVar(vtype=GRB.CONTINUOUS, name="mu")
        lmbda = (mu - _np.sum(self.V)) / _np.sum(self.P[action][state])
        objective = LinExpr()
        objective += mu
        objective += - (1 + self.beta)*lmbda
        for j in range(self.S):
            objective += lmbda * self.P[action][state][j] * _np.log(lmbda*self.P[action][state][j] / (mu - self.V[j]))

        model.addConstr(lmbda > 0, name='Lambda greater than zero')
        model.addConstr(mu >= _np.sum(self.V), name='Mu greater equal to v_max')

        model.setObjective(objective, GRB.MAXIMIZE)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal
