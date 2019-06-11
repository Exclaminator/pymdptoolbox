import mdptoolbox.example
from mdptoolbox.mdp import ValueIteration, _printVerbosity
from gurobipy import *
import numpy as _np
from decimal import *

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
    def __init__(self, transitions, reward, discount, innerfunction, epsilon=0.01, max_iter=1000, initial_value=0, skip_check=False):
        ValueIteration.__init__(self, transitions, reward, discount, epsilon, max_iter, initial_value, skip_check)

        # bind context of inner function and make it accessable
        self.innerfunction = innerfunction(self)

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
                    self.sigma = self.innerfunction(s, a)
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
                self.sigma = self.innerfunction(s, a)
                v_a = self.R[a][s] + self.discount * self.sigma
                if v_a > v_next[s]:
                    v_next[s] = v_a
                    self.policy[s] = a

        #return policy
        self._endRun()

    class innerMethod:
        def Interval(p_upper, p_lower):
            def innerInterval(self):
                if p_lower.shape == (self.S,):
                    self.p_lower = _np.repeat([_np.repeat([p_lower], self.S, axis=0)], self.A, axis=0)
                if p_upper.shape == (self.S,):
                    self.p_upper = _np.repeat([_np.repeat([p_upper], self.S, axis=0)], self.A, axis=0)

                assert self.p_lower.shape == (self.A, self.S, self.S), "p_lower must be in the shape A*S*S or S*1."
                assert self.p_upper.shape == (self.A, self.S, self.S), "p_upper must be in the shape A*S*S or S*1."

                self.p_lower = _np.maximum(self.p_lower, 0)
                self.p_upper = _np.minimum(self.p_upper, 1)

                def IntervalModel(state, action):
                    model = Model('IntervalModel')
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
                    return model.objVa
                return IntervalModel
            return innerInterval

        # Chi squared distance
        def Elipsoid(beta):
            def innerElipsoid(self):
                def ElipsoidModel(state, action):
                    model = Model('ElipsoidModel')
                    pGurobi = model.addVars(self.S, vtype=GRB.CONTINUOUS, name="p")
                    p = _np.transpose(_np.array(pGurobi.items()))[1]
                    objective = LinExpr()
                    objective += _np.dot(p, self.V)
                    model.setObjective(objective, GRB.MINIMIZE)
                    model.addConstr(_np.sum(
                            _np.divide(
                                _np.multiply(
                                    _np.subtract(p, self.P[action][state]),
                                    _np.subtract(p, self.P[action][state])),
                                self.P[action][state] + sys.float_info.epsilon
                            )) <= beta)

                    # stay silent
                    model.setParam('OutputFlag', 0)

                    model.optimize()
                    return model.objVal
                return ElipsoidModel
            return innerElipsoid

        # Wasserstein
        def Wasserstein(beta):
            def innerWasserstein(self):
                def EMD(state, action):
                    model = Model('SigmaEMD')
                    pGurobi = model.addVars(self.S, vtype=GRB.CONTINUOUS, name="p")
                    p = _np.transpose(_np.array(pGurobi.items()))[1]
                    emdGurobi = model.addVars(self.S, vtype=GRB.CONTINUOUS, name="emd")
                    emd = _np.transpose(_np.array(emdGurobi.items()))[1]
                    emdAbsGurobi = model.addVars(self.S, vtype=GRB.CONTINUOUS, name="emd_abs")
                    emdAbs = _np.transpose(_np.array(emdAbsGurobi.items()))[1]
                    objective = LinExpr()
                    objective += _np.dot(p, self.V)
                    model.setObjective(objective, GRB.MINIMIZE)
                    for i in range(self.S):
                        if i == 0:
                            model.addConstr(emd[i] == p[i] - self.P[action][state][i])
                        else:
                            model.addConstr(emd[i] == p[i] - self.P[action][state][i] + emd[i-1])
                        model.addConstr(emd[i] <= emdAbs[i])
                        model.addConstr(-emd[i] <= emdAbs[i])
                    model.addConstr((-_np.sum(emdAbs)) <= beta)
                    model.addConstr(_np.sum(emdAbs) <= beta)

                    # stay silent
                    model.setParam('OutputFlag', 0)

                    model.optimize()
                    return model.objVal
                return EMD
            return innerWasserstein

        # Log likelihood model
        def Likelihood(beta, delta):
            def innerLikelihood(self):
                self.bMax = _np.zeros(self.A)
                for a in range(self.A):
                    for i in range(self.S):
                        for j in range(self.S):
                            self.bMax[a] -= self.P[a][i][j]*math.log(self.P[a][i][j] + sys.float_info.epsilon)

                if beta > _np.max(self.bMax):
                    print("Beta will be cut of to " + str(_np.max(self.bMax)))
                self.beta = _np.minimum(beta, _np.max(self.bMax))

                def computeSigmaMaximumLikelihoodModel(state, action):
                    mu_lower = _np.max(self.V)
                    e_factor = math.pow(math.e, self.beta - self.bMax[action]) - sys.float_info.epsilon
                    mu_upper = (_np.max(self.V) - e_factor*_np.average(self.V)) / (1 - e_factor)
                    mu = (mu_upper + mu_lower)/2
                    while (mu_upper - mu_lower) > delta*(1+2*mu_lower):
                        mu = (mu_upper + mu_lower)/2
                        if derivativeOfSigmaLikelyhoodModel(mu, state, action) < 0:
                            mu_upper = mu
                        else:
                            mu_lower = mu
                    lmbda = lambdaLikelyhoodModel(mu, state, action)
                    if _np.abs(lmbda - sys.float_info.epsilon) <= sys.float_info.epsilon:
                        return mu
                    return mu - (1 + self.beta)*lmbda + lmbda*_np.sum(
                        _np.multiply(
                            self.P[action][state],
                            _np.log(sys.float_info.epsilon + _np.divide(
                                    lambdaLikelyhoodModel(mu, state, action)*self.P[action][state],
                                    _np.subtract(_np.repeat(mu, self.S), self.V)))))

                def derivativeOfSigmaLikelyhoodModel(mu, state, action):
                    dsigma = - self.beta + _np.sum(
                        _np.multiply(
                            self.P[action][state],
                            _np.log(
                                sys.float_info.epsilon +
                                _np.divide(
                                    lambdaLikelyhoodModel(mu, state, action)*self.P[action][state],
                                    _np.subtract(_np.repeat(mu, self.S), self.V)+ sys.float_info.epsilon))))
                    dsigma *= _np.sum(_np.divide(self.P[action][state], _np.power(mu * _np.ones(self.S) - self.V, 2)))
                    dsigma /= math.pow(_np.sum(_np.divide(self.P[action][state], mu * _np.ones(self.S) - self.V)), 2)
                    return dsigma

                def lambdaLikelyhoodModel(mu, state, action):
                    return 1 / _np.sum(_np.divide(self.P[action][state], mu*_np.ones(self.S) - self.V + sys.float_info.epsilon))

                return computeSigmaMaximumLikelihoodModel
            return innerLikelihood

    
# run robust
if __name__ == "__main__":
    P, R = mdptoolbox.example.forest()
    s = RobustModel(P, R, 0.9, RobustModel.innerMethod.Elipsoid(23))
    s.run()
    print(s.policy)
