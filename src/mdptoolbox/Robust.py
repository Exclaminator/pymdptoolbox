from abc import abstractmethod
from gurobipy import *
import mdptoolbox.example
from mdptoolbox.mdp import ValueIteration
from numpy import *
from scipy.stats import wasserstein_distance


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
            self.v_next = full(self.V.shape, -inf)
            self.sigma = 0

        def getName(self):
            return self.innerfunction.getName()

        def getInnerfunction(self):
            return self.innerfunction

        def run(self):
            # Run the modified policy iteration algorithm.
            self._startRun()

            # Itterate
            while True:
                self.iter += 1

                # update value
                for s in range(self.S):
                    for a in range(self.A):
                        self.sigma = self.innerfunction.run(s, a)
                        self.v_next[s] = max(self.v_next[s], self.R[a][s]+self.discount*self.sigma)
                if self.verbose:
                    print("iter {}/{}".format(self.iter, self.max_iter))
                # see if there is no more improvement
                if linalg.norm(self.V - self.v_next) < (1 - self.discount) * self.epsilon / (2.0 * self.discount):
                    self.V = self.v_next
                    break

                self.V = self.v_next
                if self.iter >= self.max_iter:
                    break

            # make policy
            self.policy = zeros(self.S, dtype=int)
            v_next = full(self.V.shape, -inf)
            for s in range(self.S):
                self.policy[s] = 0
                for a in range(self.A):
                    # choose a corresponding sigma
                    self.sigma = self.innerfunction.run(s, a)
                    v_a = self.R[a][s] + self.discount * self.sigma
                    if v_a > v_next[s]:
                        v_next[s] = v_a
                        self.policy[s] = a
            # return policy
            self._endRun()
    return RobustModel


class InnerMethod:
    def __init__(self):
        self.problem = None

    def attachProblem(self, problem):
        self.problem = problem

    @abstractmethod
    def run(self, state, action):
        pass

    @abstractmethod
    def inSample(self, p, p2) -> bool:
        pass

    @abstractmethod
    def getName(self):
        pass


class Ellipsoid(InnerMethod):
    # Initialize Ellipsoid
    def __init__(self, beta):
        InnerMethod.__init__(self)
        self.beta = beta

    # get name
    def getName(self):
        return "Ellipsoid(" + str(self.beta) + ")"

    # see if a transition kernel p is in sample
    def inSample(self, p) -> bool:
        for action in range(self.problem.A):
            for state in range(self.problem.S):
                out_of_bounds = mean(power(
                    subtract(p, self.problem.P[action][state]),
                    2)
                ) >= self.beta
                if out_of_bounds:
                    return False

        return True

    # calculate update scalar for inner method
    def run(self, state, action):
        model = Model('EllipsoidModel')
        pGurobi = model.addVars(self.problem.S, vtype=GRB.CONTINUOUS, name="p")
        p = transpose(array(pGurobi.items()))[1]
        objective = LinExpr()
        objective += dot(p, self.problem.V)
        model.setObjective(objective, GRB.MINIMIZE)
        # below is old. But this criteria rejected a lot of samples as probabilities can go to 0
        # also I don't think it fully matches the paper implementation
        # model.addConstr(divide(sum(
        #     multiply(
        #         subtract(p, self.problem.P[action][state]),
        #         subtract(p, self.problem.P[action][state]))
        # ), self.problem.P[action][state]
        # ) <= self.beta)

        model.addConstr(mean(
            multiply(
                subtract(p, self.problem.P[action][state]),
                subtract(p, self.problem.P[action][state]))
        ) <= self.beta)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal


class Interval(InnerMethod):
    # Initialize Interval
    def __init__(self, p_lower, p_upper):
        InnerMethod.__init__(self)
        self.p_upper = p_upper
        self.p_lower = p_lower

    # get name
    def getName(self):
        return "Interval"

    def attachProblem(self, problem):
        InnerMethod.attachProblem(self, problem)
        if self.p_lower.shape == (self.problem.S,):
            self.p_lower = repeat([repeat([self.p_lower], self.problem.S, axis=0)], self.problem.A, axis=0)
        if self.p_upper.shape == (self.problem.S,):
            self.p_upper = repeat([repeat([self.p_upper], self.problem.S, axis=0)], self.problem.A, axis=0)

        assert self.p_lower.shape == (self.problem.A, self.problem.S, self.problem.S),\
            "p_lower must be in the shape A*S*S or S*1."
        assert self.p_upper.shape == (self.problem.A, self.problem.S, self.problem.S),\
            "p_upper must be in the shape A*S*S or S*1."

        self.p_lower = maximum(self.p_lower, 0)
        self.p_upper = minimum(self.p_upper, 1)

    # see if a transition kernel p is in sample
    def inSample(self, p) -> bool:
        for a in range(self.problem.A):
            for s in range(self.problem.S):
                for s2 in range(self.problem.S):
                    if p[a][s][s2] < self.p_lower[a][s][s2] or p[a][s][s2] > self.p_upper[a][s][s2]:
                        return False
        return True

    # calculate update scalar for inner method
    def run(self, state, action):
        model = Model('IntervalModel')
        index = range(len(self.problem.V))
        # shouldn't mu have i different values? Old version had 1x1 value instead of len(V)x1
        mu = model.addVar(name="mu", vtype=GRB.CONTINUOUS,)

        lu = model.addVars(index, name="lu", vtype=GRB.CONTINUOUS)
        ll = model.addVars(index, name="ll", vtype=GRB.CONTINUOUS)
        for i in index:
            model.addConstr(mu - lu[i] + ll[i] == self.problem.V[i])
            model.addConstr(lu[i] >= 0)
            model.addConstr(ll[i] >= 0)

        objective = LinExpr()
        objective += mu #sp.eye(len(index))

        for i in index:
            objective += -(self.p_upper[action][state][i] * lu[i])
            objective += (self.p_lower[action][state][i] * ll[i])

        model.setObjective(objective, GRB.MAXIMIZE)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        # todo: debug -> AttributeError: b"Unable to retrieve attribute 'objVal'"
        # the equation seems to be infeasible
        return model.objVal


class Likelihood(InnerMethod):
    # Initialize Elipsoid
    def __init__(self, beta, delta):
        InnerMethod.__init__(self)
        self.beta = beta
        self.delta = delta
        self.bMax = None

    # get name
    def getName(self):
        return "Likelihood(" + str(self.beta) + ", " + str(self.delta) + ")"

    # attach problem
    def attachProblem(self, problem):
        InnerMethod.attachProblem(self, problem)
        self.bMax = zeros(self.problem.A)
        for a in range(self.problem.A):
            for i in range(self.problem.S):
                for j in range(self.problem.S):
                    self.bMax[a] -= self.problem.P[a][i][j] * math.log(self.problem.P[a][i][j] + sys.float_info.epsilon)

    #TODO beta computation is a bit more complicate (paper removes subscripts)
     #   if self.beta > max(self.bMax):
     #       print("Beta will be cut of to " + str(max(self.bMax)))
     #   self.beta = minimum(self.beta, max(self.bMax))

    # see if a transition kernel p is in sample
    # TODO: make sure this works
    def inSample(self, p) -> bool:
        for a in range(self.problem.A):
            for s in range(self.problem.S):
                # todo: debug, as it is comparing deep negative numbers with beta
                if sum(self.problem.P[a][s] * log(p[a][s] + sys.float_info.epsilon)) > self.beta:
                    return False
        return True

    # calculate update scalar for inner method
    def run(self, state, action):
        beta = self.beta #TODO calculate beta
        # todo combine beta with beta_max
        mu_upper = min(self.problem.V)
        e_factor = math.pow(math.e, beta) - sys.float_info.epsilon
        v_avg = dot(self.problem.V, self.problem.P[action][state])
        mu_lower = (min(self.problem.V) - e_factor * v_avg) / (1 - e_factor)  # TODO bug
        mu = (mu_upper + mu_lower) / 2
      #  if mu_upper < mu_lower and mu_lower - mu_upper > self.delta:
      #      print("BUG")

     #   print("{} - {}".format(mu_lower, mu_upper))
        while (mu_upper - mu_lower) > self.delta:  # TODO
            diff = mu_upper - mu_lower
            mu = (mu_upper + mu_lower) / 2
            if self.derivativeOfSigmaLikelyhoodModel(mu, state, action) < 0:
                mu_upper = mu
            else:
                mu_lower = mu
     #   print(mu)
        lmbda = self.lambdaLikelyhoodModel(mu, state, action)
        if  abs(lmbda - sys.float_info.epsilon) <= sys.float_info.epsilon:
            return mu
        return mu + (1 + beta) * lmbda - lmbda *  sum(
             multiply(
                self.problem.P[action][state],
                 log(sys.float_info.epsilon +  divide(
                    self.lambdaLikelyhoodModel(mu, state, action) * self.problem.P[action][state],
                     subtract(self.problem.V,  repeat(mu, self.problem.S))))))

    # privately used methods

    def derivativeOfSigmaLikelyhoodModel(self, mu, state, action):
        dsigma =  self.beta -  sum(
             multiply(
                self.problem.P[action][state],
                 log(
                    sys.float_info.epsilon +
                     divide(
                        self.lambdaLikelyhoodModel(mu, state, action)*self.problem.P[action][state],
                         subtract(self.problem.V,  repeat(mu, self.problem.S))+ sys.float_info.epsilon))))
        dsigma *= (- sum( divide(self.problem.P[action][state],  power( self.problem.V - mu *  ones(self.problem.S), 2))))
        dsigma /= math.pow( sum( divide(self.problem.P[action][state],  self.problem.V - mu *  ones(self.problem.S))), 2)
        return dsigma

    def lambdaLikelyhoodModel(self, mu, state, action):
        return 1 /  sum( divide(self.problem.P[action][state], self.problem.V - mu* ones(self.problem.S) + sys.float_info.epsilon))


class Wasserstein(InnerMethod):
    # Initialize Wasserstein
    def __init__(self, beta):
        InnerMethod.__init__(self)
        self.beta = beta

    # get name
    def getName(self):
        return "Wasserstein(" + str(self.beta) + ")"

    # see if a transition kernel p is in sample
    def inSample(self, p) -> bool:
        max_distance = 0;
        for a in range(self.problem.A):
            for s in range(self.problem.S):
                max_distance = max(max_distance, wasserstein_distance(self.problem.P[a][s], p[a][s]))
        return max_distance < self.beta

    # calculate update scalar for inner method
    def run(self, state, action):
        model = Model('SigmaEMD')
        pGurobi = model.addVars(self.problem.S, vtype=GRB.CONTINUOUS, name="p")
        p = transpose(array(pGurobi.items()))[1]
        emdGurobi = model.addVars(self.problem.S, vtype=GRB.CONTINUOUS, name="emd")
        emd = transpose(array(emdGurobi.items()))[1]
        emdAbsGurobi = model.addVars(self.problem.S, vtype=GRB.CONTINUOUS, name="emd_abs")
        emdAbs = transpose(array(emdAbsGurobi.items()))[1]
        objective = LinExpr()
        objective += dot(p, self.problem.V)
        model.setObjective(objective, GRB.MINIMIZE)
        for i in range(self.problem.S):
            if i == 0:
                model.addConstr(emd[i] == p[i] - self.problem.P[action][state][i])
            else:
                model.addConstr(emd[i] == p[i] - self.problem.P[action][state][i] + emd[i - 1])
            model.addConstr(emd[i] <= emdAbs[i])
            model.addConstr(-emd[i] <= emdAbs[i])
        model.addConstr((-sum(emdAbs)) <= self.beta)
        model.addConstr(sum(emdAbs) <= self.beta)

        # stay silent
        model.setParam('OutputFlag', 0)

        model.optimize()
        return model.objVal


if __name__ == "__main__":
    P, R = mdptoolbox.example.forest()
    m = Robust(Wasserstein(0.12))(P, R, 0.94)
    m.run()
    print(m.policy)
    print(m.V)
