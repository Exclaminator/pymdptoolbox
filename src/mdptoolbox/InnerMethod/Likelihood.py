from mdptoolbox.InnerMethod.InnerMethod import InnerMethod
from numpy import *
import sys


class Likelihood(InnerMethod):
    # Initialize Elipsoid
    def __init__(self, beta, delta):
        InnerMethod.__init__(self)
        self.beta = beta
        self.delta = delta
        self.bMax = None

    # attach problem
    def attachProblem(self, problem):
        InnerMethod.attachProblem(self, problem)
        self.bMax = zeros(self.problem.A)
        for a in range(self.problem.A):
            for i in range(self.problem.S):
                for j in range(self.problem.S):
                    self.bMax[a] -= self.problem.P[a][i][j] * math.log(self.problem.P[a][i][j] + sys.float_info.epsilon)

        if self.beta > max(self.bMax):
            print("Beta will be cut of to " + str(max(self.bMax)))
        self.beta = minimum(self.beta, max(self.bMax))
        
    # see if a transition kernel p is in sample
    # TODO: make sure this works
    def inSample(self, p) -> bool:
        for a in range(self.problem.A):
            for s in range(self.problem.S):
                # todo: debug, as it is comparing deep negative numbers with beta
                if sum(self.problem.P[a][s]*log(p[a][s]+sys.float_info.epsilon)) > self.beta:
                    return False
        return True

    # calculate update scalar for inner method
    def run(self, state, action):
        mu_lower = max(self.problem.V)
        e_factor = math.pow(math.e, self.beta - self.bMax[action]) - sys.float_info.epsilon
        mu_upper = (max(self.problem.V) - e_factor*average(self.problem.V)) / (1 - e_factor)
        mu = (mu_upper + mu_lower)/2
        while (mu_upper - mu_lower) > self.delta*(1+2*mu_lower):
            mu = (mu_upper + mu_lower)/2
            if self.derivativeOfSigmaLikelyhoodModel(mu, state, action) < 0:
                mu_upper = mu
            else:
                mu_lower = mu
        lmbda = self.lambdaLikelyhoodModel(mu, state, action)
        if abs(lmbda - sys.float_info.epsilon) <= sys.float_info.epsilon:
            return mu
        return mu - (1 + self.beta)*lmbda + lmbda*sum(
            multiply(
                self.problem.P[action][state],
                log(sys.float_info.epsilon + divide(
                        self.lambdaLikelyhoodModel(mu, state, action)*self.problem.P[action][state],
                        subtract(repeat(mu, self.problem.S), self.problem.V)))))

    # privately used methods
    def derivativeOfSigmaLikelyhoodModel(self, mu, state, action):
        dsigma = - self.beta + sum(
            multiply(
                self.problem.P[action][state],
                log(
                    sys.float_info.epsilon +
                    divide(
                        self.lambdaLikelyhoodModel(mu, state, action)*self.problem.P[action][state],
                        subtract(repeat(mu, self.problem.S), self.problem.V)+ sys.float_info.epsilon))))
        dsigma *= sum(divide(self.problem.P[action][state], power(mu * ones(self.problem.S) - self.problem.V, 2)))
        dsigma /= math.pow(sum(divide(self.problem.P[action][state], mu * ones(self.problem.S) - self.problem.V)), 2)
        return dsigma

    def lambdaLikelyhoodModel(self, mu, state, action):
        return 1 / sum(divide(self.problem.P[action][state], mu*ones(self.problem.S) - self.problem.V + sys.float_info.epsilon))
