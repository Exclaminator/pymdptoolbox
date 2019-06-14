from abc import abstractmethod

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
