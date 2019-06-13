

class ProblemSet(object):
    """
    representation of a collection of problems.
    """

    def __init__(self, problem_base, ambiguity_model):
        self.problem_base = problem_base
        self.ambiguity_model = ambiguity_model

    def filter_problem_set(self, all_problems):
        # all_problems is a list instead of a ProblemSet object
        all_problems.filter(self.ambiquity_model.inSample)

