from algorithm_x import AlgorithmX
import numpy as np

'''
Finds the combination of proposed associations which make sense and has the highest
likelihood given a list of statistical functions as objects of class statFunc.
'''

class optimizer():
    class Likelihood0(Exception): pass
    class CondtionsNotFound(Exception): pass

    def __init__(self, stat_funcs):
        self.stat_funcs = stat_funcs

    def optimize(self, associations):
        self._prep(associations)
        mem = [0, 0, 0]
        for X in self.iter:
            likelihood = self.calculate_likelihood_given_some_hypothesis(X)
            if likelihood > mem[2]: mem = [X, self.all_likelihoods[(X, self.pos)], likelihood]
        if mem[2] == 0 : optimizer.Likelihood0(len(X))
        return mem

    def calculate_likelihood_given_some_hypothesis(self, X):
        return np.prod(self.all_likelihoods[(X, self.pos)])

    def _prep(self, associations):
        self.associations, self.Ys, n = associations
        self.pos = np.arange(len(self.associations))
        self._calculate_likelihoods()
        search_space = optimization_reducer(n, self.associations)
        self.iter = search_space.generator()

    def _calculate_likelihoods(self):
        positive_likelihoods = np.zeros(self.pos.size, dtype = np.float64)
        for i, measurments in enumerate(self.Ys):
            positive_likelihoods[i] = self.main_likelihood_func(*measurments)
        positive_likelihoods = positive_likelihoods[:, np.newaxis].T
        negative_likelihoods = 1 - positive_likelihoods
        self.all_likelihoods = np.concatenate((negative_likelihoods, positive_likelihoods))

    '''
    Finds the apropriate statistical function for every association passed to it
    and applies it.
    If the likelihood is not in (0; 1], error is raised
    If no function is found for an association, Error is raised
    '''
    def main_likelihood_func(self, measurments1, measurments2):
        for stat_func in self.stat_funcs:
            if stat_func.check_conditions(measurments1, measurments2):
                likelihood = stat_func(measurments1, measurments2)
                if not 1 >= likelihood > 0: raise optimizer.Likelihood0(str(stat_func) + '=> %.4f'%likelihood)
                return likelihood
        optimizer.CondtionsNotFound(str(measurments1) + ';' + str(measurments2))

class optimization_reducer():
    def __init__(self, num, associations):
        self.num = len(associations)
        self.solver = AlgorithmX(num)
        for asc in associations: self.solver.appendRow(asc)

    def generator(self):
        for solution in self.solver.solve():
            X = np.zeros(self.num, dtype = np.uint8)
            X[solution] = 1
            yield X