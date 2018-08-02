import math
import numpy as np
import random
import matplotlib.pyplot as plt

# Adapted to use numpy array as entry from https://github.com/chncyhn/simulated-annealing-tsp

class SimAnneal(object):
    def __init__(self, Mw, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.N = Mw.shape[0]
        self.T = math.sqrt(self.N) if T == -1 else T
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 0.00000001 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.dist_matrix = Mw
        self.nodes = [i for i in range(self.N)]

        self.cur_solution = self.initial_solution()
        self.best_solution = list(self.cur_solution)

        self.cur_fitness = self.fitness(self.cur_solution)
        self.initial_fitness = self.cur_fitness
        self.best_fitness = self.cur_fitness

        self.fitness_list = [self.cur_fitness]

    def initial_solution(self):
        """
        Greedy algorithm to get an initial solution (closest-neighbour)
        """
        cur_node = random.choice(self.nodes)
        solution = [cur_node]
        free_list = list(self.nodes)
        free_list.remove(cur_node)
        
        while free_list:
            closest_dist = np.amin(self.dist_matrix[cur_node,free_list])
            #np.where returns a list with indexes containing the closest_dist (which may be duplicated), intersect that with indexes still available (free_list) and get the first index available containing the closest_dist
            cur_node = np.intersect1d(np.where(self.dist_matrix[cur_node] == closest_dist)[0], np.array(free_list))[0]
            free_list.remove(cur_node)
            solution.append(cur_node)

        return solution


    def fitness(self, sol):
        """ Objective value of a solution """
        return round(sum([self.dist_matrix[sol[i - 1],sol[i]] for i in range(1, self.N)]) +
                     self.dist_matrix[sol[self.N - 1],sol[0]], 6)

    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current
        Depends on the current temperature and difference between candidate and current
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        """
        Accept with probability 1 if candidate is better than current
        Accept with probabilty p_accept(..) if candidate is worse
        """
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness = candidate_fitness
            self.cur_solution = candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_solution = candidate

        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness = candidate_fitness
                self.cur_solution = candidate

    def anneal(self):
        """
        Execute simulated annealing algorithm
        """
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i:(i + l)] = reversed(candidate[i:(i + l)])
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1
            self.fitness_list.append(self.cur_fitness)
