import numpy as np
from copy import deepcopy
from operator import attrgetter


class Solution:
    def __init__(self, pixels, values, x, p_size):
        self.pixels = pixels  # list of Integers
        self.values = values  # list of Binary tuples, i.e. [0, 1, 1]
        self.x = x  # (w x w x 3)
        self.fitnesses = []
        self.is_adversarial = None
        self.w = x.shape[0]
        self.delta = len(self.pixels)
        self.domination_count = None
        self.dominated_solutions = None
        self.rank = None
        self.crowding_distance = None

        self.loss = None
        self.p_size = p_size

    def copy(self):
        return deepcopy(self)

    def euc_distance(self, img):
        return np.sum((img - self.x.copy()) ** 2)

    def generate_image(self):
        x_adv = self.x.copy()
        for i in range(self.delta):
            x_adv[self.pixels[i] // self.w, self.pixels[i] % self.w] += (self.values[i] * self.p_size)

        return np.clip(x_adv, 0, 1)

    def evaluate(self, loss_function, include_dist):
        img_adv = self.generate_image()
        fs = loss_function(img_adv)
        self.is_adversarial = fs[0]  # Assume first element is boolean always
        self.fitnesses = fs[1:]
        if include_dist:
            dist = self.euc_distance(img_adv)
            self.fitnesses.append(dist)
        else:
            self.fitnesses.append(0)

        self.fitnesses = np.array(self.fitnesses)
        self.loss = fs[1]

    def dominates(self, soln):
        if self.is_adversarial is True and soln.is_adversarial is False:
            return True

        if self.is_adversarial is False and soln.is_adversarial is True:
            return False

        if self.is_adversarial is True and soln.is_adversarial is True:
            return True if self.fitnesses[1] < soln.fitnesses[1] else False

        if self.is_adversarial is False and soln.is_adversarial is False:
            return True if self.fitnesses[0] < soln.fitnesses[0] else False


def fast_nondominated_sort(population):
    fronts = [[]]
    for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other_individual in population:
            if individual.dominates(other_individual):
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual):
                individual.domination_count += 1
        if individual.domination_count == 0:
            individual.rank = 0
            fronts[0].append(individual)
    i = 0
    while len(fronts[i]) > 0:
        temp = []
        for individual in fronts[i]:
            for other_individual in individual.dominated_solutions:
                other_individual.domination_count -= 1
                if other_individual.domination_count == 0:
                    other_individual.rank = i + 1
                    temp.append(other_individual)
        i = i + 1
        fronts.append(temp)

    return fronts


def calculate_crowding_distance(front):
    if len(front) > 0:
        solutions_num = len(front)
        for individual in front:
            individual.crowding_distance = 0

        for m in range(len(front[0].fitnesses)):
            front.sort(key=lambda individual: individual.fitnesses[m])
            front[0].crowding_distance = 10 ** 9
            front[solutions_num - 1].crowding_distance = 10 ** 9
            m_values = [individual.fitnesses[m] for individual in front]
            scale = max(m_values) - min(m_values)
            if scale == 0: scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (front[i + 1].fitnesses[m] - front[i - 1].fitnesses[m]) / scale


def crowding_operator(individual, other_individual):
    if (individual.rank < other_individual.rank) or ((individual.rank == other_individual.rank) and (
            individual.crowding_distance > other_individual.crowding_distance)):
        return 1
    else:
        return -1


def __tournament(population, tournament_size):
    participants = np.random.choice(population, size=(tournament_size,), replace=False)
    best = None
    for participant in participants:
        if best is None or (
                crowding_operator(participant, best) == 1):  # and self.__choose_with_prob(self.tournament_prob)):
            best = participant

    return best


def tournament_selection(population, tournament_size):
    parents = []
    while len(parents) < len(population) // 2:
        parent1 = __tournament(population, tournament_size)
        #parent2 = parent1
        #while parent1 == parent2:
        while np.all([pi in parent2.pixels for pi in parent1.pixels]):
            parent2 = __tournament(population, tournament_size)

        parents.append([parent1, parent2])
    return parents
