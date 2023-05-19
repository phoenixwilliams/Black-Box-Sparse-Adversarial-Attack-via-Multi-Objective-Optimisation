from MOAA.operators import *
from MOAA.Solutions import *
import numpy as np
import time

def p_selection(it, p_init, n_queries):
    it = int(it / n_queries * 10000)
    if 0 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 5
    elif 500 < it <= 1000:
        p = p_init / 6
    elif 1000 < it <= 2000:
        p = p_init / 8
    elif 2000 < it <= 4000:
        p = p_init / 10
    elif 4000 < it <= 6000:
        p = p_init / 12
    elif 6000 < it <= 8000:
        p = p_init / 15
    elif 8000 < it:
        p = p_init / 20
    else:
        p = p_init

    return p


class Population:
    def __init__(self, solutions: list, loss_function, include_dist):
        self.population = solutions
        self.fronts = None
        self.loss_function = loss_function
        self.include_dist = include_dist

    def evaluate(self):
        for pi in self.population:
            pi.evaluate(self.loss_function, self.include_dist)

    def find_adv_solns(self, max_dist):
        adv_solns = []
        for pi in self.population:
            if pi.is_adversarial and pi.fitnesses[1] <= max_dist:
                adv_solns.append(pi)

        return adv_solns


class Attack:
    def __init__(self, params):
        self.params = params
        self.fitness = []

        self.data = []

    # def update_data(self, front):

    def completion_procedure(self, population: Population, loss_function, fe, success):

        #print(success, fe)
        #print(1/0)

        adversarial_labels = []
        for soln in population.fronts[0]:
            adversarial_labels.append(loss_function.get_label(soln.generate_image()))

        d = {"front0_imgs": [soln.generate_image() for soln in population.fronts[0]],
             "queries": fe,
             "true_label": loss_function.true,
             "adversarial_labels": adversarial_labels,
             "front0_fitness": [soln.fitnesses for soln in population.fronts[0]],
             "fitness_process": self.fitness,
             "success": success
             }

        # print(d["true_label"], d["adversarial_labels"])
        np.save(self.params["save_directory"], d, allow_pickle=True)

    def attack(self, loss_function):
        start = time.time()
        # print(loss_function(self.params["x"]))
        # print(self.params["n_pixels"])
        # Minimizes
        h, w, c = self.params["x"].shape[0:]
        pm = self.params["pm"]
        n_pixels = h * w
        all_pixels = np.arange(n_pixels)
        ones_prob = (1 - self.params["zero_probability"]) / 2
        init_solutions = [Solution(np.random.choice(all_pixels,
                                                    size=(self.params["eps"]), replace=False),
                                   np.random.choice([-1, 1, 0], size=(self.params["eps"], 3),
                                                    p=(ones_prob, ones_prob, self.params["zero_probability"])),
                                   self.params["x"].copy(), self.params["p_size"]) for _ in
                          range(self.params["pop_size"])]

        population = Population(init_solutions, loss_function, self.params["include_dist"])
        population.evaluate()
        fe = len(population.population)
        for it in range(1, self.params["iterations"]):
            #pm = p_selection(it, self.params["pm"], self.params["iterations"])
            pm = self.params["pm"]
            population.fronts = fast_nondominated_sort(population.population)

            adv_solns = population.find_adv_solns(self.params["max_dist"])
            if len(adv_solns) > 0:
                self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
                self.completion_procedure(population, loss_function, fe, True)
                return

            self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)

            #print(fe, self.fitness[-1])

            for front in population.fronts:
                calculate_crowding_distance(front)
            parents = tournament_selection(population.population, self.params["tournament_size"])
            children = generate_offspring(parents,
                                          self.params["pc"],
                                          pm,
                                          all_pixels,
                                          self.params["zero_probability"])

            offsprings = Population(children, loss_function, self.params["include_dist"])
            fe += len(offsprings.population)
            offsprings.evaluate()
            population.population.extend(offsprings.population)
            population.fronts = fast_nondominated_sort(population.population)
            front_num = 0
            new_solutions = []
            while len(new_solutions) + len(population.fronts[front_num]) <= self.params["pop_size"]:
                calculate_crowding_distance(population.fronts[front_num])
                new_solutions.extend(population.fronts[front_num])
                front_num += 1

            calculate_crowding_distance(population.fronts[front_num])
            population.fronts[front_num].sort(key=attrgetter("crowding_distance"), reverse=True)
            new_solutions.extend(population.fronts[front_num][0:self.params["pop_size"] - len(new_solutions)])

            population = Population(new_solutions, loss_function, self.params["include_dist"])

        population.fronts = fast_nondominated_sort(population.population)
        self.fitness.append(min(population.population, key=attrgetter('loss')).fitnesses)
        self.completion_procedure(population, loss_function, fe, False)
        #print(time.time() - start)ff
        return
