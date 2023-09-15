from MOAA.Solutions import Solution
import numpy as np


def mutation(soln: Solution, pm: float, all_pixels: np.array, zero_prob: float):
    all_pixels = all_pixels.copy()
    pixels = soln.pixels.copy()
    rgbs = soln.values.copy()

    eps_it = max([int(len(soln.pixels) * pm), 1])
    eps = len(soln.pixels)

    # select pixels to keep
    A_ = np.random.choice(eps, size=(eps - eps_it,), replace=False)
    new_pixels = pixels[A_]
    new_rgbs = rgbs[A_]

    # select new pixels to replace
    u_m = np.delete(all_pixels, pixels)
    B = np.random.choice(u_m, size=(eps_it,), replace=False)

    ones_prob = (1 - zero_prob) / 2
    rgbs_ = np.random.choice([-1, 1, 0], size=(eps_it, 3), p=(ones_prob, ones_prob, zero_prob))
    pixels_ = all_pixels[B]

    new_pixels = np.concatenate([new_pixels, pixels_], axis=0)
    new_rgbs = np.concatenate([new_rgbs, rgbs_], axis=0)

    soln.pixels = new_pixels
    soln.values = new_rgbs


def crossover(soln1: Solution, soln2: Solution, pc: float):
    l = max([int(len(soln1.pixels) * pc), 1])
    k = len(soln1.pixels)
    # S1 crossover with S2
    # 1. Generate set of different pixels in S2
    delta = np.asarray([pi for pi in range(k) if soln2.pixels[pi] not in soln1.pixels])

    offspring1 = soln1.copy()
    if len(delta)>0:
        l = l if l <= len(delta) else len(delta)
        switched_pixels = np.random.choice(delta, size=(l,))
        offspring1.pixels[switched_pixels] = soln2.pixels[switched_pixels].copy()
        offspring1.values[switched_pixels] = soln2.values[switched_pixels].copy()

    # S2 crossover with S1
    # 1. Generate set of different pixels in S2
    delta = np.asarray([pi for pi in range(k) if soln1.pixels[pi] not in soln2.pixels])
    offspring2 = soln1.copy()
    if len(delta)>0:
        l = l if l <= len(delta) else len(delta)
        switched_pixels = np.random.choice(delta, size=(l,))
        offspring2.pixels[switched_pixels] = soln1.pixels[switched_pixels].copy()
        offspring2.values[switched_pixels] = soln1.values[switched_pixels].copy()

    return offspring1, offspring2


def generate_offspring(parents, pc, pm, all_pixels, zero_prob):
    children = []
    for pi in parents:
        offspring1, offspring2 = crossover(pi[0], pi[1], pc)
        mutation(offspring1, pm, all_pixels, zero_prob)
        mutation(offspring2, pm, all_pixels, zero_prob)

        assert len(np.unique(offspring1.pixels)) == len(offspring1.pixels)
        assert len(np.unique(offspring2.pixels)) == len(offspring2.pixels)
        children.extend([offspring1, offspring2])

    return children




