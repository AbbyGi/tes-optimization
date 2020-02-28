import numpy as np
from random import random, uniform
import matplotlib.pyplot as plt


def ackley(x, a=20, b=0.2, c=2*np.pi):
    # n dimensional
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum(x**2)
    s2 = sum(np.cos(c * x))
    return -a*np.exp(-b*np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)


def bukin(x):
    # 2 dimensional
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)


def griewank(x):
    # n dimensional
    dim = len(x)
    j = np.arange(1, dim + 1)
    sq_list = [i ** 2 for i in x]
    s = sum(sq_list)
    p = np.prod(np.cos(x / np.sqrt(j)))
    return s / 4000 - p + 1


def rosenbrock(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return sum(100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2)


def zakharov(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    dim = len(x)
    j = np.arange(1, dim + 1)
    s1 = sum(0.5 * j * x)
    return sum(x ** 2) + s1 ** 2 + s1 ** 4


def levy(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    w = 1 + (x - 1) / 4
    return (np.sin(np.pi * w[0])) ** 2 \
        + sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)) \
        + (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1]) ** 2))


def rastrigin(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    dim = len(x)
    return 10 * dim + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def schwefel(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    dim = len(x)
    return 418.9829 * dim - sum(x * np.sin(np.sqrt(np.abs(x))))


def sphere(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    return sum(x ** 2)


def sum_diff_powers(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    dim = len(x)
    j = np.arange(1, dim + 1)
    return sum(np.abs(x) ** (j + 1))


def sum_of_squares(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    dim = len(x)
    j = np.arange(1, dim + 1)
    return sum(j * x ** 2)


def rotated_hyper_ellipsoid(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    dim = len(x)
    total = 0
    for i in range(1, dim + 1):
        for j in range(1, i + 1):
            total += (x[j - 1] ** 2)
    return total


def ellipse(x):
    # n dimensional
    x = np.asarray_chkfinite(x)
    return np.mean((1 - x) ** 2) + 100 * np.mean(np.diff(x) ** 2)


def schafferN6(x):
    # 2 dimensional
    x = np.asarray_chkfinite(x)
    num = np.sin(np.sqrt(x[0] ** 2 + x[1] ** 2)) ** 2 - 0.5
    denom = (1 + 0.001*(x[0] ** 2 + x[1] ** 2)) ** 2
    return 0.5 + num / denom


def simple_parabola(x):
    x = np.asarray(x)
    return -3 * x ** 2 + 3


def beamline_test_function(x):
    x = np.asarray(x)
    return np.sin(4 * x) - np.cos(8 * x) + 2


def ensure_bounds(vec, bounds):
    # Makes sure each individual stays within bounds and adjusts them if they aren't
    vec_new = []
    # cycle through each variable in vector
    for i in range(len(vec)):
        # variable exceeds the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])
        # variable exceeds the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])
        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])
    return vec_new


def omea(positions, func):
    evaluations = []
    min_positions = []
    min_evals = []
    # get first position eval
    evaluations.append(func(positions[0]))

    for i in range(1, len(positions)):
        hold_eval = []
        in_between = np.linspace(positions[i - 1], positions[i], 50)
        between = in_between[1:-1]
        for t in range(len(between)):
            hold_eval.append(func(between[t]))
        # get eval of next position
        evaluations.append(func(positions[i]))
        # find index of min
        ii = np.argmin(hold_eval)
        min_positions.append(between[ii])
        min_evals.append(hold_eval[ii])
    for i in range(len(min_positions)):
        if min_evals[i] < evaluations[i + 1]:
            evaluations[i + 1] = min_evals[i]
            for k in range(len(min_positions[i])):
                positions[i + 1][k] = min_positions[i][k]
    return positions, evaluations


def rand_1(pop, popsize, t_indx, mut, bounds):
    # v = x_r1 + F * (x_r2 - x_r3)
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b, c = np.random.choice(idxs, 3, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_3 = pop[c]

    x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    v_donor = [x_1_i + mut * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def best_1(pop, popsize, t_indx, mut, bounds, ind_sol):
    # v = x_best + F * (x_r1 - x_r2)
    x_best = pop[ind_sol.index(np.min(ind_sol))]
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b = np.random.choice(idxs, 2, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]

    x_diff = [x_1_i - x_2_i for x_1_i, x_2_i in zip(x_1, x_2)]
    v_donor = [x_b + mut * x_diff_i for x_b, x_diff_i in zip(x_best, x_diff)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def current_to_best_1(pop, popsize, t_indx, mut, bounds, ind_sol):
    # v = x_curr + F * (x_best - x_curr) + F * (x_r1 - r_r2)
    x_best = pop[ind_sol.index(np.min(ind_sol))]
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b = np.random.choice(idxs, 2, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_curr = pop[t_indx]

    x_diff1 = [x_b - x_c for x_b, x_c in zip(x_best, x_curr)]
    x_diff2 = [x_1_i - x_2_i for x_1_i, x_2_i in zip(x_1, x_2)]
    v_donor = [x_c + mut * x_diff_1 + mut * x_diff_2 for x_c, x_diff_1, x_diff_2
               in zip(x_curr, x_diff1, x_diff2)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def best_2(pop, popsize, t_indx, mut, bounds, ind_sol):
    # v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - r_r4)
    x_best = pop[ind_sol.index(np.min(ind_sol))]
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b, c, d = np.random.choice(idxs, 4, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_3 = pop[c]
    x_4 = pop[d]

    x_diff1 = [x_1_i - x_2_i for x_1_i, x_2_i in zip(x_1, x_2)]
    x_diff2 = [x_3_i - x_4_i for x_3_i, x_4_i in zip(x_3, x_4)]
    v_donor = [x_b + mut * x_diff_1 + mut * x_diff_2 for x_b, x_diff_1, x_diff_2
               in zip(x_best, x_diff1, x_diff2)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def rand_2(pop, popsize, t_indx, mut, bounds):
    # v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - r_r5)
    idxs = [idx for idx in range(popsize) if idx != t_indx]
    a, b, c, d, e = np.random.choice(idxs, 5, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_3 = pop[c]
    x_4 = pop[d]
    x_5 = pop[e]

    x_diff1 = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    x_diff2 = [x_4_i - x_5_i for x_4_i, x_5_i in zip(x_4, x_5)]
    v_donor = [x_1_i + mut * x_diff_1 + mut * x_diff_2 for x_1_i, x_diff_1, x_diff_2
               in zip(x_1, x_diff1, x_diff2)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def mutate(population, strategy, mut, bounds, ind_sol):
    mutated_indv = []
    for i in range(len(population)):
        if strategy == 'rand/1':
            v_donor = rand_1(population, len(population), i, mut, bounds)
        elif strategy == 'best/1':
            v_donor = best_1(population, len(population), i, mut, bounds, ind_sol)
        elif strategy == 'current-to-best/1':
            v_donor = current_to_best_1(population, len(population), i, mut, bounds, ind_sol)
        elif strategy == 'best/2':
            v_donor = best_2(population, len(population), i, mut, bounds, ind_sol)
        elif strategy == 'rand/2':
            v_donor = rand_2(population, len(population), i, mut, bounds)
        mutated_indv.append(v_donor)
    return mutated_indv


def crossover(population, mutated_indv, crosspb):
    crossover_indv = []
    for i in range(len(population)):
        v_trial = []
        x_t = population[i]
        for j in range(len(x_t)):
            crossover_val = random()
            if crossover_val <= crosspb:
                v_trial.append(mutated_indv[i][j])
            else:
                v_trial.append(x_t[j])
        crossover_indv.append(v_trial)
    return crossover_indv


def select(population, crossover_indv, ind_sol, func):
    positions = [elm for elm in crossover_indv]
    positions.insert(0, population[0])
    positions, evals = omea(positions, func)
    positions = positions[1:]
    evals = evals[1:]
    for i in range(len(evals)):
        if evals[i] < ind_sol[i]:
            population[i] = positions[i]
            ind_sol[i] = evals[i]
    population.reverse()
    ind_sol.reverse()
    return population, ind_sol


def diff_ev(bounds, func, threshold, popsize=10, crosspb=0.8, mut=0.05, mut_type='rand/1'):
    # Initial population
    population = []
    best_fitness = [10]
    for i in range(popsize):
        indv = []
        for j in range(len(bounds)):
            indv.append(uniform(bounds[j][0], bounds[j][1]))
        population.append(indv)
    init_pop = population[:]

    # Evaluate fitness/OMEA
    init_pop.sort()
    pop, ind_sol = omea(init_pop, func)
    # reverse for efficiency with motors
    pop.reverse()
    ind_sol.reverse()

    # Termination conditions
    v = 0  # generation number
    consec_best_ctr = 0  # counting successive generations with no change to best value
    old_best_fit_val = 0
    while not (consec_best_ctr >= 5 and old_best_fit_val <= threshold):
        print('\nGENERATION ' + str(v + 1))
        best_gen_sol = []  # holding best scores of each generation
        mutated_trial_pop = mutate(pop, mut_type, mut, bounds, ind_sol)
        cross_trial_pop = crossover(pop, mutated_trial_pop, crosspb)
        pop, ind_sol = select(pop, cross_trial_pop, ind_sol, func)

        # score keeping
        gen_best = np.min(ind_sol)  # fitness of best individual
        best_indv = pop[ind_sol.index(gen_best)]  # best individual positions
        best_gen_sol.append(best_indv)
        best_fitness.append(gen_best)

        print('      > BEST FITNESS:', gen_best)
        print('         > BEST POSITIONS:', best_indv)

        v += 1
        if np.round(gen_best, 6) == np.round(old_best_fit_val, 6):
            consec_best_ctr += 1
            print('Counter:', consec_best_ctr)
        else:
            consec_best_ctr = 0
        old_best_fit_val = gen_best

        if consec_best_ctr >= 5 and old_best_fit_val <= threshold:
            print('Finished')
            break
        else:
            change_index = ind_sol.index(np.max(ind_sol))
            changed_indv = pop[change_index]
            for k in range(len(changed_indv)):
                changed_indv[k] = uniform(bounds[k][0], bounds[k][1])
            # OMEA would be here too
            # not sure how to do this with functions
            ind_sol[change_index] = func(changed_indv)

    x_best = best_gen_sol[-1]
    print('\nThe best individual is', x_best, 'with a fitness of', gen_best)
    print('It took', v, 'generations')

    # plot best fitness
    plot_index = np.arange(len(best_fitness))
    plt.figure()
    plt.plot(plot_index, best_fitness)


diff_ev(bounds=[(-10, 10)] * 20, func=sphere, threshold=0.05, popsize=10, crosspb=0.8, mut=0.2)
