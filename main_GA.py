import numpy as np
import pygad
import random
import time

from environment.Maze import Maze
# from utils.plot import plot_population

'''
This main file uses classic genetic algorithm to resolving maze
'''
seed = str(time.time()).replace(".", "")[8:]
seed = int(seed)


def main():

    ### Creating environment ###

    maze_seed = 883695529
    maze_env = Maze(height=8, width=8)
    maze, agent, solution_path = maze_env.create_maze(seed=maze_seed)
    cromosome_lenght = len(solution_path)
    expected_result = maze_env.convert_path_to_int_list(solution_path=solution_path)
    assert cromosome_lenght == len(expected_result), "Error during conversion, length must be the same"
    print(
        f"Number of steps to perform for exit is {cromosome_lenght} hence chromosome will be {cromosome_lenght} length")

    ### Creating Genetic Algorithm ###

    genetic_alg_seed = 899794116
    random.seed(genetic_alg_seed)

    # Define fitness function
    def fitness_func(solution, solution_idx) -> float:
        current = np.array(solution)
        expected = np.array(expected_result)
        differences = list(np.array(current) != np.array(expected))
        nb_differences = differences.count(True)
        fitness = 1 / (nb_differences + 1)
        return fitness

    fitness_function = fitness_func

    # Define chromosome and population
    num_generations = 200
    num_parents_mating = 20
    sol_per_pop = 70
    num_genes = cromosome_lenght

    # Defining existing range of each gene
    gene_type = int
    init_range_low = min(Maze.DICT_POSITION_ENC_FROM_INT.keys())
    init_range_high = max(Maze.DICT_POSITION_ENC_FROM_INT.keys())
    random_mutation_min_val = init_range_low
    random_mutation_max_val = init_range_high
    gene_space = list(Maze.DICT_POSITION_ENC_FROM_INT.keys())
    gene_space.sort()

    # Defining genetic operator
    # Selection function --> steady-state selection
    parent_selection_type = "sss"
    # Crossover function --> two points crossover
    crossover_type = "two_points"
    # Mutation function --> random select of gene from a set of permissible values
    mutation_type = "random"
    # Probability of selecting gene to apply mutation
    mutation_probability = 0.2
    # Probability of selecting a parent for crossover operation
    crossover_probability = 0.9
    # Percentage of gene to mutate in a chromosome
    mutation_percent_genes = 30
    # Elitism Genotype --> we keep the best solution to the next generation
    keep_elitism = 1

    # Early stopping if fitness is 1 (max possible value based on fitness function)
    stop_criteria = "reach_1"

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           gene_type=gene_type,
                           keep_elitism=keep_elitism,
                           mutation_probability=mutation_probability,
                           crossover_probability=crossover_probability,
                           stop_criteria=stop_criteria,
                           gene_space=gene_space,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           random_seed=genetic_alg_seed)

    ga_instance.run()
    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    best_solution = list(best_solution)
    print(f"Fitness of the best solution is {best_solution_fitness}")
    print(f"Best solution is {best_solution}, expected solution is {expected_result}")
    if best_solution == expected_result:
        print("MAZE RESOLVED WITH SUCCESS")
    else:
        print("FAILED RESOLVING MAZE")
    found_path = maze_env.convert_path_to_tuple_list(best_solution)
    ga_instance.plot_fitness()
    # plot_population(population=ga_instance.population, winner_idx=best_solution_idx, seed=0)
    # This is only for animation
    maze.tracePath({agent: found_path})
    maze.run()


if __name__ == '__main__':
    main()
