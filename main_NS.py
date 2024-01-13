import pygad
import random
import time

from maze_environment.Maze import Maze
from novelty_search.NoveltyArchive import NoveltyArchive
from utils.plot import plot_population

'''
This main file uses novelty search algorithm to resolving maze
'''
seed = str(time.time()).replace(".", "")[8:]
seed = int(seed)


def main():
    ### Creating maze_environment ###

    maze_seed = 883695529
    maze_env = Maze(height=6, width=6)
    maze, agent, solution_path = maze_env.create_maze(seed=maze_seed)
    cromosome_lenght = len(solution_path)
    expected_result = maze_env.convert_path_to_int_list(solution_path=solution_path)
    assert cromosome_lenght == len(expected_result), "Error during conversion, length must be the same"
    print(
        f"Number of steps to perform for exit is {cromosome_lenght} hence chromosome will be {cromosome_lenght} length")

    ### Creating Genetic Algorithm ###

    genetic_alg_seed = seed
    if genetic_alg_seed is not None:
        print(f"Set NS seed: {genetic_alg_seed}")
    random.seed(genetic_alg_seed)

    # Define chromosome and population
    num_generations = 50
    num_parents_mating = 40
    sol_per_pop = 1000
    num_genes = cromosome_lenght

    k = 15
    assert k <= sol_per_pop, f"You must specify a k value minor or equal than population size. k = {k}, pop size = {sol_per_pop}"
    novelty_archive = NoveltyArchive(genomes=[], k=k, max_archive_dim=50)
    temp_population_archive = NoveltyArchive(genomes=[], k=k, max_archive_dim=-1)

    # Define fitness function
    def fitness_func(solution, solution_idx) -> float:
        genome = list(solution)
        _, novelty_value = temp_population_archive.get_novelty_value(genomes=[genome])
        novelty_value = novelty_value[0]
        return novelty_value

    def on_generation_callback(ga_instance):
        # We call two times gen 0, at start and at the end so we delete the first population an we keep only the second one
        # In this case we are in the second 0-generation hence we drop the archive
        #if ga_instance.generations_completed == 0 and novelty_archive.get_size() > 0: novelty_archive.clean_archive()
        print(f"Elaborating generation {ga_instance.generations_completed}")
        temp_population_archive.clean_archive()
        converted_population = [list(genome) for genome in ga_instance.population]
        if novelty_archive.get_size() > 0: converted_population = converted_population + novelty_archive.get_genomes()
        temp_population_archive.add_genomes_to_archive(new_genomes=converted_population)
        most_novel_genomes, _ = temp_population_archive.get_most_novel_genome(number=3)
        novelty_archive.add_genomes_to_archive(new_genomes=most_novel_genomes)

    def on_start_callback(ga_instance):
        # This function is called just on time
        converted_population = [list(genome) for genome in ga_instance.population]
        temp_population_archive.add_genomes_to_archive(new_genomes=converted_population)

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
    crossover_probability = 0.90
    # Percentage of gene to mutate in a chromosome
    mutation_percent_genes = 30
    # Elitism Genotype --> we keep the best solution to the next generation
    keep_elitism = 5

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
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
                           gene_space=gene_space,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           random_seed=genetic_alg_seed,
                           on_generation=on_generation_callback,
                           on_start=on_start_callback)
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
    # plot_population(population=novelty_archive.get_genomes(convert_to_numpy=True), seed=0)
    print(str(novelty_archive))
    # This is only for animation
    # maze.tracePath({agent: found_path})
    # maze.run()


if __name__ == '__main__':
    main()
