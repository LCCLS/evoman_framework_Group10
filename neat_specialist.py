import pickle
import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from neat_controller import NeatController

# imports other libs
import time
import numpy as np
import os

import neat


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


def custom_fitness(env, x, gamma=0.9, alpha=0.1, mode='custom_fitness'):
    """
    this is a custom function with default keyword parameters, change these for experiment
    """
    f, p, e, t = env.play(pcont=x)
    fitness = gamma * (100 - e) + alpha * p - np.log(t)

    if mode == 'custom_fitness':
        return fitness
    elif mode == 'final_run':
        return fitness, p, e, t


def eval_genomes(genomes, config):
    """
    runs the evolution for n generations and saves the mean, std, and max fitness of each generation to the exp file
    """
    generation = []
    global gen
    for genome_id, genome in genomes:
        genome.fitness = 0
        genome.fitness = simulation(env, genome)
        generation.append(genome.fitness)

    fitness_gens.append(np.mean(generation))
    fitness_max.append(np.max(generation))
    fitness_std.append(np.std(generation))

    # saving experiment information

    file_aux = open(f"{experiment_name}/EXP_{i + 1}" + "/results.txt", "a")
    file_aux.write("\ngen best mean std")
    file_aux.write(
        f"\ngen-{gen}: "
        + str((fitness_max[-1]))
        + " "
        + str((fitness_gens[-1]))
        + " "
        + str((fitness_std[-1]))
    )
    file_aux.close()
    gen += 1


def run(configuration_filepath):
    # Load the config file (neat_config)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        configuration_filepath)

    pop = neat.Population(config)

    # Ass stdout reporter to show porgress
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5))  # don't know what is that

    # Run NEAT for 30 generations
    winner = pop.run(eval_genomes, 50)

    # Show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    # save the winning genome
    with open(f"{experiment_name}/EXP_{i + 1}" + "/best_genome.txt", 'wb') as f:
        pickle.dump(winner, f)


def run_best_genome(env, dir_path="NEAT_implementation_1/EXP_1"):

    with open(dir_path + "/best_genome.txt", "rb") as f:
        genome = pickle.load(f)

    genome_fitness, p, e, t = custom_fitness(env, genome, mode='final_run')
    performance_result = f"Best Genome fitness: {genome_fitness}\n" \
                         f"Best Genome Player Health:{p}\n" \
                         f"Best Genome Enemy Health:{e}\n" \
                         f"Best Genome time played:{t}\n"
    print(f"""
-------------------------------------------------------------------------------------------------------------------
    {performance_result}
-------------------------------------------------------------------------------------------------------------------

    """)

    file_performance = open(f"{experiment_name}/EXP_{i + 1}_performance" + "/best_genome_performance", "a")
    file_performance.write(f"Best genome performance:\n{performance_result}")
    file_performance.close()


if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # other parameters that need to be set:
    experiment_name = "Neat_implementation_1"
    N_runs = 1

    # create directory if it does not exist yet
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=[2],
        playermode="ai",
        player_controller=NeatController(),
        enemymode="static",
        level=2,
        speed="fastest",
        multiplemode="no",
        randomini="yes",
    )

    # default environment fitness is assumed for experiment
    env.state_to_log()  # checks environment state

    for i in range(N_runs):
        if not os.path.exists(f"{experiment_name}/EXP_{i + 1}"):
            os.makedirs(f"{experiment_name}/EXP_{i + 1}")

        n_hidden_neurons = 10
        fitness_gens = []
        fitness_max = []
        fitness_std = []
        gen = 0

        run('neat_config.txt')

    for i in range(N_runs):

        if not os.path.exists(f"{experiment_name}/EXP_{i + 1}_performance"):
            os.makedirs(f"{experiment_name}/EXP_{i + 1}_performance")

        run_best_genome(env, dir_path=f"{experiment_name}" + f"/EXP_{i + 1}")
    print('ALl games played. All files saved. ')
