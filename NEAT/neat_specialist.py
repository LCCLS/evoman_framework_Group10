import numpy as np
import os
import neat
import pickle
import statistics
import sys
import time

import pandas as pd

sys.path.insert(0, 'evoman')
from environment import Environment
from neat_controller import NeatController
from neat_plotting import *


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
    else:
        raise KeyError("This mode of custom function does not exist.")


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
    if gen == 0:
        file_aux.write("gen,best,mean,std")
        file_aux.write(
            f"\n{gen},"
            + str((fitness_max[-1]))
            + ","
            + str((fitness_gens[-1]))
            + ","
            + str((fitness_std[-1]))
        )
    else:
        file_aux.write(
            f"\n{gen},"
            + str((fitness_max[-1]))
            + ","
            + str((fitness_gens[-1]))
            + ","
            + str((fitness_std[-1]))
        )

    file_aux.close()
    gen += 1


def run(configuration_filepath, ):
    # Load the config file (neat_config)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        configuration_filepath)

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5))  # this loads checkpoints for every 5 generations
    # runnning for n generations
    winner = pop.run(eval_genomes, 50)

    # Show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    # save the winning genome
    with open(f"{experiment_name}/EXP_{i + 1}" + "/best_genome.txt", 'wb') as f:
        pickle.dump(winner, f)


def run_best_genome(env, dir_path="NEAT_implementation_1/EXP_1"):
    """
    loads the best genome for each exxperiment and performs 10 games on it and averages the results
    returns: saves the averaged resutls in the performance file related to the experiment
    """
    with open(dir_path + "/best_genome.txt", "rb") as f:
        genome = pickle.load(f)

    total_performance = {'f': 0, 'p': 0, 'e': 0, 't': 0}

    for i in range(10):
        genome_fitness, p, e, t = custom_fitness(env, genome, mode='final_run')

        total_performance['f'] += genome_fitness
        total_performance['p'] += p
        total_performance['e'] += e
        total_performance['t'] += t

    total_performance['f'] = total_performance['f'] / 10
    total_performance['p'] = total_performance['p'] / 10
    total_performance['e'] = total_performance['e'] / 10
    total_performance['t'] = total_performance['t'] / 10

    performance_result = [[total_performance['f'], total_performance['p'], total_performance['e'], total_performance['t']]]
    #performance_df_2 = pd.DataFrame(total_performance)
    performance_df = pd.DataFrame(performance_result, columns=['fitness', 'player_health', 'enemy_health', 'time'])
    print(f"""
-------------------------------------------------------------------------------------------------------------------
    f"Best Genome fitness: {total_performance['f']}\n" \
                             f"Best Genome Player Health:{total_performance['p']}\n" \
                             f"Best Genome Enemy Health:{total_performance['e']}\n" \
                             f"Best Genome time played:{total_performance['t']}\n"
-------------------------------------------------------------------------------------------------------------------

    """)
    performance_df.to_csv(f"{dir_path}_performance" + "/best_genome_performance.csv")


if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # here we do the experiment for three individual enemies --> 3 specialist agents
    all_enemies = [2, 7, 8]
    for enemy in all_enemies:

        ### parameters for the experiment###
        experiment_name = f"NEAT/NEAT_ENEMY_{enemy}"
        N_runs = 1

        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        env = Environment(  # if any changes are to be made in the environment, we have to report them in the paper
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=NeatController(),
            enemymode="static",
            level=2,  # this parameter should not be changed
            contacthurt='player',  # this parameter should not be changed
            speed="fastest",
            multiplemode="no",  # sequentially changing enemies in the order given above
            randomini="yes",
        )

        env.state_to_log()

        for i in range(N_runs):
            if not os.path.exists(f"{experiment_name}/EXP_{i + 1}"):
                os.makedirs(f"{experiment_name}/EXP_{i + 1}")

            n_hidden_neurons = 10
            fitness_gens = []
            fitness_max = []
            fitness_std = []
            gen = 0

            run(config_path)
            generation_plotting(f"{experiment_name}/EXP_{i + 1}", enemy)  # creates plot for the generational fitness

        for i in range(N_runs):

            if not os.path.exists(f"{experiment_name}/EXP_{i + 1}_performance"):
                os.makedirs(f"{experiment_name}/EXP_{i + 1}_performance")

            run_best_genome(env, dir_path=f"{experiment_name}" + f"/EXP_{i + 1}")

        print('ALl games played. All files saved. ')
