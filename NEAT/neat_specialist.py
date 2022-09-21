import numpy as np
import os
import neat
import pickle
import sys
import time
import pandas as pd

sys.path.insert(0, 'evoman')

from csv import writer
from environment import Environment
from neat_controller import NeatController
from neat_utils import *


def simulation(env, x):
    """
    not really needed anymore.
    """
    f, p, e, t = env.play(pcont=x)
    return f


def custom_fitness(env, x, gamma=0.9, alpha=0.1, mode='custom_fitness'):
    """
    this is a custom function with default keyword parameters, change these for experiment

    BE CAREFUL WITH THE ffunction parameter, this is passed into the environment class and was recently instantiated
    --> may run into troubles but if done correctly, the logs should print the correct fitness values
    """

    if mode == 'default':
        f, p, e, t = env.play(pcont=x, ffunction='linear')
        return f

    elif mode == 'exp_fitness':

        f, p, e, t = env.play(pcont=x, ffunction='exponential')

        fit = gamma * (100 - e) + alpha * p
        exp_f = 8.7483 * (1.0247 ** fit)

        max_f = 8.7483 * (1.0247 ** 100)
        min_f = 8.7483 * (1.0247 ** 0)
        norm_fit = (exp_f - min_f) / (max_f - min_f) * 100

        final_fit = norm_fit - np.log(t)
        return final_fit

    elif mode == 'final_run':
        f, p, e, t = env.play(pcont=x, ffunction='linear')

        return f, p, e, t

    else:
        raise KeyError("This mode of custom function does not exist.")


def eval_genomes(genomes, config):
    """
    runs the evolution for n generations and saves the mean, std, and max fitness of each generation to the EXP_ file.
    - the config file needs to be passed into the function as a parameter.
    """
    fitness_dict = {'best': 0, 'mean': 0, }
    generation = []
    global gen

    for genome_id, genome in genomes:
        genome.fitness = 0

        # DEFAULT EXPERIMENT LINEAR FITNESS  --> use mode:'default'
        # EXPONENTIAL EXPERIMENT FITNESS  --> use mode:'exp_fitness'
        genome.fitness = custom_fitness(env, genome, gamma=0.9, alpha=0.1, mode='exp_fitness')  # change the mode
        # parameter here
        generation.append(genome.fitness)

    fitness_gens.append(np.mean(generation))
    fitness_max.append(np.max(generation))
    fitness_std.append(np.std(generation))
    fitness_dict['best'] = np.max(generation)
    fitness_dict['mean'] = np.mean(generation)

    #  saving experiment information
    with open(f"{experiment_name}/EXP_{i}" + "/results.txt", "a") as f:

        if gen == 0:
            f.write("gen,best,mean,std")

        f.write(f"\n{gen},"
                + str((fitness_max[-1]))
                + ","
                + str((fitness_gens[-1]))
                + ","
                + str((fitness_std[-1]))
                )

        gen += 1


def run():
    """
    loads the neat cofiguration file and its parameters for inititation of a neat model. calls the eval_genome()
    function to evaluate the population with n number of generations
    """

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5))

    winner = pop.run(eval_genomes, 30)

    print(f"\nBest genome:\n{winner}")

    with open(f"{experiment_name}/EXP_{i}" + "/best_genome.txt", 'wb') as f:
        pickle.dump(winner, f)


def run_best_genome(env, dir_path="NEAT_implementation_1/EXP_1"):
    """
    loads the best genome for each exxperiment and performs 10 games on it and averages the results
    returns: saves the averaged resutls in the performance file related to the experiment
    """
    with open(dir_path + "/best_genome.txt", "rb") as f:
        genome = pickle.load(f)

    for trial in range(5):

        total_performance = list(custom_fitness(env, genome, mode='final_run'))

        with open(f"{dir_path}_performance" + "/best_genome_performance.csv", 'a') as f_object:
            writer_object = writer(f_object)

            if trial == 0:
                writer_object.writerow(['fitness', 'player_health', 'enemy_health', 'time'])

            writer_object.writerow(total_performance)
            f_object.close()

        print(f"""
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Best Genome run {trial}completed. Saved files to EXP_performance. %% 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """)


if __name__ == '__main__':

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    all_enemies = [2]  # experiments for enemies: 2, 7, 8 # otherwise we could try 2, 5, 8
    for enemy in all_enemies:

        #  PARAMETERS  #
        experiment_name = f"NEAT/NEAT_ENEMY_{enemy}"

        #  Change RUNS for the number of repetitions
        RUNS = 1
        N_runs = RUNS + 1  # don't change this parameter
        N_trials = 5

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

        for i in range(1, N_runs):
            if not os.path.exists(f"{experiment_name}/EXP_{i}"):
                os.makedirs(f"{experiment_name}/EXP_{i}")

            n_hidden_neurons = 10
            fitness_gens = []
            fitness_max = []
            fitness_std = []
            gen = 0

            run()

        for i in range(1, N_runs):

            if not os.path.exists(f"{experiment_name}/EXP_{i}_performance"):
                os.makedirs(f"{experiment_name}/EXP_{i}_performance")

            #  5 TRIAL RUNS FOR THE BEST GENOME OF EACH OF THE 10 RUNS
            run_best_genome(env, dir_path=f"{experiment_name}/EXP_{i}")

        average_experiment_gens(f"{experiment_name}")
        pretty_generation_plotting(f"{experiment_name}/EXP_AVERAGE", enemy)

        print(f"""
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% ENEMY {enemy}: ALL TRIALS PLAYED. ALL FILES SAVED. %% 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """)
