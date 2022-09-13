import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from neat_controller import NeatController

# imports other libs
import time
import numpy as np
import os

import neat
import multiprocessing

headless = True  # this does not display visuals and makes the experiment faster
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'neat_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# this sets up the environment in an individual evolution mode
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=NeatController(),
                  enemymode="static",
                  level=1,
                  speed="fastest")

env.state_to_log()  # checks environment state

ini = time.time()  # sets time marker

run_mode = 'train'  # train or test


# _______________Implement NEAT - 1st attempt____________________________

# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        genome.fitness = simulation(env, genome)


def run(configuration_filepath):
    # Load the confih file (neat_config)

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
    winner = pop.run(eval_genomes, 30)

    # Show final stats
    print('\nBest genome:\n{!s}'.format(winner))


# ________________________________________________________________________


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')
    run(config_path)


    N_runs = 10

    experiment_name = "Neat_enemies_78"
    # create directory if it does not exist yet
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')
    #run(config_path)


    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=[7, 8],
        playermode="ai",
        player_controller=NeatController(),
        enemymode="static",
        level=2,
        speed="fastest",
        multiplemode="yes",
        randomini="yes",
    )

    run(config_path)
    # default environment fitness is assumed for experiment
