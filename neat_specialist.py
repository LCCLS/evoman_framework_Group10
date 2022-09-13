import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from evoman_framework_Group10.demo_controller import player_controller
from evoman_framework_Group10.neat_controller import NeatController

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


def run(config_path):
    # Load the confih file (neat_config)
    neat_config = os.path.dirname('evoman_framework_Group10.neat_config.txt')
    config_path = os.path.join(neat_config, 'neat_config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

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
