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
from specialist.neat_controller import NeatController
from specialist.neat_utils import *


def test_best_genome(experiment_name):
    env = Environment(  # if any changes are to be made in the environment, we have to report them in the paper
        experiment_name=experiment_name,
        enemies=[2],
        playermode="ai",
        player_controller=NeatController(),
        enemymode="static",
        level=2,  # this parameter should not be changed
        contacthurt='player',  # this parameter should not be changed
        speed="normal",
        multiplemode="no",  # sequentially changing enemies in the order given above
        randomini="yes",
    )

    env.state_to_log()

    with open(experiment_name + "/EXP_7/best_genome.txt", "rb") as f:
        genome = pickle.load(f)

    total_performance = list(env.play(pcont=genome))

    with open(f"{experiment_name}/EXP_1_performance" + "/best_genome_example.csv", 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(['fitness', 'player_health', 'enemy_health', 'time'])
        writer_object.writerow(total_performance)
        f_object.close()

        print(f"""
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Best Genome run completed. Fitness: {total_performance[0]}, Player Health: {total_performance[1]}, Enemy 
        Health: {total_performance[2]}, Time played: {total_performance[3]} %% 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """)


if __name__ == '__main__':

    headless = False
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    exp_name = 'finished_experiments/LIN_NEAT_ENEMY_2'
    test_best_genome(exp_name)
