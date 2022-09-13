import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import Player_Controller

# imports other libs
import time
import numpy as np
import os

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
                  player_controller=Player_Controller(n_hidden_neurons),
                  enemymode="static",
                  level=1,
                  speed="fastest")

env.state_to_log()  # checks environment state

ini = time.time()  # sets time marker

run_mode = 'train'  # train or test
