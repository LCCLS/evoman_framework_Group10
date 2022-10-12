import pickle
from _csv import writer

import numpy as np
import os

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import sys

sys.path.insert(0, 'evoman')

from environment import Environment
from demo_code.demo_controller import Controller

from generalist_old.nsga2_utils import generation_line_plot


def simulation(x):
    """
    not really needed anymore.
    """
    f, p, e, t = env.play(pcont=x)
    return [f, p - e]




def run():
    """
    evolution using nsga2.

    Problem (class): defines the objective of the evolution
    -- look into objectives arg -- passing in play function or fitness?? adapt environment file??

    Evolution (class):
    -- look into expand arg -- what is the input to the objectives ?? list of respective enemy input or vector ??
    """

    class ProblemWrapper(Problem):

        def _evaluate(self, designs, out, *args, **kwargs):
            res = []
            for design in designs:
                res.append(simulation(design))

            out['F'] = np.array(res)

    problem = ProblemWrapper(n_var=int(n_vars), n_obj=int(2), xl=0.0, xu=100.0)
    algorithm = NSGA2(pop_size=5)
    stop_criterion = get_termination('n_gen', 2)



    EVO_results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=stop_criterion
    )


    print('\nResults F:\n', EVO_results.F)

    with open(f"{experiment_name}/Enemies - {all_enemies} results.txt", 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(['Fitness', 'Gain'])

        for i in EVO_results.F:
            writer_object.writerow(i)

        f_object.close()

    with open(f"{experiment_name}/Enemies - {all_enemies}.txt", 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(EVO_results.X)
        f_object.close()

    generation_line_plot(experiment_name, all_enemies)



if __name__ == '__main__':

    #  PARAMETERS  #
    all_enemies = [3, 5]

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = f"NSGA2_GEN"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = Environment(
        experiment_name=experiment_name,
        enemies=all_enemies,
        playermode="ai",
        player_controller=Controller(),
        enemymode="static",
        level=2,
        contacthurt='player',
        speed="fastest",
        multiplemode="yes",
        randomini="yes",
    )
    env.state_to_log()

    fitness_gens = []
    fitness_max = []
    fitness_std = []
    gen = 0

    n_hidden_neurons = 10
    n_vars = int((env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5)

    run()