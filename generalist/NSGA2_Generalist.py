import numpy as np
import os

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import sys

sys.path.insert(0, 'evoman')

from environment import Environment
from demo_controller import Controller


def simulation(x):
    """
    not really needed anymore.
    """
    f, p, e, t = env.play(pcont=x)
    return p - e


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

    problem = ProblemWrapper(n_var=int(n_vars), n_obj=int(2), xl=0., xu=100.)
    algorithm = NSGA2(pop_size=100)
    stop_criterion = get_termination('n_gen', 3)

    EVO_results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=stop_criterion
    )

    print(EVO_results.F)


if __name__ == '__main__':

    #  PARAMETERS  #
    all_enemies = [2, 5]

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

    n_hidden_neurons = 10
    n_vars = int((env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5)

    run()
