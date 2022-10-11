import numpy as np
import os
import statistics

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
    return np.multiply(f, -1)


def run():
    """
    ProblemWrapper(): problem class that inherits from pymoo.Problems with custom _evaluate function
    evaluates each genome based on individual gain as specified in the fitness function
    """

    class ProblemWrapper(Problem):

        def _evaluate(self, designs, out, *args, **kwargs):

            res = []
            generation = []

            for design in designs:
                mul_fit = simulation(design)
                res.append(mul_fit)
                generation.append(sum(mul_fit) / len(mul_fit) * - 1)

            fitness_gens.append(np.mean(generation))
            fitness_max.append(np.max(generation))
            fitness_std.append(np.std(generation))

            if len(os.listdir(f"../Experiments/generalist_experiments")) == 0:
                with open(f"../Experiments/generalist_experiments/results.txt", "a") as f:

                    f.write("best,mean,std, \n")
                    f.write(f"{str((fitness_max[-1]))}, {str((fitness_gens[-1]))}, {str((fitness_std[-1]))}, \n")
            else:

                with open(f"../Experiments/generalist_experiments/results.txt", "a") as f:
                    f.write(f"{str((fitness_max[-1]))}, {str((fitness_gens[-1]))}, {str((fitness_std[-1]))}, \n")

            out['F'] = np.array(res)

    problem = ProblemWrapper(n_var=int(n_vars), n_obj=int(2), xl=0.0, xu=100.0)
    algorithm = NSGA2(pop_size=30)
    stop_criterion = get_termination('n_gen', 50)
    EVO_results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=stop_criterion,
        verbose=True,
    )

    pop = EVO_results.pop
    print(pop.get('X'))
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

    enemy_env_dict = {}

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

    fitness_gens = []
    fitness_max = []
    fitness_std = []

    run()
