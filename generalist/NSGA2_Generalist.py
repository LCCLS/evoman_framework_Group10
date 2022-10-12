import csv
import pickle

import numpy as np
import os
from _csv import writer
import ast

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from generalist.nsga2_utils import write_results_to_files, write_genomes_to_files

import sys

sys.path.insert(0, 'evoman')

from environment import Environment
from generalist.demo_controller import Controller


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
                generation.append(sum(res) / len(res) * - 1)

            fitness_gens.append(np.mean(generation))
            fitness_max.append(np.max(generation))
            fitness_std.append(np.std(generation))


            ''' save the results of performance to files '''
            write_results_to_files(fitness_max, fitness_gens, fitness_std, problem_name, experiment_no)
            '''/ save the results of performance to files /'''


            out['F'] = np.array(res)



    problem = ProblemWrapper(n_var=int(n_vars), n_obj=int(1), xl=0.0, xu=100.0)
    algorithm = NSGA2(pop_size=5)
    stop_criterion = get_termination('n_gen', 1)

    EVO_results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=stop_criterion
    )

    ''' save the genomes to files '''
    genomes = EVO_results.pop.get("X")
    write_genomes_to_files(genomes, problem_name, experiment_no)
    '''/ save the genomes to files /'''



   # print('\nGenomes: \n', pop.get('X'))
   # print('\nResults: \n', EVO_results.F)




def evaluate_best_genomes(experiment_name, all_enemies):

    experiment_results_folder = f"../Experiments/generalist_experiments/{problem_name}/{experiment_no}"
    genomes_file_name = f"/Genomes.txt"

    genomes_file_path = experiment_results_folder + genomes_file_name


    # Pre processing of the file that we are reading
    with open(genomes_file_path, 'r') as f:
        genomes = f.read()
        genomes = genomes.replace('\n', '')
        genomes = genomes.replace('\t', '')
        genomes = genomes.replace('"', '')
        genomes = genomes.replace("'", '')
        genomes = genomes.replace("  ", ' ')
        genomes = genomes.split(',')

    test_results = []
    for genome in genomes:
        test_results.append(simulation(genome))


    print('\nBest genome Results: \n', test_results)





if __name__ == '__main__':

    #  PARAMETERS  #
    all_enemies = [2, 5]

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    problem_name = f"nsga2 - {all_enemies}"
    for i in range(4):

        experiment_no = f"EXP_{i+1}"
        experiment_name = f"NSGA2_GEN"

        if not os.path.exists(problem_name):
            os.makedirs(problem_name)

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

    evaluate_best_genomes(experiment_name, all_enemies)
