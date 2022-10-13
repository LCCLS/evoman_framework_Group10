import numpy as np
import os

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination

from NSGA2_setup import MyOutput, ProblemWrapper
from NSGA2_utils import write_genomes_to_files, get_genomes_log, write_results_to_files, evaluate, nsga2_multiple, \
    save_genome, load_genome, save_best_genome_result

import sys

sys.path.insert(0, 'evoman')

from demo_controller import Controller
from environment import Environment


class NSGA2_Optimization:

    def __init__(self,
                 enemies,
                 generations,
                 n_var,
                 run_num,
                 pop_size,
                 experiment_name,
                 headless,
                 Experiment
                 ):

        self.arena = None
        self.enemies = enemies
        self.generations = int(generations)
        self.n_vars = n_var
        self.xl = -1.
        self.xu = 1.
        self.run_num = run_num
        self.n_obj = len(enemies)
        self.pop_size = pop_size
        self.experiment_name = experiment_name
        self.best_genomes = None
        self.tournament_enemies = range(1, 8)
        self.headless = headless
        self.Experiment = Experiment

    def evolve(self):
        """
        PROBLEM DEPENDENT IMPLEMENTATION

        ProblemWrapper(): problem class that inherits from pymoo.Problems with custom _evaluate function
        evaluates each genome based on individual gain as specified in the fitness function
        """
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            Environment.multiple = nsga2_multiple

            self.arena = Environment(

                experiment_name=self.experiment_name,
                enemies=self.enemies,
                playermode="ai",
                multiplemode="yes",
                player_controller=Controller(),
                enemymode="static",
                level=2,
                speed='fastest',
                logs="off",
                savelogs="no",
                sound="off",
                randomini="yes",
            )

        problem = ProblemWrapper(env=self.arena, n_var=self.n_vars, n_obj=self.n_obj, xl=self.xl, xu=self.xu)
        algorithm = NSGA2(pop_size=self.pop_size)
        stop_criterion = get_termination('n_gen', self.generations)

        algorithm.setup(problem, termination=stop_criterion, verbose=True, output=MyOutput())

        while algorithm.has_next():
            pop = algorithm.ask()
            algorithm.evaluator.eval(problem, pop)
            algorithm.tell(infills=pop)

            # ----- save the results to files ----- #
            f_results = get_genomes_log(algorithm.pop.get("F").tolist())
            write_results_to_files(f_results, self.experiment_name, self.enemies)
            # ------------------------------------- #

        # ----- save the genomes to files ----- #
        self.best_genomes = algorithm.pop.get("X").tolist()
        write_genomes_to_files(self.best_genomes, self.experiment_name)
        # ------------------------------------- #

        self.best_genome_tournament()

    def best_genome_tournament(self):
        """
        tournament of all pareto-optimal solutions to determine the best performing one.
        METRIC: individual gain (mean)

        FOR COMPETITION: create new environment in which each best genome is tested against ALL enemies!
        """

        winner_genome = None
        winner_performance = -np.inf

        for player_genome in self.best_genomes:

            f, p, e, t, ig = evaluate(player_genome, self.arena)

            if ig[0] >= winner_performance:
                winner_genome = player_genome
                winner_performance = ig[0]

        if winner_genome is not None:
            # write_genomes_to_files(np.array(winner_genome), self.experiment_name, tournament=True)
            save_genome(self.experiment_name, winner_genome)

    def run_best_genome_five_times(self):
        """
        this method should only be called in an experiment setting !!!
        -- runs the best genome of each run 5 times and saves the results in form of individual gain
        """

        for TESTING in range(5):
            print(f"{self.experiment_name}, happening.")
            print(f"TESTING run {TESTING} happening.")

            fighter = load_genome(self.experiment_name)
            f, p, e, t, ig = evaluate(fighter, self.arena)
            ig_list = [element for ig_list in ig[1] for element in ig_list]
            #ig_performance = f"{ig_list[0]}, {ig_list[1]}, {ig_list[2]}"
            save_best_genome_result(self.experiment_name, TESTING, total_performance=ig_list,
                                    enemies=self.enemies)


if __name__ == '__main__':

    #  PARAMETERS  #
    ENEMIES = [2, 5, 8]
    GENERATIONS = 40
    POP_SIZE = 70
    N_VAR = 265
    EXP_RUNS = 1
    TEST_RUNS = 5
    N_HIDDEN_NEURONS = 10
    EXPERIMENT_NAME = f"NSGA2_GEN"
    EXPERIMENT = True

    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    for RUN in range(EXP_RUNS):
        optimizer = NSGA2_Optimization(
            enemies=ENEMIES,
            generations=int(GENERATIONS),
            n_var=N_VAR,
            run_num=EXP_RUNS,
            pop_size=POP_SIZE,
            experiment_name=EXPERIMENT_NAME + f"/RUN_{RUN}",
            headless=True,
            Experiment=EXPERIMENT
        )

        optimizer.evolve()
        optimizer.run_best_genome_five_times()
