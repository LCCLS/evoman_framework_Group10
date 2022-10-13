import numpy as np
import os
from _csv import writer


def evaluate(individual, env):
    """
    Method to actually run a play simulation.
    :param individual: one individual from the population
    """

    fitness, player_life, enemy_life, time, ind_gains = env.play(pcont=individual)
    return fitness, player_life, enemy_life, time, ind_gains


def nsga2_multiple(self, pcont, econt):
    """
    optimizing runtime --> remove individual gains from calculations
    """
    vfitness, vplayerlife, venemylife, vtime = [], [], [], []

    for e in self.enemies:
        fitness, playerlife, enemylife, time = self.run_single(e, pcont, econt)

        vfitness.append(fitness)
        vplayerlife.append(playerlife)
        venemylife.append(enemylife)
        vtime.append(time)

    vgains = list(zip(nsga2_cons_multi(np.array(vplayerlife))[1] - nsga2_cons_multi(np.array(venemylife))[1]))
    vgains = nsga2_cons_multi(np.array(vgains))
    vfitness = nsga2_cons_multi(np.array(vfitness))
    vplayerlife = sum(vplayerlife)
    venemylife = sum(venemylife)
    vtime = self.cons_multi(np.array(vtime))

    return vfitness, vplayerlife, venemylife, vtime, vgains


def nsga2_cons_multi(values):
    """
    takes in a list of values and returns: mean of list , all_values in original list format, and std of list
    -- original function subtracted std from mean which i changed
    """
    values_mean = values.mean()  # - values.std()
    std = values.std() * -1
    return (values_mean, values, std)


def write_genomes_to_files(genomes, problem_name, tournament=False):
    """ Store the genomes into respective file
            e.g. "Experiments/generalist_experiments/NSGA2_GEN/RUN_0/Genomes.txt
    ."""

    if tournament:
        experiment_results_folder = f"../Experiments/generalist_experiments/{problem_name}"
        genomes_file_name = "/BEST_GENOME.txt"

    else:
        experiment_results_folder = f"../Experiments/generalist_experiments/{problem_name}"
        genomes_file_name = "/Genomes.txt"

    if not os.path.exists(experiment_results_folder):
        os.makedirs(experiment_results_folder)

    with open(experiment_results_folder + genomes_file_name, 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(genomes)
        f_object.close()


def write_results_to_files(f_results, problem_name, enemies):
    """
    Writes the results (performance & genomes) to files
        e.g. "Experiments/generalist_experiments/NSGA2_GEN/RUN_0/results.txt
    CURRENTLY ONLY FOR TWO ENEMIES
    """

    experiment_results_folder = f"../Experiments/generalist_experiments/{problem_name}"
    results_file_name = "/results.txt"

    # If experiment directory doesn't exist create it
    if not os.path.exists(experiment_results_folder):
        os.makedirs(experiment_results_folder)

    # Write the respective files ( results & genomes )
    if len(os.listdir(experiment_results_folder)) == 0:

        with open(experiment_results_folder + results_file_name, 'w') as f:
            f.write(f"total_max, total_mean, total_std, max_E{enemies[0]}, mean_E{enemies[0]}, std_E{enemies[0]},"
                    f"max_E{enemies[1]}, mean_E{enemies[1]}, std_E{enemies[1]}, \n")
            f.write(f"{str((f_results[0]))}, {str((f_results[1]))}, {str((f_results[2]))}, "
                    f"{str((f_results[3]))}, {str((f_results[4]))}, {str((f_results[5]))}, "
                    f"{str((f_results[6]))}, {str((f_results[7]))}, {str((f_results[8]))}, \n")
            f.close()
    else:
        with open(experiment_results_folder + results_file_name, 'a') as f:
            f.write(f"{str((f_results[0]))}, {str((f_results[1]))}, {str((f_results[2]))}, "
                    f"{str((f_results[3]))}, {str((f_results[4]))}, {str((f_results[5]))}, "
                    f"{str((f_results[6]))}, {str((f_results[7]))}, {str((f_results[8]))}, \n")
            f.close()


def get_genomes_log(total_F):
    """
    get all the result information for each generation
    """
    x_max_total = "{:.8f}".format(float(min(list(map(min, total_F)))))
    x_mean_total = "{:.8f}".format(np.mean(sum(total_F, [])))
    x_std_total = "{:.8f}".format(np.std(sum(total_F, [])))

    x_max_E2 = "{:.8f}".format(np.min([item[0] for item in total_F]))
    x_mean_E2 = "{:.8f}".format(np.mean([item[0] for item in total_F]))
    x_std_E2 = "{:.8f}".format(np.std([item[0] for item in total_F]))

    x_max_E5 = "{:.8f}".format(np.min([item[1] for item in total_F]))
    x_mean_E5 = "{:.8f}".format(np.mean([item[1] for item in total_F]))
    x_std_E5 = "{:.8f}".format(np.std([item[1] for item in total_F]))

    return [x_max_total, x_mean_total, x_std_total, x_max_E2, x_mean_E2, x_std_E2, x_max_E5, x_mean_E5, x_std_E5]
