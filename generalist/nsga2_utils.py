import os
from _csv import writer




def write_results_to_files(fitness_max, fitness_gens, fitness_std, problem_name, experiment_no):
    ''' Writes the results (performance & genomes) to files
    .'''

    experiment_results_folder = f"../Experiments/generalist_experiments/{problem_name}/{experiment_no}"
    results_file_name = f"/Results.txt"


    ''' If experiment directory doesn't exist create it '''
    if not os.path.exists(experiment_results_folder):
        os.makedirs(experiment_results_folder)



    ''' Write the respective files ( results & genomes ) '''
    if len(os.listdir(experiment_results_folder)) == 0:

        with open(experiment_results_folder + results_file_name, 'w') as f:
            f.write("best,mean,std, \n")
            f.write(f"{str((fitness_max[-1]))}, {str((fitness_gens[-1]))}, {str((fitness_std[-1]))}, \n")
            f.close()

    else:
        with open(experiment_results_folder + results_file_name, 'a')as f:
            f.write(f"{str((fitness_max[-1]))}, {str((fitness_gens[-1]))}, {str((fitness_std[-1]))}, \n")
            f.close()


def write_genomes_to_files(genomes, problem_name, experiment_no):
    """ Store the genomes into respective file
        eg. "Experiments/generalist_experiments/NSGA2 - [2, 5]/EXP_1/Genomes.txt
    ."""

    experiment_results_folder = f"../Experiments/generalist_experiments/{problem_name}/{experiment_no}"
    genomes_file_name = f"/Genomes.txt"



    with open(experiment_results_folder + genomes_file_name, 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(genomes)
        f_object.close()

