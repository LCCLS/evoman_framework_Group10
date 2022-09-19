import pandas as pd
import matplotlib.pyplot as plt
import os


def average_experiment_gens(root_dir):
    """
    reads the individual dataframes from each experimetnal repetition and saves a new file which contains their average
    """
    #  CHECK IN THE NEAT_SPECIALIST.PY FILE FOR N_RUNS AND UPDATE THE NUMBER OF FILES TO READ

    exp1 = pd.read_csv(f"{root_dir}/EXP_1/results.txt", index_col=False)
    exp2 = pd.read_csv(f"{root_dir}/EXP_2/results.txt", index_col=False)
    exp3 = pd.read_csv(f"{root_dir}/EXP_3/results.txt", index_col=False)
    # exp4 = pd.read_csv(f"{root_dir}/EXP_4/results.txt")
    # exp5 = pd.read_csv(f"{root_dir}/EXP_5/results.txt")
    # exp6 = pd.read_csv(f"{root_dir}/EXP_6/results.txt")
    # exp7 = pd.read_csv(f"{root_dir}/EXP_7/results.txt")
    # exp8 = pd.read_csv(f"{root_dir}/EXP_8/results.txt")
    # exp9 = pd.read_csv(f"{root_dir}/EXP_9/results.txt")
    # exp10 = pd.read_csv(f"{root_dir}/EXP_10/results.txt")

    #  FINAL EXPERIMENT CONCATENATION OF DFS
    # total_exp = pd.concat([exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10]).groupby(level=0).mean()

    total_exp = pd.concat([exp1, exp2, exp3]).groupby(level=0).mean()

    if not os.path.exists(f"{root_dir}/EXP_MEAN"):
        os.makedirs(f"{root_dir}/EXP_MEAN")

    total_exp.to_csv(f"{root_dir}/EXP_MEAN/results.txt")


def generation_plotting(dir_filepath, enemy):
    """
    plotting the fitness per generation for the best and the mean values per generation
    """
    file = dir_filepath + '/results.txt'
    generations = pd.read_csv(file)

    ax = plt.gca()
    generations.plot(kind='line', x='gen', y='mean', ax=ax)
    generations.plot(kind='line', x='gen', y='best', ax=ax)

    plt.title(f"Generational Fitness for Enemy {enemy}")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    plt.savefig(dir_filepath + '/generation_fitness.png')
    #  plt.show()


#  filepath = "NEAT_ENEMY_2/EXP_1/results.txt"  # practice filepath
#  generation_plotting(filepath)