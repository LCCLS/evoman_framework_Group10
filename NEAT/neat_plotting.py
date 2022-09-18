import pandas as pd
import matplotlib.pyplot as plt


def generation_plotting(dir_filepath):
    """
    plotting the fitness per generation for the best and the mean values per generation
    """
    file = dir_filepath + '/results.txt'
    generations = pd.read_csv(file)

    ax = plt.gca()
    generations.plot(kind='line', x='gen', y='mean', ax=ax)
    generations.plot(kind='line', x='gen', y='best', color='red', ax=ax)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.savefig(dir_filepath + '/generation_fitness.png')
    #  plt.show()


#  filepath = "NEAT_ENEMY_2/EXP_1/results.txt"  # practice filepath
#  generation_plotting(filepath)
