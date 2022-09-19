import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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

    #  NOT SURE IF THIS FULLY WORKS
    # total_avg = (exp1[['best', 'mean']] + exp2[['best', 'mean']] + exp3[['best', 'mean']]) / 3 # different approach
    # that also works in case for double-checking result
    total_mean_exp = pd.concat([exp1, exp2, exp3]).groupby(level=0).mean()
    total_std_exp = pd.concat([exp1, exp2, exp3]).groupby(level=0).std()

    total_mean_exp = total_mean_exp.rename(columns={"gen": "Generation", "best": "Avg_Maximum_Fitness",
                                                    "mean": "Avg_Mean_Fitness"})
    total_mean_exp = total_mean_exp.drop(['std'], axis=1)

    total_std_exp = total_std_exp.rename(columns={"best": "Avg_Maximum_Std", "mean": "Avg_Mean_Std"})
    total_std_exp = total_std_exp.drop(['gen', 'std'], axis=1)
    final_df = pd.concat([total_mean_exp, total_std_exp], axis=1)

    if not os.path.exists(f"{root_dir}/EXP_AVERAGE"):
        os.makedirs(f"{root_dir}/EXP_AVERAGE")

    final_df.to_csv(f"{root_dir}/EXP_AVERAGE/results.txt")


def generation_plotting(dir_filepath, enemy):
    """
    plotting the fitness per generation for the best and the mean values per generation

    --> using matplotlib so not so pretty
    NOT BEING USED AT THE MOMENT
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
    raise AssertionError('wrong plotting function. use: pretty_generation_plotting')

    #  plt.show()


def pretty_generation_plotting(dir_filepath, enemy):
    """
    plotting the fitness per generation for the best and the mean values per generation

    --> using plotly which is muc nicer and does not mess up the plots
    """
    file = dir_filepath + '/results.txt'
    generations = pd.read_csv(file)

    df_melt = generations.melt(id_vars='Generation', value_vars=['Avg_Maximum_Fitness', 'Avg_Mean_Fitness',
                                                                 "Avg_Maximum_Std", "Avg_Mean_Std"])
    fig = px.line(df_melt, x='Generation', y='value', color='variable')
    fig.update_layout(
        title={'text': f"Generational Fitness for Enemy {enemy}"},
        xaxis_title="Generation",
        yaxis_title="Fitness",
        legend_title="Metrics", )
    fig.write_image(dir_filepath + '/generation_fitness.png')
    # fig.show()


# enemies = [2, 7, 8]
# for i in enemies:
#    filepath = f"../NEAT_ENEMY_{i}/EXP_1"  # practice filepath
#    pretty_generation_plotting(filepath, i)

#dir_files = 'NEAT_ENEMY_2'
#average_experiment_gens(dir_files)
#pretty_generation_plotting(dir_files + "/EXP_AVERAGE", 2)
