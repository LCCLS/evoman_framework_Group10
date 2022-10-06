import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import plotly.graph_objs as go


def group_by_mean(root_dir, mode=None):
    exp1 = pd.read_csv(f"{root_dir}/EXP_1/results.txt")
    exp2 = pd.read_csv(f"{root_dir}/EXP_2/results.txt")
    exp3 = pd.read_csv(f"{root_dir}/EXP_3/results.txt")
    exp4 = pd.read_csv(f"{root_dir}/EXP_4/results.txt")
    exp5 = pd.read_csv(f"{root_dir}/EXP_5/results.txt")
    exp6 = pd.read_csv(f"{root_dir}/EXP_6/results.txt")
    exp7 = pd.read_csv(f"{root_dir}/EXP_7/results.txt")
    exp8 = pd.read_csv(f"{root_dir}/EXP_8/results.txt")
    exp9 = pd.read_csv(f"{root_dir}/EXP_9/results.txt")
    exp10 = pd.read_csv(f"{root_dir}/EXP_10/results.txt")

    if mode == 'mean':
        return pd.concat([exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10]).groupby(level=0).mean()

    elif mode == 'std':
        return pd.concat([exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10]).groupby(level=0).std()


def average_experiment_gens(root_dir):
    """
    reads the individual dataframes from each experimetnal repetition and saves a new file which contains their average
    """
    #  CHECK IN THE NEAT_SPECIALIST.PY FILE FOR N_RUNS AND UPDATE THE NUMBER OF FILES TO READ

    total_mean_exp = group_by_mean(root_dir, mode='mean')
    total_std_exp = group_by_mean(root_dir, mode='std')

    total_mean_exp = total_mean_exp.rename(columns={"gen": "Generation", "best": "Maximum_Fitness",
                                                    "mean": "Mean_Fitness"})

    total_mean_exp = total_mean_exp.drop(['std'], axis=1)

    total_mean_exp["Maximum Standard Deviation"] = total_std_exp['best']
    total_mean_exp["Mean Standard Deviation"] = total_std_exp['mean']

    if not os.path.exists(f"{root_dir}/EXP_AVERAGE"):
        os.makedirs(f"{root_dir}/EXP_AVERAGE")

    total_mean_exp.to_csv(f"{root_dir}/EXP_AVERAGE/results.txt")


def generation_line_plot(directory, function=None):
    """
    plotting the fitness per generation for the best and the mean values per generation
    the values are averaged using the average_experiment_gen. the standard deviations for mean and max are the
    standard deviation across these 10 runs
    """
    file = directory + '/EXP_AVERAGE/results.txt'
    enemy = int(directory[-1])
    generations = pd.read_csv(file)

    fig = go.Figure([
        go.Scatter(
            name='Max. Fitness',
            x=generations['Generation'],
            y=generations['Maximum_Fitness'],
            mode='lines',
            line=dict(color='rgb(255, 0, 0)'),
            legendrank=4
        ),
        go.Scatter(
            name='Max. Standard Deviation',
            x=generations['Generation'],
            y=generations['Maximum_Fitness'] + generations["Maximum Standard Deviation"],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False,
        ),
        go.Scatter(
            name='Max. Standard Deviation',
            x=generations['Generation'],
            y=generations['Maximum_Fitness'] - generations["Maximum Standard Deviation"],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(255, 0, 0, 0.2)',
            fill='tonexty',
            showlegend=True,
            legendrank=2
        ),
        go.Scatter(
            name='Mean Fitness',
            x=generations['Generation'],
            y=generations['Mean_Fitness'],
            mode='lines',
            line=dict(color='rgb(0, 0, 255)'),
            legendrank=3
        ),
        go.Scatter(
            name='Mean Standard Deviation',
            x=generations['Generation'],
            y=generations['Mean_Fitness'] + generations["Mean Standard Deviation"],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Mean Standard Deviation',
            x=generations['Generation'],
            y=generations['Mean_Fitness'] - generations["Mean Standard Deviation"],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(0, 0, 255, 0.2)',
            fill='tonexty',
            showlegend=True,
            legendrank=1
        )
    ])
    fig.update_layout(
        yaxis_title='Fitness',
        xaxis_title='Generation',
        title=f'Generational Fitness for Enemy {enemy}, using {function} fitness function',
        hovermode="x",
        legend_title_text='Fitness Metrics',

    )

    if not os.path.exists(f"EXPERIMENT RESULTS"):
        os.makedirs(f"EXPERIMENT RESULTS")

    fig.write_image(f"EXPERIMENT RESULTS/GENERATION_FITNESS_E{enemy}_{function}.png")
    #  fig.show()


def best_solution_boxplots(exp_directory1, default_directory2, enemy=None):
    """
    making three boxplots for each enemy, each comparing the individual gain for the two algorithms
    for the best solution of each tested 5 times

    ###
    updated function works for all three enemies if the root directory is passed in.
    ###

    """
    gains = pd.DataFrame()

    for run in range(1, 10):  # the range value has to be updated in accordance to N_runs in neat_spec.

        enemy_run_filepath = f"/EXP_{run}_performance/best_genome_performance.csv"

        genomes1 = pd.read_csv(f"{exp_directory1}/{enemy_run_filepath}", usecols=['Individual Gain'])
        genomes1['Fitness Function'] = ['Exponential'] * len(genomes1['Individual Gain'].values)

        genomes2 = pd.read_csv(f"{default_directory2}/{enemy_run_filepath}", usecols=['Individual Gain'])
        genomes2['Fitness Function'] = ['Default'] * len(genomes2['Individual Gain'].values)

        exp_gains = pd.concat([genomes1, genomes2], axis=0)

        gains = pd.concat([gains, exp_gains])

    fig = px.box(gains, x="Fitness Function", y=gains["Individual Gain"],
                 title=f"Individual Gain for Enemy {enemy} per Fitness Function ")

    if not os.path.exists(f"EXPERIMENT RESULTS"):
        os.makedirs(f"EXPERIMENT RESULTS")

    fig.write_image(f"EXPERIMENT RESULTS/INDIVIDUAL_GAIN_E{enemy}.png")
    # fig.show()

# enemies = [2, 5, 8]

# for enemy in enemies:

#    exp_file = f'finished_experiments/EXP_NEAT_ENEMY_{enemy}'
#    lin_file = f'finished_experiments/LIN_NEAT_ENEMY_{enemy}'

#    average_experiment_gens(exp_file)
#    average_experiment_gens(lin_file)

#    generation_line_plot(exp_file, function='Exponential')
#    generation_line_plot(lin_file, function='Default')

#    best_solution_boxplots(exp_file, lin_file, enemy=enemy)
