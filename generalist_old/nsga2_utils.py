import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import plotly.graph_objs as go
import plotly.express as px
import numpy as np





def generation_line_plot(directory, enemies, function=None):

    """
    plotting the fitness per generation for the best and the mean values per generation
    the values are averaged using the average_experiment_gen. the standard deviations for mean and max are the
    standard deviation across these 10 runs
    """

    file = f'{directory}/Enemies - {enemies} results.txt'
    #file = directory
  #  enemy = int(directory[-1])
    generations = np.array(pd.read_csv(file))
    generations = generations.T

    print('\n\n', generations)

    fig = go.Figure(data=go.Scatter(x=generations[0], y=generations[1], mode='markers'))
    fig.show()

    """
    fig = go.Figure([
        go.Scatter(
            name='Max. Fitness',
            x=generations['Fitness'],
            y=generations['Gain'],
            mode='markers',
            line=dict(color='rgb(255, 0, 0)'),
            legendrank=4
        )
    ])
    fig.update_layout(
        yaxis_title='Gain',
        xaxis_title='Fitness',
        title=f'Final Gain for Enemies {enemies}',
        hovermode="x",
        legend_title_text='Fitness Metrics',

    )
    """
    if not os.path.exists(f"EXPERIMENT RESULTS"):
        os.makedirs(f"EXPERIMENT RESULTS")

    #with open(file, 'w') as f_object:

    fig.write_image(f"{directory}/Enemies {enemies}_plot.png")
    #  fig.show()

