import pickle
import pygad
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np 
def matlab_style(fig: go.Figure, size: int = 20) -> None:
    font_dict=dict(family='Arial',
            size = size,
            color='black')

    fig.update_layout(font=font_dict,
                plot_bgcolor='white',
                width=850,  # figure width
                height=700,  # figure height
                margin=dict(r=20,t=20,b=10) ) # remove white space
        
    fig.update_yaxes(showline=True,  # add line at x=0
                linecolor='black',  # line color
                linewidth=2.4, # line size
                ticks='outside',  # ticks outside axis
                tickfont=font_dict, # tick label font
                mirror='allticks',  # add ticks to top/right axes
                tickwidth=2.4,  # tick width
                tickcolor='black')
    fig.update_xaxes(showline=True,
                    showticklabels=True,
                    linecolor='black',
                    linewidth=2.4,
                    ticks='outside',
                    tickfont=font_dict,
                    mirror='allticks',
                    tickwidth=2.4,
                    tickcolor='black')

def plot_fittness(GA: pygad.GA, title: str = '', labels: List[str] = []):
    NumGenerations: int = GA.generations_completed #type: ignore
    if NumGenerations > 0:
        fig = go.Figure()
        x: List[int] = list(range(NumGenerations))
        data: List[NDArray[np.float64]] = GA.best_solutions_fitness #type: ignore
        Ndim = data[0].shape[0]
        customlabels = False 
        if len(labels) == Ndim: customlabels = True


        for i in range(Ndim):
            y = [abs(d[i]) for d in data]
            if customlabels:
                name = labels[i]
            else:
                name = f'Obj {i}'
            line = go.Scatter(x = x, y = y, mode = 'lines', name = name)
            fig.add_trace(line)

        fig.update_layout(title = title)
        matlab_style(fig)
        fig.show()
    else:
        print('0 Generations completed, Exiting...')
        return 

def plot_genes(GA: pygad.GA, GeneNames: List[str] = [], ytitles: List[str] = []):
    N: int = GA.num_genes # type: ignore
    data: NDArray = GA.best_solutions # type: ignore
    Ngen: int = GA.generations_completed

    if not GeneNames:
        GeneNames = [f'Gene {i}' for i in range(N)] 
    x = list(range(Ngen))

    fig = make_subplots(rows = N, cols = 1,
                        shared_xaxes = True,
                    x_title = 'Generation Number',
                    vertical_spacing = 0.04,
                    subplot_titles= GeneNames,
                    specs = [[{}],[{}],[{}],[{'b': 0.05}]]) 

    for i in range(N):
        y = data[:,i]
        trace = go.Scatter(x = x, y = y, marker = {'size': 3,}, showlegend = False, mode = 'markers')
        fig.add_trace(trace, row = i+1, col = 1)
        # fig.update_xaxes(title_text = GeneNames[i], row = i+1, col = 1)
        fig.update_yaxes(title_text = ytitles[i], row = i+1, col = 1)

    

    fig.update_layout(title_text = 'Gene Evolution')
    # matlab_style(fig)
    fig.show()


with open('GeneticAlgorithmResults/12.7.2024/Genetic_Algorithm_Run.pkl', 'rb') as f:
    Run: pygad.GA = pickle.load(f)


plot_fittness(Run, labels = ['Mass', 'Flutter Velocity'])
plot_genes(Run, GeneNames= [ 'Thickness', '$\\theta_1$', '$\\theta_2$', '$\\theta_3$'],
            ytitles = ['mm', 'deg', 'deg', 'deg'] )
# Run.plot_genes(plot_type = 'scatter', solutions = 'best', )