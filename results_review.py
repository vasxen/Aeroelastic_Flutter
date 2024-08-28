import pickle
import pygad
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
import typing
from typing import TypeVar, Generic, cast
from itertools import cycle
from optimization import ObjectiveFunction, PlySymmetry, FlutterSummary
Times_New_Roman = {'fontname': 'Times New Roman'}

DType = TypeVar("DType")
class Array(np.ndarray, Generic[DType]):
    def __getitem__(self, key) -> DType:
        return super().__getitem__(key) # type: ignore

def plot_fittness(GA: pygad.GA, title: str = '', labels: List[str] = []):
    NumGenerations: int = GA.generations_completed + 1 #type: ignore
    if NumGenerations > 0:
        fig, ax = plt.subplots(layout = 'constrained')
        fig.suptitle('Fitness Evolution', fontname = 'Times New Roman', fontsize = 14)
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
            ax.plot(x, y, label = name, marker = '.', markersize = 2)
        ax.grid(which = 'both')
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Num. Generation', Times_New_Roman, fontsize = 12 )
        ax.set_ylabel('Fittness', Times_New_Roman, fontsize = 12 )
    else:
        print('0 Generations completed, Exiting...')
        return 

def plot_genes(GA: pygad.GA, GeneNames: List[str] = [], ytitles: List[str] = []):
    N: int = GA.num_genes # type: ignore
    data: NDArray = GA.best_solutions # type: ignore
    Ngen: int = GA.generations_completed + 1

    if not GeneNames:
        GeneNames = [f'Gene {i}' for i in range(N)] 
    x = list(range(Ngen))
    fig, axess = plt.subplots(N, 1, figsize = (8,6) ,sharex = True, layout = 'constrained')
    axess = cast(Array[Axes], axess)
    fig.suptitle('Gene Evolution', fontname = 'Times New Roman', fontsize = 14)
    for i in range(N):
        y = data[:,i]
        ax = axess[i]
        ax.scatter(x, y, s = 5, label = GeneNames[i])
        ax.set_ylabel(ytitles[i])
        # ax.legend(loc = 'upper right')
        ax.grid(which= 'both')
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-1, 3), useMathText=True)
        if i != 0:
            ax.set_ylim((-90,90))
            ax.set_yticks(np.arange(-90,91,45))
            # ax.legend(loc = 'best')

    ax.set_xlabel('Num. Generation', Times_New_Roman, fontsize = 12)

def PlotFlutter(flt_file: str, lastmode: int, firstmode: int = 0, blacknwhite: bool = False):
    FltSummary = FlutterSummary(flt_file)
    fig, axes = plt.subplots(2,1, layout = 'constrained')
    axes = cast(Array[Axes], axes)
   
    axes[0].axhline(y=0, color='black')
    axes[0].set_ylabel('Damping', Times_New_Roman)
    axes[0].grid(which = 'both')
    axes[0].set_title('V-g Plot', Times_New_Roman)
    # axes[0].set_xlabel('Velocity', Times_New_Roman)
    range(0,1)
    axes[1].grid(which = 'both')
    axes[1].set_ylabel('Frequency [Hz]', Times_New_Roman)
    axes[1].set_xlabel('Velocity [m/s]', Times_New_Roman)
    axes[1].set_title('V-f Plot', Times_New_Roman)

    linestyles = cycle(['-', '--', '-.', ':'])
    markers = cycle(['.', '^', 's', 'p', 'd'])

    for i in range (firstmode, lastmode):
        linestyle = next(linestyles)
        markerstyle = next(markers)

        line_d = FltSummary.Subcases[0].Points[i].Plot(axes[0], 2, 3, f'Mode {i+1}')
        if blacknwhite:
            line_d.set_color('black')
            line_d.set_linestyle(linestyle)
            line_d.set_marker(markerstyle)

        line_f = FltSummary.Subcases[0].Points[i].Plot(axes[1], 2, 4, f'Mode {i+1}')
        if blacknwhite:
            line_f.set_color('black')
            line_f.set_linestyle(linestyle)
            line_f.set_marker(markerstyle)

    axes[0].legend()
    axes[1].legend()
    return fig

def main():
        
    with open('GeneticAlgorithmResults/12.7.2024/Genetic_Algorithm_Run.pkl', 'rb') as f:
        Run: pygad.GA = pickle.load(f)

    plot_fittness(Run, labels = ['Mass', 'Flutter Velocity'])
    plot_genes(Run, GeneNames= [ 'Thickness', '$\\theta_1$', '$\\theta_2$', '$\\theta_3$'],
                ytitles = ['$Thickness \\; [m]$', '$\\vartheta_1 \\; [deg]$.', '$\\vartheta_2 \\; [deg]$', '$\\vartheta_3 \\; [deg]$'] )
    plt.show()

    f = PlotFlutter('C:/Users/vasxen/OneDrive/Thesis/code/GeneticAlgorithmResults/12.7.2024/ASW28 Wing.flt', 4)
    print(FlutterSummary('C:/Users/vasxen/OneDrive/Thesis/code/GeneticAlgorithmResults/12.7.2024/ASW28 Wing.flt').FlutterInfo())
    plt.show()

main()