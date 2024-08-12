import pickle
import pygad
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
import typing
from typing import TypeVar, Generic, cast
Times_New_Roman = {'fontname': 'Times New Roman'}

DType = TypeVar("DType")
class Array(np.ndarray, Generic[DType]):
    def __getitem__(self, key) -> DType:
        return super().__getitem__(key) # type: ignore

def plot_fittness(GA: pygad.GA, title: str = '', labels: List[str] = []):
    NumGenerations: int = GA.generations_completed + 1 #type: ignore
    if NumGenerations > 0:
        fig, ax = plt.subplots()
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
            ax.plot(x, y, label = name)

        ax.set_title(title)
        ax.set_xlabel('Num. Generation', Times_New_Roman )
        ax.set_ylabel('Fittness', Times_New_Roman )
        plt.show()
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
    fig, axess = plt.subplots(N, 1, figsize = (8,6) ,sharex = True)
    axess = cast(Array[Axes], axess)
    
    for i in range(N):
        y = data[:,i]
        ax = axess[i]
        ax.scatter(x, y, s = 2, label = GeneNames[i])
        ax.set_ylabel(ytitles[i])
        ax.legend(loc = 'upper right')
        ax.grid()
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-1, 3), useMathText=True)
        if i != 0:
            ax.set_ylim((-90,90))
            ax.set_yticks(np.arange(-90,91,45))
            ax.legend(loc = 'best')

    ax.set_xlabel('Num. Generation', Times_New_Roman, fontsize = 16)

    plt.show()


with open('GeneticAlgorithmResults/12.7.2024/Genetic_Algorithm_Run.pkl', 'rb') as f:
    Run: pygad.GA = pickle.load(f)

# plot_fittness(Run, labels = ['Mass', 'Flutter Velocity'])
plot_genes(Run, GeneNames= [ 'Thickness', '$\\theta_1$', '$\\theta_2$', '$\\theta_3$'],
            ytitles = ['m', 'deg.', 'deg.', 'deg.'] )
# Run.plot_genes(plot_type = 'scatter', solutions = 'best', )