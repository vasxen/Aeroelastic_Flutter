import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pickle
import pandas as pd
import numpy as np
from typing import TypeVar, Generic, cast
from optimization import FlutterSubcase, FlutterSummary, FlutterAnalysisPoint
from itertools import cycle
Times_New_Roman = {'fontname': 'Times New Roman'}

DType = TypeVar("DType")
class Array(np.ndarray, Generic[DType]):
    def __getitem__(self, key) -> DType:
        return super().__getitem__(key) # type: ignore
    
def Preprocessing(Optimization_History: pd.DataFrame) -> pd.DataFrame:
    Optimization_History[['t1', 't2', 't3', 'theta1', 'theta2', 'theta3']] = Optimization_History['Input Vector'].str.strip('()').str.split(', ', expand=True)
    Optimization_History[['t1', 't2', 't3', 'theta1', 'theta2', 'theta3']] = Optimization_History[['t1', 't2', 't3', 'theta1', 'theta2', 'theta3']].astype(float)
    Optimization_History = Optimization_History.drop(['Input Vector'], axis=1)
    return Optimization_History

def plotObjective(Optimization_History: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.plot(Optimization_History['Function Value'], marker = '.', color = 'black')
    ax.set_ylim((min(Optimization_History['Function Value'].to_list())-2, 100))
    ax.grid(which = 'both')
    ax.set_title('Optimization History',Times_New_Roman)
    ax.set_xlabel('Iteration',Times_New_Roman)
    ax.set_ylabel('Objective Value',Times_New_Roman)
    return fig

def plotInputVector(Optimization_History: pd.DataFrame):
    fig, axes = plt.subplots(4,1, sharex= True)
    axes = cast(Array[Axes], axes)
    vars = ['t1', 'theta1', 'theta2', 'theta3']
    for i, var in enumerate(vars):
        ax = axes[i]
        ax.plot(Optimization_History[var], label = var, color ='black', marker = '.')
        ax.grid(which = 'both')
        ax.legend()

        if var != 't1':
            ax.set_ylim((-90,90))
            ax.set_yticks(np.arange(-90,91,45))
            ax.set_ylabel('deg', Times_New_Roman)
        else:
            ax.set_ylabel('Thickness [m]', Times_New_Roman)
            ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-1, 3), useMathText=True)
    
    ax.set_xlabel('Iteration', Times_New_Roman)
    fig.suptitle('Optimization Variables', fontname = 'Times New Roman')

    return fig

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
    #======================================================================================
    # Plot Results with ALL optimization variables (objective function 1)
    #======================================================================================
    resultsdir1 = 'C:/Users/vasxen/OneDrive/Thesis/code/results/Powell Results 25_8_2024'
    # Plot optimization history 
    Optimization_History = pd.read_excel(resultsdir1 + '/OptimizationHistory.xlsx')
    Optimization_History = Preprocessing(Optimization_History)
    Objfig = plotObjective(Optimization_History)     # Objective function
    Varfig = plotInputVector(Optimization_History)     #Variables
    Objfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Powell/Objective1.svg', format = 'svg')
    Varfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Powell/Variables1.svg', format = 'svg')

    # Plot the first 4 modes
    Flutterfig = PlotFlutter(resultsdir1 + '/ASW28 Wing.flt', lastmode = 4)
    Flutterfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Powell/Flutter1.svg')
    
    # Plot the first divergent mode (MODE 3)
    Flutterfig = PlotFlutter(resultsdir1 + '/ASW28 Wing.flt', firstmode = 2, lastmode = 3)
    Flutterfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Powell/Flutter1Mode3.svg')


    #======================================================================================
    # Plot Results with angle variables only (objective function 2)
    #======================================================================================

    resultsdir2 = 'C:/Users/vasxen/OneDrive/Thesis/code/results/Powell Results 12_8_2024'
    # Plot optimization history 
    Optimization_History = pd.read_excel(resultsdir2 + '/OptimizationHistory.xlsx')
    Optimization_History = Preprocessing(Optimization_History)
    Objfig = plotObjective(Optimization_History)     # Objective function
    Varfig = plotInputVector(Optimization_History)     #Variables
    Objfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Powell/Objective2.svg', format = 'svg')
    Varfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Powell/Variables2.svg', format = 'svg')

    # Plot the first 4 modes
    Flutterfig = PlotFlutter(resultsdir2 + '/ASW28 Wing.flt', lastmode = 4)
    Flutterfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Powell/Flutter2.svg')

    # Plot the first divergent mode (MODE 3)
    Flutterfig = PlotFlutter(resultsdir2 + '/ASW28 Wing.flt', firstmode = 2, lastmode = 3)
    Flutterfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Powell/Flutter2Mode3.svg')


    #======================================================================================
    # Plot Initial Flutter solution
    #======================================================================================
    
    Flutterfig = PlotFlutter("C:/Users/vasxen/OneDrive/Thesis/GiagkosModel/Attempt18/ASW.flt", 4)
    Flutterfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Flutter/InitialFlutter.svg', format = 'svg')
    Flutterfig = PlotFlutter("C:/Users/vasxen/OneDrive/Thesis/GiagkosModel/Attempt18/ASW.flt", 3,2)
    Flutterfig.savefig('C:/Users/vasxen/OneDrive/Thesis/images/Results/Flutter/InitialFlutterMode3.svg', format = 'svg')
    print(FlutterSummary("C:/Users/vasxen/OneDrive/Thesis/GiagkosModel/Attempt18/ASW.flt").FlutterInfo())


    # plt.show()



main()

