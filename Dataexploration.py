import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb



def DataProcessing(dataset: pd.DataFrame):
    Pairgrid = sb.pairplot(dataset[['Thickness', 'θ1', 'θ2', 'θ3', 'Flutter Velocity']], diag_kws={'color':'black'})
    Pairgrid.figure.subplots_adjust(top=0.95)
    Pairgrid.figure.suptitle('pair plot', fontsize = 16)

    Pairgrid.figure.savefig('results/DataExploration/pairplot.svg')
    dataset.describe().transpose().to_excel('results/DataExploration/statistics.xlsx')

def Pareto(dataset: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.scatter(dataset['Flutter Velocity'], dataset['Mass'], alpha= 0.6, marker= '.')
    ax.set_xlabel('Flutter Velocity')
    ax.set_ylabel('Mass')
    ax.set_title('Pareto Front')
    ax.grid(which= 'both')
    ax.invert_yaxis()
    plt.show()
    
    dataset = dataset.sort_values(['Mass', 'Flutter Velocity'])
    a_changes = dataset['Mass'] != dataset['Mass'].shift(-1)
    best_solutions = dataset[a_changes]
    best_solutions.to_excel('results/DataExploration/beset_solutions.xlsx')
    

def main():
    raw_dataset = pd.read_excel('FunctionDataPoints.xlsx')
    raw_dataset.columns = ['Thickness', 'θ1', 'θ2', 'θ3', 'Flutter Velocity', 'Mass']
    # dataset = DataProcessing(raw_dataset)
    Pareto(raw_dataset)


  

main()