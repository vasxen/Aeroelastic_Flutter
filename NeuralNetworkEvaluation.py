import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import keras
from bokeh.plotting import figure, show
from bokeh.io import export_svg, export_png

# def ResultEvalutaion(ytrue, ypred):
#     fig1, ax1 = plt.subplots()
#     ax1.scatter(ytrue, ypred)
#     ax1.plot(ytrue, ytrue)
#     ax1.set
#     ### 
#     Residuals = ytrue - ypred
#     fig2, ax2 = plt.subplots()
#     ax2.scatter(ytrue, Residuals)
#     plt.show()

def ResultEvalutaion(ytrue, ypred):
    p = figure(width=800, height=800)
    p.scatter(ytrue,ypred,size=2)
    p.line(ytrue,ytrue, color='black')
    p.output_backend = 'svg' # type: ignore
    export_svg(p,filename= 'test.svg')
    # show(p)
    # export_png(p,filename= 'test.png')

    ### 
    Residuals = ytrue - ypred
    p2 = figure(width=800, height=800)
    p2.scatter(ytrue, Residuals, size = 3)
    p2.output_backend = 'svg' #type: ignore


def main():
    raw_dataset = pd.read_excel('FunctionDataPoints.xlsx')
    Ytrue = raw_dataset['Y1'].to_numpy()
    model: keras.Model = keras.saving.load_model('keras_models/tunedmodel2.keras') #type: ignore
    Ypred:NDArray  = model.predict(raw_dataset[['X0','X1','X2','X3']])
    Ypred = Ypred.flatten()
    ResultEvalutaion(Ytrue,Ypred)


main()
    





    


