import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import keras

def ResultEvalutaion(ytrue, ypred):
    fig1, ax1 = plt.subplots()
    ax1.scatter(ytrue, ypred, s=10, color="blue", alpha=0.6, label="Predicted vs Actual")
    ax1.plot(ytrue, ytrue, color='black', label="Ideal Fit")  # Ideal line
    ax1.set_title("Actual vs Predicted")
    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")
    ax1.legend()
    ax1.grid(which = 'both')

    # Residuals calculation
    Residuals = ytrue - ypred
    
    # Scatter plot: ytrue vs Residuals
    fig2, ax2 = plt.subplots()
    ax2.scatter(ytrue, Residuals, s=10, color="red", alpha=0.6, label="Residuals")
    ax2.axhline(0, color='black', linestyle='--', label="Zero Residual Line")
    ax2.set_title("Residuals vs Actual Values")
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Residuals")
    ax2.legend()
    ax2.grid(True)

    plt.show()


def main():
    raw_dataset = pd.read_excel('FunctionDataPoints.xlsx')
    Ytrue = raw_dataset['Y1'].to_numpy()
    model: keras.Model = keras.saving.load_model('keras_models/tunedmodel2.keras') #type: ignore
    Ypred:NDArray  = model.predict(raw_dataset[['X0','X1','X2','X3']])
    Ypred = Ypred.flatten()
    ResultEvalutaion(Ytrue,Ypred)


main()
    





    


