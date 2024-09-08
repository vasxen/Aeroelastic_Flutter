import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import keras
from scipy.signal import savgol_filter


def ResultEvalutaion(ytrue, ypred):
    fig1, ax1 = plt.subplots()
    ax1.scatter(ytrue, ypred, s=10, color="blue", alpha=0.6, label="Predicted vs Actual")
    ax1.plot(ytrue, ytrue, color='black', label="Ideal Fit")  # Ideal line
    ax1.set_title("Actual vs Predicted")
    ax1.set_xlabel("True Flutter Speed")
    ax1.set_ylabel("Predicted Flutter Speed")
    ax1.legend()
    ax1.grid(which = 'both')

    # Residuals calculation
    Residuals = ytrue - ypred
    
    # Scatter plot: ytrue vs Residuals
    fig2, ax2 = plt.subplots()
    ax2.scatter(ytrue, Residuals, s=10, color="red", alpha=0.6, label="Residuals")
    ax2.axhline(0, color='black', linestyle='--', label="Zero Residual Line")
    ax2.set_title("Residuals vs Actual Values")
    ax2.set_xlabel("Actual Flutter Speed")
    ax2.set_ylabel("Residuals")
    ax2.legend()
    ax2.grid(True)

    return fig1, fig2

def plot_loss(history, title: str = ''):
    y1 = history.history['loss']
    y2 = history.history['val_loss']
    x = list(range(1, len(y1) +1))
    fig, ax = plt.subplots()
    l1, = ax.plot(x, y1, marker = '.', label='Training Loss')
    ax.plot(x, savgol_filter(y1, int(len(x)/5), 2), linestyle = '--', marker = None, color = l1.get_color())

    l2, = ax.plot(x, y2, marker = '.', label='Validation Loss')
    ax.plot(x, savgol_filter(y2, int(len(x)/5), 2), linestyle = '--', marker = None, color = l2.get_color())

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(which='both')
    return fig

def Preprocessing(dataset: pd.DataFrame, train_test_ratio:float = 0.9):
    dataset.pop('Y2')
    train = dataset.sample(frac = train_test_ratio, random_state = 44)
    test = dataset.drop(train.index)
    train_features = train.copy()
    test_features = test.copy()

    train_labels = train_features.pop('Y1')
    test_labels = test_features.pop('Y1')

    return train_features, train_labels, test_features, test_labels

def Evaluate(modelpath: str, savepath: str, epochs: int):

    model: keras.Model = keras.saving.load_model(modelpath) #type: ignore
    raw_dataset = pd.read_excel('FunctionDataPoints.xlsx')
    # retrain model to see loss history
    train_features, train_labels, test_features, test_labels = Preprocessing(raw_dataset)
    history = model.fit(train_features, train_labels, validation_data = (test_features, test_labels), epochs=epochs)
    fig1 = plot_loss(history, title='MAE Training History')

    # Evaluate predictions on the whole dataset
    Ytrue = raw_dataset['Y1'].to_numpy()
    Ypred:NDArray  = model.predict(raw_dataset[['X0','X1','X2','X3']])
    Ypred = Ypred.flatten()
    fig2, fig3 = ResultEvalutaion(Ytrue,Ypred)

    fig1.savefig(savepath + '/traininghistory.svg')
    fig2.savefig(savepath + '/Ytrue_vs_Ypred.svg')
    fig3.savefig(savepath + '/Residuals_vs_Ytrue.svg')


    # Plot model
    keras.utils.plot_model(model, savepath +'/structure.png', show_shapes= True, show_layer_activations= True, show_layer_names= False, show_trainable= False )


def main():
    # Evaluate(modelpath = 'keras_models/tunedmodel_24_09_2024_a.keras', savepath= 'results/tunedmodel_24_09_2024', epochs = 200)
    Evaluate(modelpath = 'keras_models/1Hiddenlayers.keras', savepath= 'results/1HL', epochs= 400)
    Evaluate(modelpath = 'keras_models/2Hiddenlayers.keras', savepath= 'results/2HL', epochs= 350)
    Evaluate(modelpath = 'keras_models/4Hiddenlayers.keras', savepath= 'results/4HL', epochs = 300)
    Evaluate(modelpath = 'keras_models/6Hiddenlayers.keras', savepath= 'results/6HL', epochs = 250)

    
main()

