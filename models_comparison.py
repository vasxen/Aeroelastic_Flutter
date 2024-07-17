import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
# import tensorflow as tf
import plotly.graph_objects as go
import keras
# import keras_tuner as kt


def plot_loss(history, title: str = ''):
    y1 = history.history['loss']
    y2 = history.history['val_loss']
    x = list(range(1, len(y1) +1))
    Loss = go.Scatter(x = x, y = y1, mode = 'lines+markers', name = 'loss')
    ValLoss = go.Scatter(x = x, y = y2, mode = 'lines+markers', name = 'validation loss')
    fig = go.Figure()
    fig.add_trace(Loss)
    fig.add_trace(ValLoss)

    fig.update_layout(title = title, xaxis_title = 'Epoch', yaxis_title = 'Error [FlutterVelocity]')
    fig.show()



def Preprocessing(dataset: pd.DataFrame, train_test_ratio:float = 0.9):
    dataset.pop('Y2')
    train = dataset.sample(frac = train_test_ratio, random_state = 44)
    test = dataset.drop(train.index)
    sb.pairplot(train[['X0', 'X1', 'X2', 'X3', 'Y1']])
    plt.show()
    print(train.describe().transpose())

    train_features = train.copy()
    test_features = test.copy()

    train_labels = train_features.pop('Y1')
    test_labels = test_features.pop('Y1')

    return train_features, train_labels, test_features, test_labels


raw_dataset = pd.read_excel('FunctionDataPoints.xlsx')
train_features, train_labels, test_features, test_labels = Preprocessing(raw_dataset)

model: keras.Model = keras.saving.load_model('keras_models/tunedmodel2.keras') #type: ignore
clonedmodel = keras.models.clone_model(model)
clonedmodel.compile(loss= 'mean_absolute_error', optimizer= keras.optimizers.Adam(0.001), metrics= ['mse' ,'mae']) #type: ignore
train_history = clonedmodel.fit(train_features, train_labels, validation_split=0.1, verbose= '0', epochs=200)
plot_loss(train_history, 'Trained 2')
_, MSE, MAE = clonedmodel.evaluate(test_features, test_labels)
print('Tuned Model 2 Info')
print(clonedmodel.summary())
print(f'Validation MSE = {MSE}')
print(f'Validation MAE = {MAE}')


