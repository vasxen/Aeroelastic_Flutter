import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
# import tensorflow as tf
import plotly.graph_objects as go
import keras
import keras_tuner as kt

def plot_loss(history):
  y1 = history.history['loss']
  y2 = history.history['val_loss']
  x = list(range(1, len(y1) +1))
  Loss = go.Scatter(x = x, y = y1, mode = 'lines+markers', name = 'loss')
  ValLoss = go.Scatter(x = x, y = y2, mode = 'lines+markers', name = 'validation loss')
  fig = go.Figure()
  fig.add_trace(Loss)
  fig.add_trace(ValLoss)

  fig.update_layout(xaxis_title = 'Epoch', yaxis_title = 'Error [FlutterVelocity]')
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

def staticmodel(normalizer: keras.Layer):
	model = keras.Sequential([
    normalizer,
    keras.layers.Dense(64, activation= 'relu'),
    keras.layers.Dense(64, activation= 'relu'),
    keras.layers.Dense(64, activation= 'relu'),
    keras.layers.Dense(1)
  ])
	model.compile(loss= 'mean_absolute_error', optimizer= keras.optimizers.Adam(0.001), metrics= ['mse' ,'mae']) #type: ignore
	return model


def main():
    def modelbuilder(hp: kt.HyperParameters):
        model = keras.Sequential()
        model.add(normalizer)

        NUMBER_LAYERS = hp.Int('N_Hidden_layers', 1, 10, 1)
        for i in range(NUMBER_LAYERS): #type: ignore
            UNITS = hp.Choice(f'Units L{i+1}', [2, 4, 8, 16, 32, 64, 128, 256, 512])
            model.add(keras.layers.Dense(UNITS, activation = 'relu'))
        
        model.add(keras.layers.Dense(1))
        LR = hp.Choice('learning rate', [1e-2, 1e-3, 1e-4])
        model.compile(loss= 'mean_absolute_error', optimizer= keras.optimizers.Adam(LR), metrics= ['mse' ,'mae']) #type: ignore
        return model
  
    raw_dataset = pd.read_excel('FunctionDataPoints.xlsx')
    train_features, train_labels, test_features, test_labels = Preprocessing(raw_dataset)

    # Data normalization
    normalizer = keras.layers.Normalization(axis = -1)
    normalizer.adapt(np.array(train_features))

    # model = staticmodel(normalizer)

    # train_history = model.fit(train_features, train_labels, validation_split=0.1, verbose= '2', epochs=100)
    # model.save('1Hiddenlayers.keras')
    # eval = model.evaluate(test_features, test_labels)
    # print(eval)
    # plot_loss(train_history)
    # print('===================================================')

    ## Tune Model
    tuner = kt.Hyperband(modelbuilder,
                objective= 'val_mae',
                max_epochs = 80,
                directory = 'C:/Users/vasxen/OneDrive/Thesis',
                project_name = 'Hypermodel_Tuning2')
    
    tensorboard = keras.callbacks.TensorBoard('C:/Users/vasxen/OneDrive/Thesis/tb_logs')
    tuner.search(train_features, train_labels, validation_data = (test_features, test_labels), callbacks= [tensorboard])

    bestmodel = tuner.get_best_models(1)[0]
    bestmodel.save('tunedmodel2.keras')
    print('Tuned model Info')
    print(bestmodel.summary())
    print(bestmodel.evaluate(test_features, test_labels))

if __name__ == '__main__':
    main()

# Run this command to start tensorboard
# python -m tensorboard.main --logdir=tb_logs