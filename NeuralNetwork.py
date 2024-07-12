import tensorflow
import keras
import pandas as pd
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.typing import NDArray

## Read Data

Data:pd.DataFrame = pd.read_excel('FunctionDataPoints.xlsx')
Data = Data.drop(['Y2'], axis = 1)
scaler = MinMaxScaler()
transform = scaler.fit(Data)

Data_tr = transform.transform(Data)

X = Data_tr[:,0:4]
Y = Data_tr[:,4:]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.1, random_state = 55) #type: ignore
TrainingPoints = Xtrain.shape[0]
TestPoints = Xtest.shape[0]


def model_builder(hp: kt.HyperParameters) -> keras.Model:
    model = keras.models.Sequential()
    # Architecture hyperparameters
    NLayers: int = hp.Int('Number_of_Layers', min_value= 1, max_value=5, step= 1) #type: ignore
    
    for i in range(NLayers):
        NNodes = hp.Choice(f'units L{i+1}', [2, 4, 8, 16, 32])
        activation = hp.Choice(f'activation L{i+1}', ['relu', 'tanh'])
        model.add(keras.layers.Dense(units= NNodes, activation = activation, name = f'Hidden_{i+1}'))
        
    model.add(keras.layers.Dense(1, activation= 'linear', name = 'Output'))

    # compilation hyperparameters
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    optimizer = keras.optimizers.Adam(learning_rate= lr) #type: ignore
    model.compile(optimizer = optimizer, #type: ignore
                      metrics = ['accuracy', 'mse'],
                      loss = 'mean_squared_error')
    
    return model

tuner = kt.Hyperband(model_builder, 
                    objective= 'val_accuracy',
                    max_epochs= 30,
                    factor=  3,
                    directory='my_dir',
                    project_name='intro_to_kt',
                    overwrite = True)

stopearly = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
tensorboard = keras.callbacks.TensorBoard('/tmp/tb_logs')
tuner.search(Xtrain, Ytrain, epochs = 50, validation_split = 0.1, verbose=2)

best_params: kt.HyperParameters = tuner.get_best_hyperparameters(num_trials=1)[0]

Finalmodel = tuner.hypermodel.build(best_params) #type: ignore
history = Finalmodel.fit(Xtrain, Ytrain, epochs = 50 , validation_split = 0.1)

LossValue, accuracy, mse = Finalmodel.evaluate(Xtest, Ytest)
print('============ FINAL MODEL TEST RESULTS =================')
Finalmodel.summary()
print('Accuracy: %.2f' % (accuracy*100))
print('MSE: %.2f' % (mse))
