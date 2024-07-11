import tensorflow
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

## Read Data

Data = pd.read_excel('FunctionDataPoints.xlsx')
scaler = MinMaxScaler()
transform = scaler.fit(Data)

Data_tr = transform.transform(Data)

Data2 = transform.inverse_transform(Data_tr)

X = Data_tr[:,0:4]
Y = Data_tr[:,4:]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.1, random_state = 55)

NeuralNetwork = keras.models.Sequential([keras.Input(shape = (4,),name = 'Input'),
                                        keras.layers.Dense(8, activation= 'relu', name = 'Hidden_1'),
                                        keras.layers.Dense(4, activation= 'relu', name = 'Hidden_2'),
                                        keras.layers.Dense(2, activation= 'linear', name = 'Output'),
                                        ])


NeuralNetwork.compile(optimizer = 'adam',
                      metrics = ['accuracy', 'mse'],
                      loss = 'mean_squared_error')

NeuralNetwork.fit(x = Xtrain, y = Ytrain, epochs = 2, batch_size = 5)

LossValue, accuracy, mse = NeuralNetwork.evaluate(Xtest, Ytest)
print('Accuracy: %.2f' % (accuracy*100))

Ypred = NeuralNetwork.predict(Xtest)
print('')