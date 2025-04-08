import pandas as pd
import neural as nn

train = pd.read_csv("readData\\train.csv")

train_encoded = pd.get_dummies(train, columns=["label"], dtype=int)

unformArray = train_encoded.to_numpy()
datapoints = nn.formatData(unformArray, separator=784)

recognizer = nn.NeuralNetwork((784, 32, 32, 10), activationFunction=nn.ReLu, initializeMode='u')
recognizer.layers[-1].setActivationFunction(nn.sigmoid)

recognizer.train(datapoints, 0.01, 20, batchSize=32, showAccPlot=True, printMode=True)

nn.save(recognizer, "recognizer02")