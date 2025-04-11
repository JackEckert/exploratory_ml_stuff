import pandas as pd
import neural as nn

train = pd.read_csv("readData\\train.csv")

train_encoded = pd.get_dummies(train, columns=["label"], dtype=int)

unformArray = train_encoded.to_numpy()
datapoints = nn.formatData(unformArray, separator=784)

recognizer = nn.NeuralNetwork((784, 128, 64, 10), activationFunction=nn.ReLu, costFunction=nn.catCrossEntropy)

recognizer.train(datapoints, 0.0001, 20, batchSize=32, showAccPlot=True, showCostPlot=True, printMode=True)