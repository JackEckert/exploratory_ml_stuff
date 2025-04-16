import pandas as pd
import neural as nn
import numpy as np
from PIL import Image

train = pd.read_csv("readData\\train.csv")

train_encoded = pd.get_dummies(train, columns=["label"], dtype=int)

unformArray = train_encoded.to_numpy()
datapoints = nn.formatData(unformArray, separator=784)

trainData, valData = nn.splitData(datapoints, 0.85)

recognizer = nn.NeuralNetwork((784, 256, 128, 10), activationFunction=nn.ReLu, costFunction=nn.catCrossEntropy)

recognizer.train(trainData, 0.0001, 250, batchSize=32, showAccPlot=True, showCostPlot=True, printMode=True, valSet=valData, dropoutProb=0.4, annealRate=0.8)

print(recognizer.evaluate(datapoints) / len(datapoints))

nn.save(recognizer, "YAHOOOO!!!!")