import pandas as pd
import neural as nn

print(help(nn._Layer))
exit()

train = pd.read_csv("readData\\train.csv")

train_encoded = pd.get_dummies(train, columns=["label"], dtype=int)

unformArray = train_encoded.to_numpy()
datapoints = nn.formatData(unformArray, separator=784)

recognizer = nn.NeuralNetwork((784, 128, 64, 10), activationFunction=nn.ReLu, initializeMode='u')
recognizer.layers[-1].setActivationFunction(nn.sigmoid)

recognizer.train(datapoints, 0.005, 100, batchSize=32, showAccPlot=True, printMode=True)

nn.save(recognizer, "recognizer03")