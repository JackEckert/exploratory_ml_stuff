import pandas as pd
import neural as nn

train = pd.read_csv("readData\\train.csv")

train_encoded = pd.get_dummies(train, columns=["label"], dtype=int)

unformArray = train_encoded.to_numpy()
datapoints = nn.formatData(unformArray, separator=784)

r = nn.load("testSave002.npz")
print(r.evaluate(datapoints) / len(datapoints))
exit()

trainData, testData = nn.splitTrainTest(datapoints, 0.85)

recognizer = nn.NeuralNetwork((784, 256, 128, 10), activationFunction=nn.ReLu, costFunction=nn.catCrossEntropy)

recognizer.train(trainData, 0.0001, 150, batchSize=32, showAccPlot=True, showCostPlot=True, printMode=True, testSet=testData)

nn.save(recognizer, "testSave002")