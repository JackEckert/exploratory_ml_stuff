import pandas as pd
import neural as nn

df = pd.read_csv("readData\\train.csv")

df_encoded = pd.get_dummies(df, columns=["label"], dtype=int)

unformArray = df_encoded.to_numpy()
datapoints = nn.formatData(unformArray, seperator=784)

recognizer = nn.NeuralNetwork((784, 32, 32, 10))

recognizer.train(datapoints, 1, 100, batchSize=50, showAccPlot=True, printMode=True)