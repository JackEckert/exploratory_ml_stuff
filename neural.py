import numpy as np # import
import matplotlib.pyplot as plt

# Jack Eckert - 4/04/2025

# ACTIVATION FUNCTIONS -----------------------------------------------------------------------------------------------
def sigmoid(x, derivative=False):
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s)
    return s
    
def ReLu(x, derivative=False):
    if derivative:
        return (x > 0).astype(float)
    else:
        return np.maximum(0, x)

# COST FUNCTIONS -----------------------------------------------------------------------------------------------------
def square(x, derivative=False):
    if derivative:
        return 2 * x
    else:
        return x ** 2
    
# MISC ---------------------------------------------------------------------------------------------------------------
def formatData(array, seperator: int, outputFirst = False):
    lst = []
    for row in array:
        if outputFirst:
            lst.append(Datapoint(row[seperator:], row[:seperator]))
        else:
            lst.append(Datapoint(row[:seperator], row[seperator:]))
    return np.array(lst)

# CLASSES  -----------------------------------------------------------------------------------------------------------
class Datapoint():
    def __init__(self, input, output):
        self.inputs = np.array(input)
        self.outputs = np.array(output)

class Layer():
    def __init__(self, numNodesIn: int, numNodesOut: int, activationFunction, costFunction, xavierMode):
    
        if xavierMode == "n":
            self.weightsArray = np.random.randn(numNodesIn, numNodesOut) * np.sqrt(2 / (numNodesIn + numNodesOut))
        elif xavierMode == "u":
            self.weightsArray = (np.random.random(size=(numNodesIn, numNodesOut)) - 0.5) * np.sqrt(6 / (numNodesIn + numNodesOut)) * 2
        else:
            raise ValueError("xavierMode must be 'n' for a normal distribution and 'u' for a uniform one")
        
        self.biasArray = np.zeros(shape=numNodesOut)

        self.numNodesOut = numNodesOut
        self.numNodesIn = numNodesIn
        self.activationFunction = activationFunction
        self.costFunction = costFunction
        self.preActs = None
        self.outputs = None
        self.inputs = None
        self.biasGradient = np.zeros(shape=(numNodesOut))
        self.weightsGradient = np.zeros(shape=(numNodesIn, numNodesOut))

    def calculateOutputs(self, inputs):
        self.inputs = np.array(inputs)
        self.preActs = (np.matmul(self.inputs, self.weightsArray) + self.biasArray)
        self.outputs = np.apply_along_axis(self.activationFunction, 0, self.preActs)
        return np.apply_along_axis(self.activationFunction, 0, self.preActs)
    
    def setWeights(self, newWeights):
        self.weightsArray = newWeights

    def setBiases(self, newBiases):
        self.biasArray = newBiases

    def layerCost(self, expectedOutput, actualOutput):
        return np.apply_along_axis(self.costFunction, 0, (actualOutput - expectedOutput))
    
    def updateWeights(self, learnRate, batchSize):
        self.weightsArray += learnRate * self.weightsGradient / batchSize
        self.weightsGradient = np.zeros(shape=(self.numNodesIn, self.numNodesOut))

    def updateBiases(self, learnRate, batchSize):
        self.biasArray += learnRate * self.biasGradient / batchSize
        self.biasGradient = self.biasGradient = np.zeros(shape=(self.numNodesOut))
    
class NeuralNetwork():
    def __init__(self, shape: tuple, activationFunction = sigmoid, costFunction = square, xavierMode = "n"):
        self.layers = []
        for i in range(len(shape) - 1):
            self.layers.append(Layer(shape[i], shape[i + 1], activationFunction, costFunction, xavierMode))
        self.size = len(self.layers) + 1
        self.costFunction = costFunction
 
    def calculateOutput(self, datapoint: Datapoint):
        put = datapoint.inputs
        for layer in self.layers:
            put = layer.calculateOutputs(put)
        return put
    
    def evaluatePoint(self, datapoint: Datapoint):
        output = self.calculateOutput(datapoint)
        return int(np.argmax(output, 0) == np.argmax(datapoint.outputs, 0))
    
    def evaluate(self, dataset):
        total = 0
        for datapoint in dataset:
            total += self.evaluatePoint(datapoint)
        return total
    
    def cost(self, datapoint: Datapoint, returnTotal=True):
        output = self.calculateOutput(datapoint)
        oLayer = self.layers[-1]

        if returnTotal:
            return np.sum(oLayer.layerCost(datapoint.outputs, output))
        else:
            return oLayer.layerCost(datapoint.outputs, output)
        
    def getEndVals(self, datapoint: Datapoint):
        output = self.calculateOutput(datapoint)
        oLayer = self.layers[-1]
        costDerivative = np.apply_along_axis(self.costFunction, 0, output - datapoint.outputs, True)
        activationDerivative = np.apply_along_axis(oLayer.activationFunction, 0, oLayer.preActs, True)
        return costDerivative * activationDerivative
    
    def backprop(self, datapoint: Datapoint):
        endVals = self.getEndVals(datapoint)
        self.layers.reverse() # reverses order of the layers
        for i, layer in enumerate(self.layers):
            # update weight and bias gradients
            layer.weightsGradient -= np.matmul(layer.inputs.reshape(layer.numNodesIn, 1), np.atleast_2d(endVals))
            layer.biasGradient -= endVals.flatten()

            if i + 1 >= len(self.layers):
                self.layers.reverse()
                return

            endVals = np.matmul(layer.weightsArray, endVals.reshape(layer.numNodesOut, 1))
            endVals = endVals.reshape(1, layer.numNodesIn) * np.apply_along_axis(layer.activationFunction, 0, self.layers[i + 1].preActs, True)

    def updateValues(self, learnRate, batchSize):
        for layer in self.layers:
            layer.updateWeights(learnRate, batchSize)
            layer.updateBiases(learnRate, batchSize)
            
    def train(self, dataset, learnRate, epochs, batchSize = None, targetCost = 0, targetAcc = 1.1, printMode = False, showCostPlot = False, showAccPlot = False):
        i = 0
        cost = 10000
        acc = 0
        dataset = np.array(dataset)
        datasetSize = len(dataset)

        if batchSize is None:
            batchSize = datasetSize

        batchCount = int(np.ceil(datasetSize / batchSize))

        datasetPad = np.pad(dataset, (0, batchCount * batchSize - datasetSize)).reshape(batchCount, batchSize)

        costs = []
        accs = []

        while i < epochs and cost > targetCost and targetAcc > acc:
            for batch in datasetPad:
                for datapoint in batch:
                    if datapoint == 0:
                        break
                    self.backprop(datapoint)
                
                self.updateValues(learnRate, batchSize)

            if showCostPlot or targetCost > 0:
                cost = sum([self.cost(dp) for dp in dataset])
                costs.append(cost)

            if showAccPlot or targetAcc <= 1:
                acc = self.evaluate(dataset) / len(dataset)
                accs.append(acc)

            if printMode:
                print(f"epoch {i + 1} complete")

            i += 1
        
        if showCostPlot:
            plt.plot(np.arange(i), costs)
            plt.xlabel("epoch")
            plt.ylabel("total cost")
            plt.show()

        if showAccPlot:
            plt.plot(np.arange(i), accs)
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.show()