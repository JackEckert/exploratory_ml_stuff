import numpy as np # import
import matplotlib.pyplot as plt

# Jack Eckert - 4/04/2025

# ACTIVATION FUNCTIONS -----------------------------------------------------------------------------------------------

'''
ACTIVATION FUNCTIONS

Args:
x -> input
derivative -> calculates the derivative at the given input if True
'''

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

'''
COST FUNCTIONS

Args:
x -> input
derivative -> calculates the derivative at the given input if True

Returns: float
'''

def square(x, derivative=False):
    if derivative:
        return 2 * x
    else:
        return x ** 2
    
# MISC ---------------------------------------------------------------------------------------------------------------
def formatData(array, separator: int, outputFirst = False):

    '''
    Splits each row of a 2D numpy array into inputs and outputs, then returns an array of datapoint objects using them.
    Returned array should be equal in size to number of rows in the input array

    Args:
    array -> 2D array of your data
    separator -> index of start of the output.
    outputFirst -> flips input and output. Use if your output comes before input in your data.
    '''

    lst = []
    for row in array:
        if outputFirst:
            lst.append(Datapoint(row[separator:], row[:separator]))
        else:
            lst.append(Datapoint(row[:separator], row[separator:]))
    return np.array(lst)

def save(neuralNetwork, filepath):

    lst = [np.array(neuralNetwork.shape), np.array(neuralNetwork.costFunction)]

    for layer in neuralNetwork.layers:
        lst.extend([layer.weightsArray, layer.biasArray, np.array(layer.activationFunction)])

    np.savez(filepath, *lst)

def load(filepath):
    a = np.load(filepath, allow_pickle=True)

    nn = NeuralNetwork(tuple(a["arr_0"]), costFunction=a["arr_1"])

    for i, layer in enumerate(nn.layers):
        j = i * 3
        layer.setWeights(a[f"arr_{j + 2}"])
        layer.setBiases(a[f"arr_{j + 3}"])
        layer.setActivationFunction(a[f"arr_{j + 4}"])

    return nn


# CLASSES  -----------------------------------------------------------------------------------------------------------
class Datapoint():
    def __init__(self, input, output):

        '''
        Initializes datapoint object

        Args:
        input -> input of the datapoint
        output -> expected output of the datapoint
        '''

        self.inputs = np.array(input)
        self.outputs = np.array(output)

class _Layer():

    '''
    Represents a layer of the neural network. A 'layer' in this implementation is a set of nodes and their incoming weights.

    Attributes:
    weightsArray -> matrix of incoming weights as a numpy array
    biasArray -> vector of layer's biases as a numpy array
    numNodesOut -> # of nodes in layer.
    numNodesIn -> # of nodes in preceding layer
    activationFunction -> activation function for the layer
    outputs -> values of the outputs of the nodes
    preActs -> values of the outputs before being inputted into the activation function
    inputs -> values of the inputs to the layer. Equal to the output of the previous layer.

    biasGradient -> current calculated matrix of gradients to add to the biases
    '''

    def __init__(self, numNodesIn: int, numNodesOut: int, activationFunction, initializeMode):

        '''
        Initializes layer object.

        Args:
        numNodesIn -> # of input nodes. Should be the same as the # of nodes of the preceding layer.
        numNodesOut -> # of nodes in layer.
        activationFunction -> activation function for the layer
        initializeMode -> specifications for how to initialize the weights
            'n' -> initialize use a standard normal xavier initialization
            'u' -> initialize use a standard uniform xavier initialization
        '''
    
        if initializeMode == "n":
            self.weightsArray = np.random.randn(numNodesIn, numNodesOut) * np.sqrt(2 / (numNodesIn + numNodesOut))
        elif initializeMode == "u":
            self.weightsArray = (np.random.random(size=(numNodesIn, numNodesOut)) - 0.5) * np.sqrt(6 / (numNodesIn + numNodesOut)) * 2
        else:
            raise ValueError("xavierMode must be 'n' for a normal distribution and 'u' for a uniform one")
        
        self.biasArray = np.zeros(shape=numNodesOut)

        self.numNodesOut = numNodesOut
        self.numNodesIn = numNodesIn
        self.activationFunction = activationFunction
        self.outputs = None
        self.preActs = None
        self.inputs = None
        self.weightsGradient = np.zeros(shape=(numNodesIn, numNodesOut))
        self.biasGradient = np.zeros(shape=(numNodesOut))

    def _calculateOutputs(self, inputs):
        self.inputs = np.array(inputs)
        self.preActs = (np.matmul(self.inputs, self.weightsArray) + self.biasArray)
        self.outputs = np.apply_along_axis(self.activationFunction, 0, self.preActs)
        return np.apply_along_axis(self.activationFunction, 0, self.preActs)
    
    def setWeights(self, newWeights):
        self.weightsArray = newWeights

    def setBiases(self, newBiases):
        self.biasArray = newBiases

    def _layerCost(self, expectedOutput, actualOutput, costFunction):
        return np.apply_along_axis(costFunction, 0, (actualOutput - expectedOutput))
    
    def updateWeights(self, learnRate, batchSize):
        self.weightsArray += learnRate * self.weightsGradient / batchSize
        self.weightsGradient = np.zeros(shape=(self.numNodesIn, self.numNodesOut))

    def updateBiases(self, learnRate, batchSize):
        self.biasArray += learnRate * self.biasGradient / batchSize
        self.biasGradient = self.biasGradient = np.zeros(shape=(self.numNodesOut))

    def setActivationFunction(self, newAct):
        self.activationFunction = newAct
    
class NeuralNetwork():
    def __init__(self, shape: tuple, activationFunction = sigmoid, costFunction = square, initializeMode = "n"):
        self.layers = []
        for i in range(len(shape) - 1):
            self.layers.append(_Layer(shape[i], shape[i + 1], activationFunction, initializeMode))
        self.size = len(self.layers) + 1
        self.shape = shape
        self.costFunction = costFunction
 
    def calculateOutput(self, datapoint: Datapoint):
        put = datapoint.inputs
        for layer in self.layers:
            put = layer._calculateOutputs(put)
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
            return np.sum(oLayer._layerCost(datapoint.outputs, output, self.costFunction))
        else:
            return oLayer._layerCost(datapoint.outputs, output, self.costFunction)
        
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