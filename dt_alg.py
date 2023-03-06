import pandas as pd
import numpy as np
import math
import sklearn.model_selection
import matplotlib.pyplot as plt

# load the house votes data file using python csv module
data = pd.read_csv('house_votes_84.csv')
dataCols = data.columns.tolist()

# houseVotesData: List to hold the iris data    
houseVotesData = data.to_numpy().astype(int)

# split the data into training and testing sets
# data: file data; ranNum: ensures reproducibilty; trainDt: resulting data set for training; testDt: resulting data set for testing 
def split(data, ranNum):
    # the training set will contain 80% of the data, the testing set will contain 20% of the data
    trainDt, testDt = sklearn.model_selection.train_test_split(data, train_size = 0.8, test_size = 0.2, shuffle = True, random_state = ranNum)
    return trainDt.T, testDt.T

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Testing criterias

# Q2
# calculates information gain (ID3)
# dataAttr: 2D array that represents the data, row contains data and column contains the attribute to the data instance; colAttr: names of columns in dataArr
def calcID3(dataAttr, colAttr):

    # takes a column of data as input and returns the entropy of the column
    # column: the column of data being analyzed
    def id3EntropyCalc(column):
        # initialize a dictionary to count the occurrences of each value in the input column
        cnts = {}
        for val in column:
            cnts[val] = cnts.get(val, 0) + 1
        # calculate entropy
        fEntropy = sum(-c / len(column) * math.log2(c / len(column)) for c in cnts.values())
        return fEntropy
    
    # entropy of the target variable
    trgtVar = dataAttr[-1]
    initEntropy = id3EntropyCalc(trgtVar)
    # variables for finding the best attribute
    minEntropy = float('inf')
    finalAtr = None
    # iterate over all attributes except the target variable
    for i, attr in enumerate(colAttr[:-1]):
        # extract unique values of the attribute
        attrVals = np.unique(dataAttr[i])
        # split the data based on the attribute
        attrData = [trgtVar[dataAttr[i] == value] for value in attrVals]
        # calculate entropy of the split data
        attrDataEntropy = sum(len(subset) / len(trgtVar) * id3EntropyCalc(subset) for subset in attrData)
        # update the best attribute if the split entropy is smaller
        if attrDataEntropy < minEntropy:
            minEntropy = attrDataEntropy
            finalAtr = attr
    # return the best attribute and the information gain
    infoGain = initEntropy - minEntropy
    return finalAtr, infoGain

# QE
# calculates Split Gini Index (CART)
# dataAttr: 2D array that represents the data, row contains data and column contains the attribute to the data instance; colAttr: names of columns in dataArr
def calcCart(dataAttr, colAttr):

    # calculates the Gini index of a given column using the formula
    # column: the column of data being analyzed
    def giniIndexCalc(column):
        # initialize a dictionary to count the occurrences of each value in the input column
        cnts = {}
        for val in column:
            cnts[val] = cnts.get(val, 0) + 1
        tCnt = len(column)
        # compute the gini impurity value
        giniIndexVal = 1 - sum((count / tCnt) ** 2 for count in cnts.values())
        return giniIndexVal
    # variables for finding the best attribute
    minGini = float('inf')
    finalAttr = None
    
    # iterate over all attributes except the target variable
    for attr in colAttr[:-1]:
        # extract unique values of the attribute
        attrVals = set(dataAttr[colAttr.index(attr)])
        giniVar = 0
        # split the data based on the attribute
        for value in attrVals:
            attrVals = [i for i in range(len(dataAttr[0])) if dataAttr[colAttr.index(attr)][i] == value]
            attrLabels = [dataAttr[-1][i] for i in attrVals]
            giniVar += len(attrLabels) / len(dataAttr[-1]) * giniIndexCalc(attrLabels)
        # update the best attribute if the split gini index is smaller
        if giniVar < minGini:
            minGini = giniVar
            finalAttr = attr
    # return the best attribute and the gini index
    return finalAttr, minGini
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Construct Decision Tree  
     
# Descision Tree Class
class descisionTreeClass:
    # used to store the label for the node
    nodeLabel = -float('inf')
    # used to store the majority event label if the node is a leaf node
    majEvent = -float('inf')
    # used to store the type of node, such as a start node/leaf node
    nodeType = ""
    # used to store the name of the feature being tested
    testClass = ""
    # used to store a dictionary of child nodes
    nodeEdges = {}

    # intial instance 
    def __init__(self, nodeType):
        self.nodeType = nodeType
        self.nodeLabel  = -float('inf')
        self.majEvent = -float('inf')
        self.testClass = ""
        self.nodeEdges = {}

def constructDecisionTree(data, colAttr, algor):
    # initialize decision tree node
    node = descisionTreeClass(nodeType = "start")
    # the value that occurs most frequently in that column
    vals, cnts = np.unique(data[-1], return_counts = True)
    # find the index of the most frequently occurring value in the counts array
    valIndex = np.argmax(cnts)
    # the value in the vals array at the same index as the most frequent value in the cnts array
    inpVals = vals[valIndex]

    node.majEvent = inpVals
    # check if all labels in data are same
    if len(set(data[-1])) == 1:
        # if true, create leaf node with the label
        node.nodeType, node.nodeLabel  = 'endPoint', data[-1][0]
        return node
    # check if there are no more attributes to split on
    elif not colAttr:
        # if true, create leaf node with majority event as label
        node.nodeType, node.nodeLabel  = 'endPoint', node.majEvent
        return node

    # find best attribute to split on using specified algorithm
    # returns the attribute to split the data and the corresponding metric value
    splitDataAlgor = calcID3(data, colAttr) if algor == "id3" else calcCart(data, colAttr)
    bestAttr = splitDataAlgor[0]
    # set test class and index of best attribute
    node.testClass, bestAttrIdx = bestAttr, colAttr.index(bestAttr)
    # new list of attributes without best attribute
    colAttrSamp = colAttr[:bestAttrIdx] + colAttr[bestAttrIdx + 1:]

    # construct edges for the current node
    nEdges = {}
    for nVal in set(data[bestAttrIdx]):
        index = [idx for idx, e in enumerate(data[bestAttrIdx]) if e == nVal]
        # create subset of data for each value of best attribute
        subdata = np.delete(data.T[index].T, bestAttrIdx, axis=0)
        # check if subset is empty
        if subdata.size == 0:
            # if true, reate leaf node with majority event as label
            node.nodeType, node.nodeLabel  = 'endPoint', inpVals
            return node
        # recursively create subtree for subset
        subtree = constructDecisionTree(subdata, colAttrSamp, algor)
        # add subtree to edge
        nEdges[nVal] = subtree
    # set edges of current node
    node.nodeEdges = nEdges
    
    return node

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Accuracy Section

# returns a tuple that contains a boolean value indicating whether the prediction was correct or not, the predicted value, and the actual value
# tree: decision tree; data: instance of data as input
def boolPred(tree: descisionTreeClass, data):
    # get the correct value from the last element of the instance
    corrVal = data[-1]
    # traverse the decision tree until a leaf node is reached
    while tree.nodeType != 'endPoint':
        # get the index of the test class for the current node
        tClassIdx = dataCols.index(tree.testClass)
        # check if the value of the test class for the instance is not in the edges of the current node
        if data[tClassIdx] not in tree.nodeEdges:
            # if not, return the majority event label of the current node and whether it matches the correct value
            predVal = tree.majEvent
            return predVal == corrVal, predVal, corrVal
        # move to the next node based on the value of the test class in the instance
        tree = tree.nodeEdges[data[tClassIdx]]
    # return the predicted value and whether it matches the correct value
    predVal = tree.nodeLabel
    return predVal == corrVal, predVal, corrVal


# calculates the accuracy of a decision tree model on a given dataset
# algorithm: string indicating the decision tree algorithm to use; ranNum: integer used for generating random splits; data: string indicating whether to use train or test data
def accuracyData(algorithm, ranNum, train=True):
    # calculate accuracy of a decision tree model on a given dataset
    def accuracy(data, tree):
        return sum(boolPred(tree, ins)[0] for ins in data.T) / len(data.T)
        
    accVals = []
    for i in range(1, 101):
        # split the data into training and test sets using a random seed
        trainDt, testDt = split(houseVotesData, 12200 + ranNum * i)
        dtTree = constructDecisionTree(trainDt, dataCols, algorithm)
        # calculate the accuracy of the model on the specified data
        calcAccuracy = accuracy(trainDt, dtTree) if train == True else accuracy(testDt, dtTree) if train == False else None
        accVals.append(calcAccuracy)
    return accVals

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot Section 

# Q2.1 In the first histogram, you should show the accuracy distribution when the algorithm is evaluated over training data.
plotTrainID3 = np.array(accuracyData('id3', 200, train=True))
# print(plotTrainID3)
print(f'The mean accuracy is {plotTrainID3.mean()}, and the std is {plotTrainID3.std()}')
plt.hist(plotTrainID3, density=1, bins=100, color='blue', alpha=0.5)
plt.title("Decision Tree using ID3 Algorithm On Training Data")
plt.ylabel("Accuracy Frequency")
plt.xlabel("Accuracy")
plt.axis([0.5, 1.1, 0, 120])
plt.show()

# Q2.2 In the second histogram, you should show the accuracy distribution when the algorithm is evaluated over testing data.
plotTestID3 = np.array(accuracyData('id3', 200, train=False))
# print(plotTestID3)
print(f'The mean accuracy is {plotTestID3.mean()}, and the std is {plotTestID3.std()}')
plt.hist(plotTestID3, density=1, bins=15, color='blue', alpha=0.5)
plt.title("Decision Tree using With ID3 Algorithm On Testing Data")
plt.ylabel("Accuracy Frequency")
plt.xlabel("Accuracy")
plt.axis([0.65, 1.1, 0, 30])
plt.show()

# QE.1 Repeat the experiments Q2.1 to Q2.4, but now use the Gini criterion for node splitting, instead of the Information Gain criterion.

plotTrainGini = np.array(accuracyData('cart', 200, train=True))
# print(plotTrainGini)
print(f'The mean accuracy is {plotTrainGini.mean()}, and the std is {plotTrainGini.std()}')
plt.hist(plotTrainGini, density=1, bins=100, color='blue', alpha=0.5)
plt.title("Decision Tree With Gini Algorithm On Training Data")
plt.ylabel("Accuracy Frequency")
plt.xlabel("Accuracy")
plt.axis([0.5, 1.1, 0, 120])
plt.show()

# QE.1 Repeat the experiments Q2.1 to Q2.4, but now use the Gini criterion for node splitting, instead of the Information Gain criterion.
plotTestGini = np.array(accuracyData('cart', 200, train=False))
# print(plotTestGini)
print(f'The mean accuracy is {plotTestGini.mean()}, and the std is {plotTestGini.std()}')
plt.hist(plotTestGini, density=1, bins=15, color='blue', alpha=0.5)
plt.title("Decision Tree using Gini Algorithm On Testing Data")
plt.ylabel("Accuracy Frequency")
plt.xlabel("Accuracy")
plt.axis([0.65, 1.1, 0, 30])
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

