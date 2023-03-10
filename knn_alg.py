import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from operator import itemgetter

# load the iris data file 
data = open('iris.csv')
dataReader = csv.reader(data)

# irisData: list to hold the iris data
irisData = []
for row in dataReader:
    irisData.append([float(val) if i < 4 else val for i, val in enumerate(row)])

# split the data into training and testing sets
# the training set will contain 80% of the data, the testing set will contain 20% of the data
# data: file data; ranNum: ensures reproducibilty; trainKnn: resulting data set for training; testKnn: resulting data set for testing 
def split(data, ranNum):
    trainKnn, testKnn = sklearn.model_selection.train_test_split(data, train_size = 0.8, test_size = 0.2, shuffle = True, random_state = ranNum)
    return trainKnn, testKnn

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Normalization Section

# normalize the training and testing datasets using the maximum and minimum values
# trainData: training dataset; testData: the testing dataset
def normalizedDataset(trainData, testData):

    # normalize a column of data
    # column: column to be normalized; resArr: resulting normalized array
    def normalizeCol(datColumn):
        minVal = min(datColumn)
        maxVal = max(datColumn)
        # iterate through the list and normalize each value
        resArr = [(val - minVal) / (maxVal - minVal) for val in datColumn]
        return resArr, minVal, maxVal
    
    trainKnnNorm, testKnnNorm = [], []
    colInx = 0
    # check if the current column is a feature column (i.e. columns 0-3)
    while colInx < 4:
        # normalize the testing data for the current column using the min and max vals from the training data
        trainArr, trainMin, trainMax = normalizeCol(trainData[colInx])
        # loop through each column in the training dataset
        testArr = [(val - trainMin) / (trainMax - trainMin) for val in testData[colInx]] 
        trainKnnNorm.append(trainArr)
        testKnnNorm.append(testArr)
        colInx += 1
    # append the class labels to the normalized training and testing datasets
    trainKnnNorm.append(trainData[4])
    testKnnNorm.append(testData[4])
    
    return trainKnnNorm, testKnnNorm

# normalize the training and testing datasets
# trainData: the training dataset; testData: the testing dataset
def getNormData(trainData, testData):
    # transpose the training and testing datasets
    trnspTrain = [list(row) for row in zip(*trainData)]
    trnspTest = [list(row) for row in zip(*testData)]
    # normalize the transposed data
    normTrnspTrain, normTrnspTest = normalizedDataset(trnspTrain, trnspTest)
    # transpose the normalized training and testing datasets back to their original format
    resNormTrain, resNormTest = [list(row) for row in zip(*normTrnspTrain)], [list(row) for row in zip(*normTrnspTest)]
    return resNormTrain, resNormTest

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# KNN Section

# compute the euclidean distance between a test data point and all the training data points
# testDataPt: test data point; trainDataPt: a list of training data points
def distanceCalc(testDataPt, trainDataPts):

    # Euclidean distance
    # Compute the elcudidean distance between two arrays
    def euclideanDist(arr1, arr2):
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        dist = np.linalg.norm(arr1 - arr2)
        return dist

    # feature values of the test data point
    point = testDataPt[:-1]
    # calculate the Euclidean distance between the test data point and all the training data points
    distArr = [(euclideanDist(point, elem[:-1]), elem[-1]) for elem in trainDataPts]
    # return distances in ascending order
    return sorted(distArr, key=itemgetter(0))

# k-nearest neighbours algorithm
# k: number of nearest neighbours being considered; trainData: training data set; testData: testing data set
def knnAlgor(k, trainData, testData):
    predictedLables = []
    # extract the true labels for test data and store them in a separate list
    correctLables = [datColumn[-1] for datColumn in testData]
    for datpt in testData:
        distArr = distanceCalc(datpt, trainData)
        # extract the labels of the k nearest neighbours
        labelsArr = [datColumn[1] for datColumn in distArr[:k]]
        # predict the label of the test data point based on the majority label of k nearest neighbours
        predictedLables.append(max(set(labelsArr), key=labelsArr.count))
    return predictedLables, correctLables

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Accuracy Section

# performs k-fold cross-validation to test the accuracy of the knn() function using a given value of k on the dataset.
# k = number of neighbours; randNum: number used to split dataset; data: iris data; train: bool to indicate the type of dataset
def accuracyData(k, randNum, data, train=True):

    # accuracy of the predicted labels generated by the knn function above
    # predicted: predicted labels for test data points; correct: correct labels for test data points
    def accuracy(predicted, correct):
        total = len(predicted)
        # count the number of correct predictions
        correctLables = sum(1 for i in range(total) if predicted[i] == correct[i])
        return correctLables / total
    
    # split the dataset
    trainknn, testknn = split(data, randNum)
    #normalize the dataset (comment out for Q1.6 to test on unnormalized data)
    normedtrain, normedtest = getNormData(trainknn, testknn)
    # normedtrain, normedtest = trainknn, testknn
    # predict labels for the training set using the knn function with k nearest neighbors
    predictedLables, correctLables = knnAlgor(k, normedtrain, normedtrain) if train == True else knnAlgor(k, normedtrain, normedtest)
    calcAccuracy = accuracy(predictedLables, correctLables)
    return calcAccuracy

# performs statistical analysis on the accuracy of the knn() function using a range of k values on a dataset
#  data: iris data set; train: bool to indicate the type of dataset
def statAnalysisAcc(data, train=True):
    # list to store the results for each k value
    caclKVals = []
    # varying k from 1 to 51, using only odd numbers
    for k in range(1, 52, 2):
        # empty list to store the accuracy results for each random seed
        calcAccVals = []
        # for each value of k, run the process 20 times.
        for randNum in range(12100, 12200, 5):
            # calculate the accuracy of the knn function using k-fold cross-validation with a given k value and random seed value
            if train:
                calcAccVals.append(accuracyData(k, randNum, data, train=True))
            else:
                calcAccVals.append(accuracyData(k, randNum, data, train=False))
        # append the accuracy results for the current k value to the result list
        caclKVals.append(calcAccVals)

    return np.array(caclKVals)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot Section 

# NumPy array k that contains odd integer values from 1 to 51 
k = np.arange(1,52,2)

#RESULTS
#train
kValsTrain = statAnalysisAcc(irisData, train=True) 
#test
kValsTest = statAnalysisAcc(irisData, train=False)

#Accuracy for training dataset
accValsTrain = kValsTrain.mean(axis=1)
# print("KNN accuracy for training set")
# print(accValsTrain)

#Accuracy for testing dataset
accValsTest = kValsTest.mean(axis=1)
# print("KNN accuracy for testing set")
# print(accValsTest)

# Std deviation for training
stdValsTrain = kValsTrain.std(axis = 1)
# print("KNN standard deviation for training")
# print(stdValsTrain)

stdValsTest = kValsTest.std(axis = 1)
# print("KNN standard deviation for training")
# print(stdValsTest)

# Q1.1
plt.errorbar(k, accValsTrain, yerr = stdValsTrain, fmt = "-o", color = 'green', alpha = 0.5)
plt.title("KNN On Training Data")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.show()

# Q1.2
plt.errorbar(k, accValsTest, yerr = stdValsTest, fmt = "-o", color = 'red', alpha = 0.5)
plt.title("KNN using Testing Data")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# RESULTS FOR ANALYSIS

# WITH NORMALIZATION:

# KNN accuracy for training set
# [1.         0.96791667 0.9675     0.97458333 0.97208333 0.9725
#  0.97375    0.9725     0.97083333 0.96541667 0.96416667 0.96125
#  0.96083333 0.96083333 0.96125    0.95708333 0.95458333 0.94625
#  0.94208333 0.93541667 0.93       0.9225     0.915      0.91
#  0.9075     0.90583333]

# KNN accuracy for testing set
# [0.94333333 0.94166667 0.955      0.94666667 0.94833333 0.95166667
#  0.955      0.95666667 0.95666667 0.95166667 0.94833333 0.95333333
#  0.94666667 0.945      0.94333333 0.94166667 0.94166667 0.94
#  0.93666667 0.93166667 0.91833333 0.91166667 0.90833333 0.9
#  0.89833333 0.89      ]

# KNN standard deviation for training
# [0.         0.00605243 0.00829156 0.01036655 0.00660124 0.0083749
#  0.00844714 0.01057381 0.00931695 0.01029934 0.00794949 0.00844714
#  0.00650854 0.00595119 0.00844714 0.00844714 0.00767165 0.01002601
#  0.00892679 0.01114145 0.01452966 0.01493039 0.0163724  0.01982563
#  0.02122957 0.01808545]
 
# KNN standard deviation for training
# [0.03511885 0.02327373 0.02420973 0.0244949  0.02682246 0.02466441
#  0.02420973 0.01855921 0.02603417 0.03244654 0.03068659 0.0305505
#  0.03231787 0.03032234 0.03       0.03312434 0.03476109 0.03741657
#  0.04068852 0.04530759 0.04884784 0.05403188 0.05361903 0.0505525
#  0.05320297 0.05487359]

# WITHOUT NORMALIZATION:

# KNN accuracy for training set
# [1.         0.96916667 0.975      0.97791667 0.97875    0.97875
#  0.97833333 0.97625    0.97458333 0.97333333 0.96875    0.96458333
#  0.96083333 0.95583333 0.95416667 0.95291667 0.95458333 0.9525
#  0.94958333 0.94416667 0.94291667 0.93791667 0.93375    0.92708333
#  0.9225     0.92291667]
# KNN accuracy for testing set
# [0.94833333 0.95333333 0.95333333 0.955      0.96166667 0.96333333
#  0.96       0.96333333 0.96666667 0.95166667 0.95166667 0.94833333
#  0.95       0.94666667 0.94666667 0.94833333 0.94333333 0.93666667
#  0.93333333 0.93       0.93       0.92833333 0.925      0.92333333
#  0.91666667 0.91166667]
# KNN standard deviation for training
# [0.         0.01024017 0.00833333 0.0071078  0.00670562 0.00767165
#  0.00964653 0.00660124 0.00967349 0.01073675 0.00981602 0.01232517
#  0.01293681 0.01181454 0.00931695 0.00960143 0.00852895 0.0115169
#  0.01192424 0.0115169  0.01243734 0.01501735 0.01221196 0.01464274
#  0.01538849 0.01823363]
# KNN standard deviation for training
# [0.02682246 0.02211083 0.02666667 0.03032234 0.03032234 0.02081666
#  0.02708013 0.0314466  0.02981424 0.03244654 0.02881936 0.03068659
#  0.03248931 0.03231787 0.02867442 0.03244654 0.03349959 0.03785939
#  0.03944053 0.04203173 0.03636237 0.03840573 0.04579544 0.04725816
#  0.04533824 0.04627814]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
