import pandas as pd
import numpy as np
import random

def dataPreProcessing(featureArray):
    totalColumns=featureArray.shape[1]
    scaledArray=[1,2,3,4,5,6,7,8]
    for i in range(totalColumns): #Iterating through each column
        currentColumn=featureArray[:,i]
        mean= currentColumn.mean() #Mean value for current column
        standardDeviation=currentColumn.std() #Standard Deviation for current column
        j=((featureArray[:,i]-mean)/standardDeviation)
        scaledArray[i]=j#Scaling each element of the column
    return scaledArray

def dataSplit(features, target, testSize):
    totalRows=features.shape[0]-1
    testRows=np.round(totalRows*testSize)
    randomRowNum=np.random.randint(0, int(totalRows), int(testRows)) #Randomly generating row numbers
    testingFeatures=np.array([features[i] for i in randomRowNum]) #Creating test dataset features
    testingTarget=np.array([target[i] for i in randomRowNum]) #Creating the target array for test dataset
    testingTarget=testingTarget.flatten() #Turning the target array into a 1D array
    trainingFeatures=np.delete(features, randomRowNum, axis=0) #Creating training dataset by deleting the test rows from main dataset
    trainingTarget=np.delete(target, randomRowNum) #Creating the target array for training dataset
    trainingFeatures=dataPreProcessing(trainingFeatures)#Scaling the training features
    testingFeatures=dataPreProcessing(testingFeatures)#Scaling the testing features
    trainingFeatures=np.array(trainingFeatures)
    testingFeatures=np.array(testingFeatures)
    return trainingFeatures, trainingTarget, testingFeatures, testingTarget


premDataFeatures=pd.read_excel('premDataFeatures.xlsx')
premDataFeatures=np.array(premDataFeatures, dtype=int)
premDataFeatures=np.delete(premDataFeatures, 3798, axis=0) #deleting the empty row at the end of the array

premDataTargets=pd.read_excel('premDataTargets.xlsx')
premDataTargets=np.array(premDataTargets, dtype=str)

trainingFeatures, trainingTarget, testingFeatures, testingTarget = dataSplit(premDataFeatures, premDataTargets, 0.4)



def vectorConversion(target): #Target is 1D array of past game outcomes
    length=len(target)
    #initialising an ampty 2D array to store the target vectors in
    targetVector=np.empty([length, 3], dtype=int)
    counter=0
    for entry in target: #Mapping the target variables to vecetors
        if entry == 'A':
            vector = [1, 0, 0]
        elif entry == 'D':
            vector = [0, 1, 0]
        elif entry == 'H':
            vector =[0, 0, 1]
        targetVector[counter]=vector #Populating the targetVector array
        counter=counter+1
    return targetVector

vectorTrainingTarget=vectorConversion(trainingTarget)
vectorTestingTarget =vectorConversion(testingTarget)

#Linear Predictor Function
#Calculates logit scores for each possible outcome for each feature set
#The logit score correlates to the probability of each target variable being output for a given feature set
def linearPredictor(featureMatrix,weights,biases):
    logitScores=np.array([np.empty([3]) for i in range(featureMatrix.shape[1])]) #creating empty array for each feature set
    for i in range(featureMatrix.shape[1]): #iterating through each feature set
        #caculates the logit score for each feature set then flattens the logit vector
        logitScores[i]=(weights.dot(featureMatrix[:,i].reshape(-1,1)) + biases).reshape(-1)
    return logitScores

weights=np.empty([3,8], dtype=float)
for row in range(3):
    for column in range(8):
        weights[row, column]=random.uniform(1,2)

         
biases=np.random.rand(3,1)


trainingLogitscores=linearPredictor(trainingFeatures, weights, biases)

def softmaxFunc(logitMatrix):
    #creating 1x3 empty array for each feature set
    probabilities=np.array([np.empty([3]) for i in range(logitMatrix.shape[0])])
    for i in range(logitMatrix.shape[0]):
        exponential=np.empty(3, dtype=float)
        totalExponents=0
        for counter in range(3):            
        #exponentiating each element of the logit matrix
            exponential[counter]=np.exp(logitMatrix[i, counter])
        #Adding up all the values of the exponentiated matrtix
            totalExponents=totalExponents+exponential[counter]
        #Converting logit scores to probability values
        probabilities[i]=exponential/totalExponents
        if totalExponents==0:
            print(totalExponents)
    return probabilities

trainingProbabilities=softmaxFunc(trainingLogitscores)
'''
for row in trainingProbabilities:
    for k in row:
        if k<0.0001:
            print(k)'''

#Calculates the cross entropy loss for the predictions and target variables
def crossEntropyLoss(probabilities, target):
    sampleNumber=probabilities.shape[0]
    loss=0
    totalLoss=0
    for counter in range(sampleNumber):
        loss= np.dot(target[counter], probabilities[counter])
        loss=np.log(loss)*-1
        totalLoss=totalLoss+loss
    informationLoss=totalLoss/sampleNumber
    return informationLoss

informationLoss=crossEntropyLoss(trainingProbabilities, vectorTrainingTarget)
print(format(informationLoss, 'f'))
