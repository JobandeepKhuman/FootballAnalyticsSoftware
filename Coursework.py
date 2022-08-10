#Premier League prediction Model
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data preprocessing
premData1920=pd.read_csv('PremData1920.csv')
#Seperating features and target variable
features=premData1920[['FTHG','FTAG','HS','AS','HST','AST','HC','AC']]
target=premData1920['FTR']
#Converting target variable and features to numpy arrays
target=np.array(target)
features=np.array(features)#Syntax different because using old version of pandas?
#Data Standardisation Function
#Function will iterate through each attribute of the matrix scaling them individually
def standardisationFunc(featureArray):
    totalColumns=featureArray.shape[1]
    for i in range(totalColumns): #Iterating through each column
        currentColumn=featureArray[:,i]
        mean=currentColumn.mean() #Mean value for current column
        standardDeviation=featureArray.std() #Standard Deviation for each column
        featureArray[:,i]=(featureArray[:,i]-mean)/standardDeviation #Scaling each element of the column
    return featureArray

scaledFeatures=standardisationFunc(features)
#Creating array of random weights for linear predictor function
weights=np.random.rand(3,8)
#Creating array of random biases for linear prediction function
biases=np.random.rand(3,1)

#Linear Predictor Function
#Calculates logit scores for each possible outcome for each feature set
#The logit score correlates to the probability of each target variable being output for a given feature set
def linearPredictor(featureMatrix,weights,biases):
    logitScores=np.array([np.empty([3]) for i in range(featureMatrix.shape[0])]) #creating empty array for each feature set
    for i in range(featureMatrix.shape[0]): #iterating through each feature set
        logitScores[i]=(weights.dot(featureMatrix[i].reshape(-1,1)) + biases).reshape(-1) #caculates the logit score for each feature set then flattens the logit vector
    return logitScores

#Softmax Function
#Converts logit scores for each possible outcome to probability values
def softmaxFunc(logitMatrix):
    #creating empty array for each feature set
    probabilities=np.array([np.empty([3]) for i in range(logitMatrix.shape[0])])
    for i in range(logitMatrix.shape[0]):
        #exponentiating each element of the logit matrix
        exponential=np.exp(logitMatrix[i])
        #Adding up all the values of the exponentiated matrtix
        totalExponentials=np.sum(exponential)
        #Converting logit scores to probability values
        probabilities[i]=exponential/totalExponentials
    return probabilities

#Multinomial Logistic Regression Function
#Combining the softmax and linear predicor functions to perform multinomial logistic regression
def multinomialLogisticRegression(features, weights, biases):
    logitScores=linearPredictor(features, weights, biases)
    probabilities=softmaxFunc(logitScores)
    #returns the outcomme with the highest probability
    predictions=np.array([np.argmax(i) for i in probabilities])
    return probabilities, predictions

probabilities, predictions = multinomialLogisticRegression(features,weights,biases)


#Accuracy Function
#Calculates the accuracy of the model
def accuracyCalc(predictions, target):
    correctPredictions=0
    for i in range(len(predictions)):
        if predictions[i]==target[i]:
            correctPredictions += 1
    accuracy=(correctPredictions/len(predictions))*100
    return accuracy

#accuracy=accuracyCalc(predictions,target)
#print(accuracy)

#Function to split the data into a training set and a testing set
data=premData1920[['FTHG','FTAG','HS','AS','HST','AST','HC','AC','FTR']]
print(data)
def dataSplit(data, testSize):
    data=np.array(data) #Converting data to numpy array
    print(data)
    totalRows=data.shape[0]
    testRows=np.round(totalRows*testSize)
    randomRowNum=np.random.randint(0, int(totalRows), int(testRows)) #Randomly generating row numbers
    testData=np.array([data[i] for i in randomRowNum]) #Creating test dataset
    data=np.delete(data, randomRowNum, axis=0) #Creating training dataset by deleting the test rows from main dataset
    trainingFeatures=data[:,:-1] #All but lastColum
    trainingTarget=data[:,-1] #LastColumun
    testingFeatures=testData[:,:-1] 
    testingTarget=testData[:,-1]
    
    return trainingFeatures, trainingTarget, testingFeatures, testingTarget

#Running the test train split on dataset
trainingFeatures, trainingTarget, testingFeatures, testingTarget=dataSplit(premData1920,0.2)
#Scaling the testing and training features
scaledTrainingFeatures=standardisationFunc(trainingFeatures)
scaledTestingFeatures=standardisationFunc(testingFeatures)

#Cost Function
#Calculates the cross entropy loss for the predictions and target variables
def crossEntropyLoss(probabilities, target):
    sampleNumber=probabilities.shape[0]
    loss=0
    for sample, i in zip(probabilities, target):
        loss += -np.log(sample[i])
    loss /= sampleNumber
    return loss

#GradientDescentFunction
#Applying stochastic gradient descent to the cost function
def stochaisticGradientDescent(learningRate, iterations, target, features, weights, biases):
    target=target.astype(int)
    lossList=np.array([]) #Creating an empty array   
    for i in range(iterations):
        #Calculating the probabilities for each possible outcome
        print(scaledTrainingFeatures.shape)
        print(weights.shape)
        probabilities,_=multinomialLogisticRegression(scaledTrainingFeatures, weights, biases)
        #Calculating the cross entropy loss for target and predictions
        loss=crossEntropyLoss(probabilities, target)
        #Adding the loss value for each iteration to the loss list, idk why
        lossList=np.append(lossList,loss)
        #Subtracting 1 from the scores of the correct outcomes
        probabilities[np.arange(features.shape[0]),target] -= 1
        #gradient of loss with respect to the weights
        gradientWeight=probabilities.T.dot(features)
        #gradient of loss with respect to the biases
        gradientBiases=np.sum(probabilities, axis=0).reshape(-1,1)
        
        #updating the weights and biases
        weights -= (learningRate*gradientWeight)
        biases -= (learningRate*gradientBiases)
    return weights, biases, lossList

updatedWeights, updatedBiases, lossList = stochaisticGradientDescent(0.1, 200, trainingTarget, scaledTrainingFeatures, weights, biases)
        
    
