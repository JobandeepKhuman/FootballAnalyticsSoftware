import pandas as pd
import numpy as np
import random
np.set_printoptions(precision=8, suppress=True)

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


#Linear Predictor Function
#Calculates logit scores for each possible outcome for each feature set
#The logit score correlates to the probability of each target variable being output for a given feature set
def linearPredictor(featureMatrix,weights,biases):
    logitScores=np.array([np.empty([3]) for i in range(featureMatrix.shape[1])]) #creating empty array for each feature set
    for i in range(featureMatrix.shape[1]): #iterating through each feature set
        #caculates the logit score for each feature set then flattens the logit vector
        logitScores[i]=(weights.dot(featureMatrix[:,i].reshape(-1,1)) + biases).reshape(-1)
    return logitScores

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


def multinomialLogisticRegression(features, weights, biases):
    matchNumber=features.shape[1]
    predictions=np.empty(matchNumber, dtype=str)
    logitscores=linearPredictor(features, weights, biases)
    probabilities=softmaxFunc(logitscores)
    for counter in range(matchNumber):
        if probabilities[counter,0] > probabilities[counter,1]:
            biggest=probabilities[counter,0]
            outcome='A'
        else:
            biggest=probabilities[counter, 1]
            outcome='D'
        if probabilities[counter,2] > biggest:
            biggest=probabilities[counter,2]
            outcome='H'
        predictions[counter]=outcome
    return predictions

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

#GradientDescentFunction
#Applying stochastic gradient descent to the cost function
def stochaisticGradientDescent(learningRate, iterations, target, features, weights, biases):
    for i in range(iterations):
        #Calculating the probabilities for each possible outcome
        logitscores=linearPredictor(features, weights, biases)
        probabilities=softmaxFunc(logitscores)
        #Calculating the cross entropy loss for target and predictions
        loss=crossEntropyLoss(probabilities, target)
        #Subtracting 1 from the scores of the correct outcomes
        for row in range(target.shape[0]-1):
            for column in range(target.shape[1]):
                if target[row, column] == 1:
                    probabilities[row, column] = probabilities[row, column]-1
        #gradient of loss with respect to the weights
        gradientWeight=np.matmul(probabilities.T, (features.T))
        #gradient of loss with respect to the biases
        gradientBiases=np.sum(probabilities, axis=0).reshape(-1,1)
        #updating the weights and biases
        weights = weights-(learningRate*gradientWeight)
        biases = biases-(learningRate*gradientBiases)
    return weights, biases

def accuracy(target, features, weights, biases):
    predictions=multinomialLogisticRegression(features, weights, biases)
    success=0
    for counter in range(len(predictions)):
        if predictions[counter] == target[counter]:
            success=success+1
    accuracy=(success/len(predictions))*100
    return accuracy


#Bringing all the modules together
premDataFeatures=pd.read_excel('premDataFeatures.xlsx')#Reading all the features from the excel file
premDataFeatures=np.array(premDataFeatures, dtype=int)#Assinging the features arrray to hold integers
premDataFeatures=np.delete(premDataFeatures, 3798, axis=0) #deleting the empty row at the end of the array

premDataTargets=pd.read_excel('premDataTargets.xlsx')#Reading the outcomes from the target excel file
premDataTargets=np.array(premDataTargets, dtype=str)#Assinging the target array to hold strings

#Splitting and pre-processing the data
trainingFeatures, trainingTarget, testingFeatures, testingTarget = dataSplit(premDataFeatures, premDataTargets, 0.4)
#Converting the target arrays from strings to vectors
vectorTrainingTarget=vectorConversion(trainingTarget)
vectorTestingTarget =vectorConversion(testingTarget)

#Initialising the weights array
weights=np.empty([3,8], dtype=float)
for row in range(3):
    for column in range(8):
        weights[row, column]=random.uniform(1,2)        
biases=np.random.rand(3,1)#initialising the biases array

#Applying stochaistic gradient descent to the machine learning model in order to calculate weights and biases the predict more accuratley
updatedWeights, updatedBiases=stochaisticGradientDescent(0.001, 100, vectorTrainingTarget, trainingFeatures, weights, biases)

def clubnameValidation(clubname):
    #Defining what names are valid once everythin is lowercase and spaces are removed
    validNames = ['arsenal', 'astonvilla', 'brighton', 'burnley', 'chelsea', 'crystalpalace', 'everton', 'fulham', 'leeds', 'leicester', 'liverpool', 'manchestercity', 'manchesterunited', 'newcastle', 'sheffieldUnited', 'southampton', 'tottenham', 'westbrom', 'westham', 'wolves']
    valid = False
    #standardising the input by making it lowercase and removing spaces
    clubname = clubname.lower()
    clubname = clubname.replace(" ","")
    #Checking if the standardised input is in the validNames array
    for counter in range(20):
        if clubname == validNames[counter]:
            valid = True
    if valid == False:
        #print("Please enter a valid clubname")
        n=1
    else:
        #print("Valid Clubname")
        n=2
    return clubname


#Function to calculate the average Statistics of a given team
def averageStats(clubname):
    seasonData=pd.read_excel('2021Data.xlsx')#Reading all the features from the excel file
    seasonData=np.array(seasonData, dtype=str)#Assinging the features arrray to hold integers
    #validating the clubname
    clubname=clubnameValidation(clubname)
    #number of games that the statistics will be averaged over
    k=8
    counter=0
    #will iterate through past games, chacking if the given club was the home team
    recentGame=184
    #initialising the array that will hold the clubs average statistics
    previousGamesData=np.empty([8])
    #Looping through until k home games have been iterated through
    while counter<k:
        homeTeam=seasonData[recentGame,0]
        homeTeam = homeTeam.lower()
        homeTeam = homeTeam.replace(" ","")
        if homeTeam == clubname:
            #Iterating through and updating previousGames array with the relevant statistics
            for i in range(8):
                previousGamesData[i]=previousGamesData[i]+int(seasonData[recentGame,i+2])
                #incrementing the counter
            counter=counter+1
        #Incrementing recentGame variable
        recentGame=recentGame-1
    averageData=previousGamesData / k
    return averageData


def seasonSimulation():
    #intitialising the league table with every team name and a points total of 0
    leagueTable = [['arsenal', 0], ['astonvilla', 0], ['brighton', 0], ['burnley', 0], ['chelsea', 0], ['crystalpalace', 0], ['everton', 0], ['fulham', 26], ['leeds', 55], ['leicester', 0], ['liverpool', 0], ['manchestercity', 0], ['manchesterunited', 0], ['newcastle', 0], ['sheffieldunited', 0], ['southampton', 0], ['tottenham', 0], ['westbrom', 20], ['westham', 0], ['wolves', 0]]
    #initialising an array of predicitions for the season
    predictions=np.empty([380], dtype=str)
    #reading the fixtures file
    pastSeasonData=pd.read_excel('1920Data.xlsx')
    pastSeasonData=np.array(pastSeasonData, dtype=str)
    #Iterating through each fixture combination and predicting the outcome
    for match in range(380):
        #Reading the satatistics for each game played last season
        features=pastSeasonData[match]
        features=np.delete(features,[0,1])
        features=features.reshape(8,1)
        features=features.astype('int')
        holder = multinomialLogisticRegression(features, updatedWeights, updatedBiases)
        predictions[match]=holder[0]
    #Iterating through each outcome and adjusting the points total accordingly
    for counter in range(380):
        homeTeam=pastSeasonData[counter,0]
        awayTeam=pastSeasonData[counter,1]
        awayTeam = awayTeam.lower().replace(" ","")
        homeTeam = homeTeam.lower().replace(" ","")
        if predictions[counter]=='H':
            for team in range(20):
                if leagueTable[team][0]==homeTeam:
                    leagueTable[team][1]=leagueTable[team][1]+3
        elif predictions[counter]=='A':
            for team in range(20):
                if leagueTable[team][0]==awayTeam:
                    leagueTable[team][1]=leagueTable[team][1]+3
        elif predictions[counter]=='D':
            for team in range(20):
                if leagueTable[team][0]==homeTeam:
                    leagueTable[team][1]=leagueTable[team][1]+1
                if leagueTable[team][0]==awayTeam:
                    leagueTable[team][1]=leagueTable[team][1]+1
    #sorting the league table based on points
    leagueTable=sorted(leagueTable, key=lambda x: x[1])
    return leagueTable

results=seasonSimulation()
print(results)