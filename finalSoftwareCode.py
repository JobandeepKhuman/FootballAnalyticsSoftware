from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
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
    print(matchNumber)
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
updatedWeights, updatedBiases=stochaisticGradientDescent(0.001, 10, vectorTrainingTarget, trainingFeatures, weights, biases)

def meanVariance():
    premDataFeatures=pd.read_excel('premDataFeatures.xlsx')#Reading all the features from the excel file
    premDataFeatures=np.array(premDataFeatures, dtype=int)#Assinging the features arrray to hold integers
    premDataFeatures=np.delete(premDataFeatures, 3798, axis=0) #deleting the empty row at the end of the array
    totalGoals=0
    for counter in range(3798):
        #Adding the number of home and away goals of the current match to the totalGoals variable
        totalGoals=totalGoals+premDataFeatures[counter, 0]+premDataFeatures[counter, 1]
    #Averaging the total number of goals
    meanGoals=totalGoals/3798 
    variance=0
    #Applying the variance formula
    for counter in range(3798):
        matchGoals=premDataFeatures[counter,0] + premDataFeatures[counter,1]
        #Adding the squared difference between the total goals and meanGoals to variance
        variance = variance + (matchGoals-meanGoals)**2
    #Averaging the variane
    variance=variance/3798
    meanGoals=round(meanGoals,2)
    variance=round(variance,2)
    return meanGoals, variance

def clubnameValidation(clubname):
    #Defining what names are valid once everythin is lowercase and spaces are removed
    validNames = ['arsenal', 'astonvilla', 'brighton', 'burnley', 'chelsea', 'crystalpalace', 'everton', 'fulham', 'leeds', 'leicester', 'liverpool', 'manchestercity', 'manchesterunited', 'newcastleunited', 'sheffieldunited', 'southampton', 'tottenham', 'westbrom', 'westham', 'wolverhamptonwanderers']
    valid = False
    #standardising the input by making it lowercase and removing spaces
    clubname = clubname.lower()
    clubname = clubname.replace(" ","")
    #Checking if the standardised input is in the validNames array
    for counter in range(20):
        if clubname == validNames[counter]:
            valid = True
    if valid == False:
        #clubname="invalid"
        print(clubname)
    else:
        #print("Valid Clubname")
        j=1
    return clubname


#Function to calculate the average Statistics of a given team
def averageStats(clubname):
    seasonData=pd.read_excel('2021Data.xlsx')#Reading all the features from the excel file
    seasonData=np.array(seasonData, dtype=str)#Assinging the features arrray to hold integers
    #validating the clubname
    clubname=clubnameValidation(clubname)
    print(clubname)
    #number of games that the statistics will be averaged over
    k=8
    counter=0
    #will iterate through past games, chacking if the given club was the home team
    recentGame=0
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
        recentGame=recentGame+1
    averageData=previousGamesData / k
    return averageData

seasonData=pd.read_excel('2021Data.xlsx')#Reading all the features from the excel file
seasonData=np.array(seasonData, dtype=str)#Assinging the features arrray to hold integers


def thisWeeksPredictions(gameweek, updatedWeights, updatedBiases):
    #reading the fixtures list for the current season
    fixtures=pd.read_excel('fixtures.xlsx')
    fixtures=np.array(fixtures, dtype=str)
    #Array to hold the average stats of each home team playing this week
    weeksData=np.empty([10,8])
    #start will hold the row number of the first game of the current gameweek
    start=(gameweek-1)*10
    #Iterating through each match in the current gameweek and storing the home team's average stats
    for counter in range(10):
        homeTeam=fixtures[start+counter, 1]
        homeData=averageStats(homeTeam)
        #populating the weeksData array with each home team's average stats
        for column in range(8):
            weeksData[counter, column] = homeData[column]
    weeksData=weeksData.transpose()
    weeksLogitscores=linearPredictor(weeksData, updatedWeights, updatedBiases)   
    weeksProbabilities=softmaxFunc(weeksLogitscores)
    return weeksProbabilities

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

def trueAverageStats(clubname):
    seasonData=pd.read_excel('2021Data.xlsx')#Reading all the features from the excel file
    seasonData=np.array(seasonData, dtype=str)#Assinging the features arrray to hold integers
    #validating the clubname
    clubname=clubnameValidation(clubname)
    if clubname == "invalid":
        return "invalid"
    #number of games that the statistics will be averaged over
    k=8
    counter=0
    #will iterate through past games, chacking if the given club participated in the match
    recentGame=184
    #initialising the array that will hold the clubs average statistics
    previousGamesData=np.empty([8])
    #Looping through until k home games have been iterated through
    while counter<k:
        #Declaring and standardising the home and away team for the current match
        homeTeam=seasonData[recentGame,0]
        homeTeam = homeTeam.lower()
        homeTeam = homeTeam.replace(" ","")
        awayTeam=seasonData[recentGame,0]
        awayTeam = awayTeam.lower()
        awayTeam = awayTeam.replace(" ","")
        if homeTeam == clubname:
            #Iterating through and updating previousGames array with the relevant statistics
            for i in range(8):
                previousGamesData[i]=previousGamesData[i]+int(seasonData[recentGame,i+2])
                #incrementing the counter
            counter=counter+1
        elif awayTeam == clubname:
            #Putting the awayStats when club=awayTeam in the same column as the homeStats when club=homeTeam and vice versa
            for i in range(8):
                if (i%2)==0:
                    previousGamesData[i]=previousGamesData[i]+int(seasonData[recentGame,i+3])
                elif (i%2)!=0:
                    previousGamesData[i]=previousGamesData[i]+int(seasonData[recentGame,i+1])
            counter+counter+1
        #Incrementing recentGame variable
        recentGame=recentGame-1
    averageData=previousGamesData / k
    return averageData

def seasonComparison():
    premDataFeatures=pd.read_excel('premDataFeatures.xlsx')#Reading all the features from the excel file
    premDataFeatures=np.array(premDataFeatures, dtype=int)#Assinging the features arrray to hold integers
    premDataFeatures=np.delete(premDataFeatures, 3798, axis=0) #deleting the empty row at the end of the array
    seasonData=np.zeros([10,9], dtype=float) #Initialising the array that will hold the data about every season
    #Creating an array of names that will be included in seasonData
    seasonNames=["1920", "1819", "1718", "1617", "1516", "1415", "1314", "1213", "1112", "1011"]
    #Iterating through the past 10 years
    for year in range(10):
        seasonData[year,0]=seasonNames[year]
        #Iterating through each match played that year
        for match in range(380):
            matchNumber=(year*match)+match #The total match number out or 3800, corresponds to row in premDataFeatures array
        #Iterating through each feature/statistic of that match
            for stat in range(8):
                seasonData[year,stat+1]=seasonData[year,stat+1]+premDataFeatures[matchNumber,stat]
    for row in range(10):
        for column in range(8):
            seasonData[row, column+1]=seasonData[row, column+1]/380
    return seasonData

window=tk.Tk()

def predictionsDisplay():
    #reading the fixtures list for the current season
    fixtures=pd.read_excel('fixtures.xlsx')
    fixtures=np.array(fixtures, dtype=str)
    #reading the gameweek user input
    gameweek=int(matchweekInput.get())
    if 0<gameweek<39:
        #Array to hold the the teams invloved in each match for the current match week
        weeksData=[["",""],["",""],["",""],["",""],["",""],["",""],["",""],["",""],["",""],["",""]]
        #start will hold the row number of the first game of the current gameweek
        start=(gameweek-1)*10
        #Iterating through each match in the current gameweek and storing the home team's average stats
        for counter in range(10):
            homeTeam=fixtures[start+counter][1]
            awayTeam=fixtures[start+counter][2]
            weeksData[counter][0]=homeTeam
            weeksData[counter][1]=awayTeam
        #Displaying the team names and badges
        global badge1
        global badge2
        global badge3
        global badge4
        global badge5
        global badge6
        global badge7
        global badge8
        global badge9
        global badge10
        global badge11
        global badge12
        global badge13
        global badge14
        global badge15
        global badge16
        global badge17
        global badge18
        global badge19
        global badge20
        matchProbabilities=thisWeeksPredictions(gameweek, updatedWeights, updatedBiases)
        matchPercentages=[["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],["","",""],["","",""]]
        for row in range(10):
            for column in range(3):
                element=matchProbabilities[row,column]
                element=float(element)
                percentage=round((element*100),2)
                percentage=str(percentage)+"%"
                if column==0:
                    output='A: '+percentage
                elif column==1:
                    output='D: '+percentage
                elif column==2:
                    output='H: '+percentage
                matchPercentages[row][column]=output
        for counter in range(10):
            imageShift=counter*50
            #Displaying Team names for matches
            match1 = tk.Label(window, text=weeksData[counter][0]+" VS "+weeksData[counter][1]+" "+matchPercentages[counter][2]+" "+matchPercentages[counter][1]+" "+matchPercentages[counter][0])
            match1.config(font=('Arial', 10))
            canvas.create_window(200, 130+imageShift, window=match1)
        #Displaying the badge for home team for match 1
        homeTeam=(weeksData[0][0]).lower()
        badge1 = Image.open(homeTeam+" badge.png")
        badge1 = badge1.resize((20, 20), Image.ANTIALIAS)
        badge1 = ImageTk.PhotoImage(badge1)
        canvas.create_image(100,100, anchor=NW, image=badge1)
        #Displaying away team badge for match 2
        awayTeam=(weeksData[0][1]).lower()
        badge2 = Image.open(awayTeam+" badge.png")
        badge2 = badge2.resize((20, 20), Image.ANTIALIAS)
        badge2 = ImageTk.PhotoImage(badge2)
        canvas.create_image(200,100, anchor=NW, image=badge2)
        
        #Displaying the badge for home team for match 2
        homeTeam=(weeksData[1][0]).lower()
        badge3 = Image.open(homeTeam+" badge.png")
        badge3 = badge3.resize((20, 20), Image.ANTIALIAS)
        badge3 = ImageTk.PhotoImage(badge3)
        canvas.create_image(100,150, anchor=NW, image=badge3)
        #Displaying away team badge for match 2
        awayTeam=(weeksData[1][1]).lower()
        badge4 = Image.open(awayTeam+" badge.png")
        badge4 = badge4.resize((20, 20), Image.ANTIALIAS)
        badge4 = ImageTk.PhotoImage(badge4)
        canvas.create_image(200,150, anchor=NW, image=badge4)
        
        #Displaying the badge for home team for match 3
        homeTeam=(weeksData[2][0]).lower()
        badge5 = Image.open(homeTeam+" badge.png")
        badge5 = badge5.resize((20, 20), Image.ANTIALIAS)
        badge5 = ImageTk.PhotoImage(badge5)
        canvas.create_image(100,200, anchor=NW, image=badge5)
        #Displaying away team badge for match 3
        awayTeam=(weeksData[2][1]).lower()
        badge6 = Image.open(awayTeam+" badge.png")
        badge6 = badge6.resize((20, 20), Image.ANTIALIAS)
        badge6 = ImageTk.PhotoImage(badge6)
        canvas.create_image(200,200, anchor=NW, image=badge6)    
        
        #Displaying the badge for home team for match 4
        homeTeam=(weeksData[3][0]).lower()
        badge7 = Image.open(homeTeam+" badge.png")
        badge7 = badge7.resize((20, 20), Image.ANTIALIAS)
        badge7 = ImageTk.PhotoImage(badge7)
        canvas.create_image(100,250, anchor=NW, image=badge7)
        #Displaying away team badge for match 3
        awayTeam=(weeksData[3][1]).lower()
        badge8 = Image.open(awayTeam+" badge.png")
        badge8 = badge8.resize((20, 20), Image.ANTIALIAS)
        badge8 = ImageTk.PhotoImage(badge8)
        canvas.create_image(200,250, anchor=NW, image=badge8)

        #Displaying the badge for home team for match 5
        homeTeam=(weeksData[4][0]).lower()
        badge9 = Image.open(homeTeam+" badge.png")
        badge9 = badge9.resize((20, 20), Image.ANTIALIAS)
        badge9 = ImageTk.PhotoImage(badge9)
        canvas.create_image(100,300, anchor=NW, image=badge9)
        #Displaying away team badge for match 5
        awayTeam=(weeksData[4][1]).lower()
        badge10 = Image.open(awayTeam+" badge.png")
        badge10 = badge10.resize((20, 20), Image.ANTIALIAS)
        badge10 = ImageTk.PhotoImage(badge10)
        canvas.create_image(200,300, anchor=NW, image=badge10)
        
        #Displaying the badge for home team for match 6
        homeTeam=(weeksData[5][0]).lower()
        badge11 = Image.open(homeTeam+" badge.png")
        badge11 = badge11.resize((20, 20), Image.ANTIALIAS)
        badge11 = ImageTk.PhotoImage(badge11)
        canvas.create_image(100,350, anchor=NW, image=badge11)
        #Displaying away team badge for match 6
        awayTeam=(weeksData[5][1]).lower()
        badge12 = Image.open(awayTeam+" badge.png")
        badge12 = badge12.resize((20, 20), Image.ANTIALIAS)
        badge12 = ImageTk.PhotoImage(badge12)
        canvas.create_image(200,350, anchor=NW, image=badge12)
        
        #Displaying the badge for home team for match 7
        homeTeam=(weeksData[6][0]).lower()
        badge13 = Image.open(homeTeam+" badge.png")
        badge13 = badge13.resize((20, 20), Image.ANTIALIAS)
        badge13 = ImageTk.PhotoImage(badge13)
        canvas.create_image(100,400, anchor=NW, image=badge13)
        #Displaying away team badge for match 7
        awayTeam=(weeksData[6][1]).lower()
        badge14 = Image.open(awayTeam+" badge.png")
        badge14 = badge14.resize((20, 20), Image.ANTIALIAS)
        badge14 = ImageTk.PhotoImage(badge14)
        canvas.create_image(200,400, anchor=NW, image=badge14)

        #Displaying the badge for home team for match 8
        homeTeam=(weeksData[7][0]).lower()
        badge15 = Image.open(homeTeam+" badge.png")
        badge15 = badge15.resize((20, 20), Image.ANTIALIAS)
        badge15 = ImageTk.PhotoImage(badge15)
        canvas.create_image(100,450, anchor=NW, image=badge15)
        #Displaying away team badge for match 8
        awayTeam=(weeksData[7][1]).lower()
        badge16 = Image.open(awayTeam+" badge.png")
        badge16 = badge16.resize((20, 20), Image.ANTIALIAS)
        badge16 = ImageTk.PhotoImage(badge16)
        canvas.create_image(200,450, anchor=NW, image=badge16)

        #Displaying the badge for home team for match 9
        homeTeam=(weeksData[8][0]).lower()
        badge17 = Image.open(homeTeam+" badge.png")
        badge17 = badge17.resize((20, 20), Image.ANTIALIAS)
        badge17 = ImageTk.PhotoImage(badge17)
        canvas.create_image(100,500, anchor=NW, image=badge17)
        #Displaying away team badge for match 9
        awayTeam=(weeksData[8][1]).lower()
        badge18 = Image.open(awayTeam+" badge.png")
        badge18 = badge18.resize((20, 20), Image.ANTIALIAS)
        badge18 = ImageTk.PhotoImage(badge18)
        canvas.create_image(200,500, anchor=NW, image=badge18)

        #Displaying the badge for home team for match 10
        homeTeam=(weeksData[9][0]).lower()
        badge19 = Image.open(homeTeam+" badge.png")
        badge19 = badge19.resize((20, 20), Image.ANTIALIAS)
        badge19 = ImageTk.PhotoImage(badge19)
        canvas.create_image(100,550, anchor=NW, image=badge19)
        #Displaying away team badge for match 10
        awayTeam=(weeksData[9][1]).lower()
        badge20 = Image.open(awayTeam+" badge.png")
        badge20 = badge20.resize((20, 20), Image.ANTIALIAS)
        badge20 = ImageTk.PhotoImage(badge20)
        canvas.create_image(200,550, anchor=NW, image=badge20)
    else:
        errorMessage = tk.Label(window, text="Please enter a matchweek greater than 0 and less than 39")
        errorMessage.config(font=('Arial', 10))
        canvas.create_window(200, 100, window=errorMessage)

def seasonSimulationDisplay():
    results=seasonSimulation()
    #canvas.create_image(200,550, anchor=NW, image=team1)
    for counter in range(20):
        imageShift2=counter*25
        #Displaying Team names for matches
        position=str(counter+1)
        points=str(results[19-counter][1])
        table = tk.Label(window, text=position+" "+results[19-counter][0]+" "+points)
        table.config(font=('Arial', 10))
        canvas.create_window(700, 100+imageShift2, window=table)

def trueAverageStatsDisplay():
    teamname=str(teamnameInput.get())
    #Using the trueAverageStats function to create a 1D array containing the average stats of a given team
    averageStats=trueAverageStats(teamname)
    if averageStats=="invalid":
        errorMessage = tk.Label(window, text="Please can you enter one of the following clubs:")
        errorMessage.config(font=('Arial', 10))
        canvas.create_window(1100, 100, window=errorMessage)
        errorMessage = tk.Label(window, text="Arsenal, AstonVilla, Brighton, Burnley, Chelsea")
        errorMessage.config(font=('Arial', 10))
        canvas.create_window(1100, 120, window=errorMessage)
        errorMessage = tk.Label(window, text="CrystalPalace, Everton, Fulham, Leeds, Leicester")
        errorMessage.config(font=('Arial', 10))
        canvas.create_window(1100, 140, window=errorMessage)
        errorMessage = tk.Label(window, text="Liverpool,  ManchesterCity,  ManchesterUnited")
        errorMessage.config(font=('Arial', 10))
        canvas.create_window(1100, 160, window=errorMessage)
        errorMessage = tk.Label(window, text="NewcastleUnited,  SheffieldUnited, Southampton")
        errorMessage.config(font=('Arial', 10))
        canvas.create_window(1100, 180, window=errorMessage)
        errorMessage = tk.Label(window, text="TottenhamHotspur, WestBrom, WestHam, Wolves")
        errorMessage.config(font=('Arial', 10))
        canvas.create_window(1100, 200, window=errorMessage)
    else:
        #Initialising an array which will hold the average stats alongside a label describing what they are
        averageStatsDisplay=["Goals Scored: ","","Goals Conceded: ","","Shots Taken: ","","Shots Conceded: ","","Shots on Target: ","","Shots on Target Against: ","","Corners For: ","","Corners Against: ",""]
        #Iterating through and display each statistic
        for counter in range(6):
            #Inputting the averaegStats into the averageStatsDisplay array
            averageStatsDisplay[1+counter*2]=str(averageStats[counter])
            imageShift3=counter*25
            #Displaying the average Statistics
            table = tk.Label(window, text=averageStatsDisplay[counter*2]+averageStatsDisplay[(counter*2)+1])
            table.config(font=('Arial', 10))
            canvas.create_window(1100, 100+imageShift3, window=table)  


#Defining the canvas
canvas = tk.Canvas(window, width = 1400, height = 700)
canvas.pack()

#Creating input boxes
#Creating label for the teamname input box
teamnameLabel = tk.Label(window, text="Please input the name of the team who's stats you would like to calculate")
teamnameLabel.config(font=('Arial', 10))
canvas.create_window(1100, 60, window=teamnameLabel)
#Creating the teamname input box
teamnameInput = tk.Entry(window)
canvas.create_window(1100, 80, window=teamnameInput)

#Creating the label for the matchweek input box
matchweekLabel = tk.Label(window, text="Please input the matchweek you would like to predict")
matchweekLabel.config(font=('Arial', 10))
canvas.create_window(200, 60, window=matchweekLabel)
#Creating the matchweek input box
matchweekInput = tk.Entry(window)
canvas.create_window(200, 80, window=matchweekInput)

#Creating the buttons
simulateSeasonButton = tk.Button (window, text='Simulate Season',command=seasonSimulationDisplay, bg='palegreen2', font=('Arial', 11, 'bold')) 
canvas.create_window(700, 30, window=simulateSeasonButton)

predictMatchesButton = tk.Button (window, text='Predict Matches',command=predictionsDisplay, bg='palegreen2', font=('Arial', 11, 'bold')) 
canvas.create_window(200, 30, window=predictMatchesButton)

averageStatsButton = tk.Button (window, text='calculate average stats',command=trueAverageStatsDisplay, bg='palegreen2', font=('Arial', 11, 'bold')) 
canvas.create_window(1100, 30, window=averageStatsButton)

#Creating the static displays
#calculating the average Statistics over each of teh past 10seasons
seasonData=seasonComparison()
#Creating an array of headings for the table
headings=["Year","HG","AG","HS","AS","HST","AST","HC","AC",""]
#Creating a key to explain each of the headings
keys=["HG=Home Goals; AG=Away Goals; HS=Home Shots, AS=Away Shots","HST=Home Shots on Target; AST=Away Shots on Target","HC=Home Corners; AC=Away Corners"]
#Displaying the keys
for counter in range(3):
    downShift=counter*25
    key=keys[counter]
    keyDisplay = tk.Label(window, text=key)
    keyDisplay.config(font=('Arial', 10))
    canvas.create_window(1100, 285+downShift, window=keyDisplay)    
#Iterating through rows
for row in range(10):
    downShift=row*25
    leftShift=row*35
    heading=headings[row]
    header = tk.Label(window, text=heading)
    header.config(font=('Arial', 10))
    canvas.create_window(970+leftShift, 375, window=header)
    #Iterating through columns
    for column in range(9):
        leftShift=column*35
        #rounding the data from the seasonData array to 2 decimal places
        data=round(seasonData[row][column], 2)
        #Displaying the average Statistics for each season
        table = tk.Label(window, text=data)
        table.config(font=('Arial', 10))
        canvas.create_window(970+leftShift, 400+downShift, window=table)
        
#Displaying the mean and variance of goals scored over the past 10 years
averageGoals, goalVariation = meanVariance()
averageGoalsDisplay = tk.Label(window, text="The average number of goals scored in a premier league game is: "+str(averageGoals))
averageGoalsDisplay.config(font=('Arial', 10))
canvas.create_window(1100, 660, window=averageGoalsDisplay)

goalVariationDisplay = tk.Label(window, text="The average spread of goals scored in a premier league game is: "+str(goalVariation))
goalVariationDisplay.config(font=('Arial', 10))
canvas.create_window(1100, 680, window=goalVariationDisplay)

window.mainloop()
