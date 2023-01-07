from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import pandas as pd
import numpy as np
import random

from MachineLearning import *
from StatisticalAnalysis import *
from GUI_Function_Calculation import *
from GUI_Function_Display import *

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


seasonData=pd.read_excel('2021Data.xlsx')#Reading all the features from the excel file
seasonData=np.array(seasonData, dtype=str)#Assinging the features arrray to hold integers







window=tk.Tk() 

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