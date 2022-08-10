import pandas as pd
import numpy as np


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
    
    
averageGoals, goalVariation = meanVariance()
print(averageGoals)
print(goalVariation)