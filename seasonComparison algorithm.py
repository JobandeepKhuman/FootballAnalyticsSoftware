import numpy as np
import pandas as pd
np.set_printoptions(precision=2, suppress=True)

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

seasonData=seasonComparison()
print(seasonData)


