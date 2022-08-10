import pandas as pd
import numpy as np

def clubnameValidation(clubname):
    #Defining what names are valid once everythin is lowercase and spaces are removed
    validNames = ['arsenal', 'astonvilla', 'brighton', 'burnley', 'chelsea', 'crystalpalace', 'everton', 'fulham', 'leedsunited', 'leicestercity', 'liverpool', 'manchestercity', 'manchesterunited', 'newcastleunited', 'sheffieldUnited', 'southampton', 'tottenhamhotspur', 'westbromwichalbion', 'westhamunited', 'wolverhamptonwanderers']
    valid = False
    #standardising the input by making it lowercase and removing spaces
    clubname = clubname.lower()
    clubname = clubname.replace(" ","")
    #Checking if the standardised input is in the validNames array
    for counter in range(20):
        if clubname == validNames[counter]:
            valid = True
    if valid == False:
        print("Please enter a valid clubname")
    else:
        print("Valid Clubname")
    return clubname


#Function to calculate the average Statistics of a given team
def averageStats(clubname):
    seasonData=pd.read_excel('2021Data.xlsx')#Reading all the features from the excel file
    seasonData=np.array(seasonData, dtype=str)#Assinging the features arrray to hold integers
    #validating the clubname
    clubname=clubnameValidation(clubname)
    #number of games that the statistics will be averaged over
    k=2
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

np.set_printoptions(precision=8, suppress=True)
clubname="arsenal"
averageData=averageStats(clubname)
print(averageData)