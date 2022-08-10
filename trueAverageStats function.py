import pandas as pd
import numpy as np
np.set_printoptions(precision=8, suppress=True)

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



def trueAverageStats(clubname):
    seasonData=pd.read_excel('2021Data.xlsx')#Reading all the features from the excel file
    seasonData=np.array(seasonData, dtype=str)#Assinging the features arrray to hold integers
    #validating the clubname
    clubname=clubnameValidation(clubname)
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

clubname="liverpool"
averageData=trueAverageStats(clubname)
print(averageData)