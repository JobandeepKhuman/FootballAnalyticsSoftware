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