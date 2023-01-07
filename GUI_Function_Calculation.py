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