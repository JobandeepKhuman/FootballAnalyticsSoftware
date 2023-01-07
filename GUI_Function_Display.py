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