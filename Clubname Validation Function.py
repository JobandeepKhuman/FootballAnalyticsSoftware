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

test="chelsee"
name=clubnameValidation(test)
test="chel134esa"
name=clubnameValidation(test)
test="ARSENAL"
name=clubnameValidation(test)

