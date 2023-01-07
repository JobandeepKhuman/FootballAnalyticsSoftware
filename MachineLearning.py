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