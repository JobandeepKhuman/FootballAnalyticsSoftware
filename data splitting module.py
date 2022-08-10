import pandas as pd
import numpy as np

premData=pd.read_excel('AllPremData.xlsx')
premData=np.array(premData)






#Scaling all the columns of feature array to have a mean of 0 and standard deviation of 1
def dataPreProcessing(featureArray):
    scaledArray=[1,2,3,4,5,6,7,8]
    totalColumns=featureArray.shape[1]
    for i in range(totalColumns): #Iterating through each column
        currentColumn=featureArray[:,i]
        mean=currentColumn.mean() #Mean value for current column
        standardDeviation=currentColumn.std() #Standard Deviation for current column
        j=((featureArray[:,i]-mean)/standardDeviation)#Scaling each element of the column
        scaledArray[i]=j
    return scaledArray

def dataSplit(data, testSize):
    totalRows=data.shape[0]
    testRows=np.round(totalRows*testSize)
    randomRowNum=np.random.randint(0, int(totalRows), int(testRows)) #Randomly generating row numbers
    testData=np.array([data[i] for i in randomRowNum]) #Creating test dataset
    data=np.delete(data, randomRowNum, axis=0) #Creating training dataset by deleting the test rows from main dataset
    trainingFeatures=data[:,:-1] #Assigning all columns but the full time result column of training data array to training features array
    trainingTarget=data[:,-1] #Assiging the full time result column of trainingData array to the trainingTarget array
    testingFeatures=testData[:,:-1] #Assigning all columns but the full time result column of testing data array to testing features array
    testingTarget=testData[:,-1] #Assiging the full time result column of testingData array to the testingTarget array
    testingFeatures=dataPreProcessing(testingFeatures)
    trainingFeatures=dataPreProcessing(trainingFeatures)
    return trainingFeatures, trainingTarget, testingFeatures, testingTarget

trainingFeatures, trainingTarget, testingFeatures, testingTarget = dataSplit(premData, 0.4)
print(trainingFeatures)
print(trainingTarget)
print(testingFeatures)
print(testingFeatures)



