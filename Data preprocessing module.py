import numpy as np
#Scaling all the columns of feature array to have a mean of 0 and standard deviation of 1
def dataPreProcessing(featureArray):
    totalColumns=featureArray.shape[1]
    for i in range(totalColumns): #Iterating through each column
        currentColumn=featureArray[:,i]
        mean=currentColumn.mean() #Mean value for current column
        standardDeviation=featureArray.std() #Standard Deviation for each column
        featureArray[:,i]=(featureArray[:,i]-mean)/standardDeviation #Scaling each element of the column
    return featureArray

array=[[10,6,8],[5,2,4],[20,14,4]]
array=np.array(array)
newArray=dataPreProcessing(array)
print(newArray)