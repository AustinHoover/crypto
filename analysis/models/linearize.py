import csv
import math
import numpy as np


rawData = open('./datasets/fulldata.csv')
parsedData = csv.reader(rawData)

trainLinearized = open('./datasets/linearizedtrain.csv','w',newline='')
trainWriter = csv.writer(trainLinearized)

testLinearized = open('./datasets/linearizedtest.csv','w',newline='')
testWriter = csv.writer(testLinearized)

header = next(parsedData)
valuesPerTimestamp = len(header) #number of per-timestamp things to include

incrementsPerHour = 60
hoursToTrain = 4
rowsToTrain = incrementsPerHour*hoursToTrain#This is how 'far back' we're going to 'look' 60 mins/hr * X hrs/day
hoursToTest = 1
rowsToPredict = incrementsPerHour*hoursToTest #going to try to predict 1 hr in the future

datapointFrameWidth = rowsToTrain * valuesPerTimestamp #This is the width of the frame of datapoints per input

print("Values per timestamp: ",valuesPerTimestamp," width of single datapoint: ",datapointFrameWidth)


trainHeader = []
for i in range(0,rowsToTrain * valuesPerTimestamp):
    trainHeader.append(str(i))
trainWriter.writerow(trainHeader)

testHeader = []
for i in range(0,rowsToPredict * 1): #num rows to predict * what we're predicting (close)
    testHeader.append(str(i))
testWriter.writerow(testHeader)

print("Counting number of rows...")
numRows = sum(1 for row in parsedData)
print(numRows," rows")



currentTrainPool = []
currentTestPool = []
currentTestClosePool = []

currentTrainBatch = []
currentTestBatch = []




input_shape = (1, rowsToTrain, valuesPerTimestamp,)

dataframeSize = rowsToTrain * valuesPerTimestamp

output_shape = rowsToPredict

print("Rows to train ",rowsToTrain)


#reset position
rawData.seek(0) # this sets the file pointer to POSITION 1 --- ONE CHARACTER DEEP NOT ONE ROW DINGUS
next(parsedData) # actually increments a row
#init training data
parsingStart = 0 #Beginning point to start parsing
parsingEnd = rowsToTrain #Ending point to stop parsing
for i in range(parsingStart,parsingEnd):
    currentRow = next(parsedData)
    currentFrame = list(map(lambda x: float(x),currentRow))
    currentTrainPool = currentTrainPool + currentFrame
    #print(currentFrame)
print("current Train pool size ",len(currentTrainPool))
#init testing data
parsingStart = 0 #Beginning point to start parsing
parsingEnd = rowsToPredict #Ending point to stop parsing
for i in range(parsingStart,parsingEnd):
    currentRow = next(parsedData)
    currentFrame = list(map(lambda x: float(x),currentRow))
    currentTestPool = currentTestPool + currentFrame
    currentTestClosePool.append(currentFrame[4])
#train..
parsingStart = 0
parsingEnd = math.floor((numRows - 1 - rowsToTrain - rowsToPredict))
for i in range(parsingStart,parsingEnd):
    #currentTrainPool
    currentTrainPool = currentTrainPool[valuesPerTimestamp:] + currentTestPool[0:valuesPerTimestamp]
    #currentTestPool
    currentRow = next(parsedData)
    currentFrame = list(map(lambda x: float(x),currentRow))
    currentTestPool = currentTestPool[valuesPerTimestamp:] + currentFrame
    #currentTestClosePool
    currentTestClosePool.append(currentFrame[4])
    currentTestClosePool.pop(0)
    #write out
    trainWriter.writerow(currentTrainPool)
    testWriter.writerow(currentTestClosePool)
    #print progress
    if(i % 5 == 0):
        print("\r",(i/parsingEnd),end='')
    #currentTrainBatch
    # currentTrainBatch = np.array([currentTrainPool])
    # #currentTestBatch
    # currentTestBatch = np.array([currentTestClosePool])
    # if(i % 5 != 0):
    #     #run train
    #     acc = model.train_on_batch(
    #         currentTrainBatch,
    #         y=currentTestBatch,
    #     )
    # else:
    #     #run test
    #     acc = model.test_on_batch(
    #         currentTrainBatch,
    #         y=currentTestBatch,
    #     )
    #     print("\r",(i/parsingEnd)," ",acc,end='')
print()
print("Done!")
trainWriter.close()
testWriter.close()