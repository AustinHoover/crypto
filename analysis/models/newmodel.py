import csv
import math
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

#build data
# Model / data parameters

#For each row we want
#Open
#High
#Low
#Close
#Volume (BTC)
#Volume (Currency)
#Weighted Price
#SMA
#TP
#EMA
#SMA_TP
#TPStdDev
#BOLU
#BOLD
#MACD

rawData = open('./datasets/fulldata.csv')
parsedData = csv.reader(rawData)

header = next(parsedData)
valuesPerTimestamp = len(header) #number of per-timestamp things to include

incrementsPerHour = 60
hoursToTrain = 4
rowsToTrain = incrementsPerHour*hoursToTrain#This is how 'far back' we're going to 'look' 60 mins/hr * X hrs/day
hoursToTest = 1
rowsToPredict = incrementsPerHour*hoursToTest #going to try to predict 1 hr in the future

datapointFrameWidth = rowsToTrain * valuesPerTimestamp #This is the width of the frame of datapoints per input

print("Values per timestamp: ",valuesPerTimestamp," width of single datapoint: ",datapointFrameWidth)

#        timestampFrame[currentIndex],
#        exOpenFrame[currentIndex],
#        exHighFrame[currentIndex],
#        exLowFrame[currentIndex],
#        exCloseFrame[currentIndex],
#        btcVolFrame[currentIndex],
#        usdVolFrame[currentIndex],
#        weightedPriceFrame[currentIndex],
#        SMA,
#        TP,
#        EMAt,
#        SMA_TP,
#        TPStdDev,
#        BOLU,
#        BOLD,
#        MACD



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

# x_train = x_train.astype("float32")
# x_test = x_test.astype("float32")
# # Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


#build model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        # layers.Dropout(0.5),
        layers.Dense(dataframeSize,activation="softmax"),
        layers.Dense(60,activation="softmax"),
        layers.Reshape((60,)),
    ]
)


model.summary()

batch_size = 1
epochs = 10

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


trainData = []
testData = []


parsingStart = 0 #Beginning point to start parsing
parsingEnd = rowsToTrain #Ending point to stop parsing
for i in range(parsingStart,parsingEnd):
    currentRow = next(parsedData)
    currentFrame = list(map(lambda x: float(x),currentRow))
    trainData.append(currentFrame)
    #print(currentFrame)
print("current Train pool size ",len(currentTrainPool))
#init testing data
parsingStart = 0 #Beginning point to start parsing
parsingEnd = numRows - rowsToPredict #Ending point to stop parsing
for i in range(parsingStart,parsingEnd):
    currentRow = next(parsedData)
    currentFrame = list(map(lambda x: float(x),currentRow))
    trainData.append(currentFrame)
    testData.append(currentFrame[4])
#read in final testData
parsingStart = numRows - rowsToPredict #Beginning point to start parsing
parsingEnd = numRows #Ending point to stop parsing
for i in range(parsingStart,parsingEnd):
    currentRow = next(parsedData)
    currentFrame = list(map(lambda x: float(x),currentRow))
    testData.append(currentFrame[4])
#train..
for epoch in range(1,numEpochs):
    dataLength = len(testData)
    for i in range(1,dataLength):
        


acc = 0
for epoch in range(1,epochs):
    #go to beginning of data
    rawData.seek(0)
    next(parsedData)
    #init training data
    parsingStart = 0 #Beginning point to start parsing
    parsingEnd = rowsToTrain #Ending point to stop parsing
    for i in range(parsingStart,parsingEnd):
        currentRow = next(parsedData)
        currentFrame = list(map(lambda x: float(x),currentRow))
        currentTrainPool.append(currentFrame)
        #print(currentFrame)
    print("current Train pool size ",len(currentTrainPool))
    #init testing data
    parsingStart = 0 #Beginning point to start parsing
    parsingEnd = rowsToPredict #Ending point to stop parsing
    for i in range(parsingStart,parsingEnd):
        currentRow = next(parsedData)
        currentFrame = list(map(lambda x: float(x),currentRow))
        currentTestPool.append(currentFrame)
        currentTestClosePool.append(currentFrame[4])
    #train..
    parsingStart = 0
    parsingEnd = math.floor((numRows - 1 - rowsToTrain - rowsToPredict))
    #setup time stuff
    totalIncrements = len(range(parsingStart,parsingEnd))
    timeStart = datetime.now()
    for i in range(parsingStart,parsingEnd):
        #currentTrainPool
        currentTrainPool.pop(0)
        currentTrainPool.append(currentTestPool.pop(0))
        #currentTestPool
        currentRow = next(parsedData)
        currentFrame = list(map(lambda x: float(x),currentRow))
        currentTestPool.append(currentFrame)
        #currentTestClosePool
        currentTestClosePool.append(currentFrame[4])
        currentTestClosePool.pop(0)
        #currentTrainBatch
        currentTrainBatch = np.array([currentTrainPool])
        #currentTestBatch
        currentTestBatch = np.array([currentTestClosePool])
        if(i % 5 != 0):
            #run train
            acc = model.train_on_batch(
                currentTrainBatch,
                y=currentTestBatch,
            )
        else:
            #run test
            acc = model.test_on_batch(
                currentTrainBatch,
                y=currentTestBatch,
            )
            timeCurrent = datetime.now()
            estimation = (timeCurrent - timeStart) / (i + 1) * totalIncrements
            print("\r est time: ",estimation," % done ",(i/parsingEnd)," acc: ",acc,"   ",end='')
    print("epoch ",epoch," acc ",acc)
    saveName = "./results/model-e" + str(epoch)
    model.save(saveName)

# model.save('modelState')

#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# score = model.evaluate(x_test,y_test)
# print("Test loss:",score[0])
# print("Test accuracy:",score[1])
