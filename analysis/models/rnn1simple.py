import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#Constants we're going to use for this neural network
timeWindowSize = 10 # This is the size of the window we're going to use to try to predict the next value
normalizationConstant = 100 # Constant to scale the dataset by to normalize data for easier training
predictColumn = "value" # The column in the dataset we read in that we're trying to predict the future value or
batchSize = 10 # size of batch of data
epochs = 50 # number epochs to train



#
#WORK BEGINS
#

#
#DATA READ IN AND MANIPULATION
#
# first we read in the dataset with pandas 
inFrame = pd.read_csv('../datasets/simple.csv')
# next we want to calculate the size of the raw dataset
rawDataLength = len(inFrame)
# number of values we'll actually get to train on is rawDataLength - timeWindowSize
trainDataLength = rawDataLength - timeWindowSize

# our next goal is to time shift the data we want to predict
# To do this with pandas it needs to be a series not a frame
# So we:
# - resolve the series we want to predict
# - time shift it
# - construct a new dataframe of just this time shifted series
# The length of this new frame should be equal to trainDataLength
# if it isn't we fucked up

# resolve series
seriesToPredict = inFrame[predictColumn]
# time shift
timeShiftedSeriesToPredict = seriesToPredict.shift(periods=-timeWindowSize) # shift it back one window
# construct new frame
timeShiftedTargetFrame = pd.DataFrame({predictColumn:timeShiftedSeriesToPredict[:-timeWindowSize]})

# next for our original dataframe we want to cut off the end by size timeWindowSize
# this way we have matching lengths and the future value index corresponds to the current frame index
timeAdjustedInFrame = inFrame[:-timeWindowSize]

# Lets verify they all are the same
print("These numbers should be identical: ",trainDataLength," ",len(timeAdjustedInFrame)," ",len(timeShiftedTargetFrame))
print(timeAdjustedInFrame)
print(timeShiftedTargetFrame)
print("Verified!")

normalizedFrame = inFrame.divide(normalizationConstant)

maxWindowSize = 10 #should be set to the size of the largest rolling window
futureTerm = "value"

seriesNormalizedClose = normalizedFrame[futureTerm]
seriesShiftedNormalizedClose = seriesNormalizedClose.shift(periods=-maxWindowSize)

frameClose = pd.DataFrame({futureTerm:normalizedFrame[futureTerm]})
frameShiftedClose = pd.DataFrame({futureTerm:seriesShiftedNormalizedClose})[0:-maxWindowSize]

finalFrame = pd.DataFrame({futureTerm:normalizedFrame[0:-maxWindowSize][futureTerm]})
print("dataset desc")
print(finalFrame)
print(frameShiftedClose)
print("LENGTH ",len(finalFrame))

minStartIndex = maxWindowSize
sequenceLength = maxWindowSize

dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    finalFrame,
    frameShiftedClose,
    sequence_length=sequenceLength, # When we're looking at a single window, this is the number of timestamps in said window
    sequence_stride=1, # [window starts at i],[window starts at i + sequence_stride],..
    sampling_rate=1, # [[i],[i+sampling_rate],[i+sampling_rate*2],..]
    batch_size=10,
) # end_index=len(finalFrame)-minStartIndex)

# trainShape = next(dataset.as_numpy_iterator()).shape
#read in test data
# testWindowGenerator = tf.keras.preprocessing.timeseries_dataset_from_array(justCloseFrame,None,sequence_length=maxWindowSize,start_index=minStartIndex*2)
# testShape = next(testWindowGenerator.as_numpy_iterator()).shape
#combined
# combinedSet = tf.data.Dataset.zip((trainWindowGenerator,testWindowGenerator))
# print(combinedSet)
print(dataset," ",len(dataset))
datasetIter = iter(dataset)
datasetFirst = next(datasetIter)


# Used to validate translation of values
# print("datasets")
# print("train")
# datasetIter = iter(dataset)
# for row in datasetIter:
#     print(row[0])
#     print("to")
#     print(row[1])
#     print("\n\n\n\n")

# datasetIter = iter(dataset)
# print("test")
# for row in datasetIter:
#     print(row[1])

firstTrain = datasetFirst[0]
firstTest = datasetFirst[1]
print("shapes: ",firstTrain.shape," ",firstTest.shape)
# print("vs")
# print(testShape," ",len(testWindowGenerator))

expectedInShape = (firstTrain.shape[1:])
expectedOutShape = (firstTest.shape[1:])

# This tells us the shape that's going in and the shape we're getting out
print("Model translates: ",expectedInShape,"=>",expectedOutShape)


# Hey! We get to define what the actual neural network looks like now
# Notes for each layer and call will be underneath the line of code in this phase

model = keras.Sequential()
# Sequential begins the model
# This is what we will add layers to

model.add(keras.Input(shape=expectedInShape))
# Our input shape defines the size of what we're passing in each run
# If you let the code above do its thing then this should automatically be calculated for you
# and you shouldn't have to mess with it at all

model.add(layers.LSTM(maxWindowSize))
# basic LSTM layer - pretty self explanitory
# It uses the past to predict the future

denseLayer = layers.Dense(1)
model.add(denseLayer)
# We're just looking to reshape data, so there shouldn't be an activation function
# refer to https://www.tensorflow.org/tutorials/structured_data/time_series -> ctrl + f "activation" -> first result
# to give causus beli for why there shouldn't be an activation function

model.summary()
# Lets just get a summary of that beautiful new model

# End model building phase


# Congrats! We're about to compile your model! Before we do, lets just check a few things
# Make sure the arguments are correct. Refer to
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
# Specifically, we want to make sure our loss function makes sense, refer to
# https://www.tensorflow.org/api_docs/python/tf/keras/losses
# Also check your metric
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics
# As a simple case..
# if you're CATEGORIZING THINGS use 
# loss="categorical_crossentropy"
# optimizer="adam"
# metrics=["accuracy"]
# if you're PREDICTING A VALUE use
# loss="MSE"
# metrics=["MSE"]
model.compile(loss="MSE", metrics=["MSE"])


#DO NOT SPECIFY batch_size IF YOUR INPUT IS A DATASET: https://keras.io/api/models/model_training_apis/
history = model.fit(
    dataset,
    verbose=2, # 0 for no output, 1 for progress bar, 2 for no progress bar
    epochs=epochs
)

#Lets try a prediction
prediction = model.predict([[[0.605],[0.615],[0.625],[0.635],[0.645],[0.655],[0.665],[0.675],[0.685],[0.695]]])
print(prediction)