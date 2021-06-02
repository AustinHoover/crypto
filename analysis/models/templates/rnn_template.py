import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#
#
# Constants and configuration
#
#

#Constants we're going to use for this neural network
timeWindowSize = 10 # This is the size of the window we're going to use to try to predict the next value
normalizationConstant = 100 # Constant to scale the dataset by to normalize data for easier training
predictColumn = "value" # The column in the dataset we read in that we're trying to predict the future value or
batchSize = 10 # size of batch of data
epochs = 50 # number epochs to train


#
#
# WORK BEGINS
#
#

#
# DATA READ IN
#

# first we read in the dataset with pandas 
inFrame = pd.read_csv('../../datasets/simple.csv')
# next we want to calculate the size of the raw dataset
rawDataLength = len(inFrame)
# number of values we'll actually get to train on is rawDataLength - timeWindowSize
trainDataLength = rawDataLength - timeWindowSize - timeWindowSize

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
unNormalizedTimeAdjustedOutFrame = pd.DataFrame({predictColumn:timeShiftedSeriesToPredict[timeWindowSize:-timeWindowSize]})
# normalize the out frame
timeAdjustedOutFrame = unNormalizedTimeAdjustedOutFrame.divide(normalizationConstant)


#
# METRICS CREATION
#

# This is where we append a shit ton of metrics to the original input data
# We also remove columns from the original in data if they're not valid

# This is the dataframe that's going to store the culled in variables
culledInFrame = inFrame

# remove columns
culledInFrame = culledInFrame.drop(
    ['timestamp'], # column to remove
    axis=1, # means we're removing columns not rows
)

# normalize the culled fraome
normalizedCulledFrame = culledInFrame.divide(normalizationConstant)

# First we'll start by declaring our augmented frame contents
# Basically this is a list of all the individual frames we calculate along the way
# At the end this list is going to be unioned into one big frame
augmentedFrameSubframes = [normalizedCulledFrame]

# Create a dataframe of just the main value we're predicting from
mainValueFrame = pd.DataFrame({predictColumn:inFrame[predictColumn]})
# Create rolling frame off of the main value frame
mainValueRolling = mainValueFrame.rolling(timeWindowSize,win_type=None,)

# example of adding a metric (std dev) to the augmented frame
# std = mainValueRolling.std()
# std = std.rename(columns={predictColumn:'StdDev'})
# augmentedFrameSubframes.append(std)

# Compile the augmented frame
# That big ol' list of frames we were compiling up to this point is what this works with
# It unions the list into a single really big dataframe
augmentedFrame = pd.concat(
    augmentedFrameSubframes,
    axis=1,
    join='inner',
)

# next for our original dataframe we want to cut off the end by size timeWindowSize
# this way we have matching lengths and the future value index corresponds to the current frame index
timeAdjustedInFrame = augmentedFrame[timeWindowSize:-timeWindowSize]

# Lets verify they all are the same
print("These numbers should be identical: ",trainDataLength," ",len(timeAdjustedInFrame)," ",len(timeAdjustedOutFrame))
print(timeAdjustedInFrame)
print(timeAdjustedOutFrame)
if trainDataLength==len(timeAdjustedInFrame)==len(timeAdjustedOutFrame):
    print("Verified!")
else:
    print("Data series are not same length or are not the intended length!")
    quit()


# Create the dataset object based on those carefully crafted data series
dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    timeAdjustedInFrame,
    timeAdjustedOutFrame,
    sequence_length=timeWindowSize, # When we're looking at a single window, this is the number of timestamps in said window
    # It should be obvious that this should be equivalent to the variable timeWindowSize

    sequence_stride=1, # [window starts at i],[window starts at i + sequence_stride],..
    sampling_rate=1, # [[i],[i+sampling_rate],[i+sampling_rate*2],..]
    batch_size=batchSize,
)


#
# CALCULATE INPUT-OUTPUT SHAPES
#

# Dataset objects can be cast to a python iterator using iter()
# This code uses that to get the first element we're going to train on
# It then figures out the shape of that data and uses that to figure out shapes
# for the first and last layers in the model

# Cast to iterator
datasetIter = iter(dataset)
# Get first element from iterator
datasetFirst = next(datasetIter)

# Get in element of first element and set expected in shape off of it
firstIn = datasetFirst[0]
expectedInShape = (firstIn.shape[1:]) # We chop off the first dimension because that's the size of the batch

# Get out element of first element and set the expected out shape off of it
firstOut = datasetFirst[1]
expectedOutShape = (firstOut.shape[1:]) # We chop off the first dimension because that's the size of the batch

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

model.add(layers.LSTM(
    timeWindowSize,
    # return_sequences=True   # If you want to pass the LSTM into another LSTM, must have return_sequences=True
))
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
prediction = model.predict([
    [
        [0.605],
        [0.615],
        [0.625],
        [0.635],
        [0.645],
        [0.655],
        [0.665],
        [0.675],
        [0.685],
        [0.695]
        ]
    ])
print(prediction)