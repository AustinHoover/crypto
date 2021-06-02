
import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

#
#
# Constants and configuration
#
#

#Constants we're going to use for this neural network
timeWindowSizeIn = 60 # This is the size of the window we're going to use to try to predict the next value
timeWindowSizeOut = 10
normalizationConstant = 400 # Constant to scale the dataset by to normalize data for easier training
predictColumn = "Close" # The column in the dataset we read in that we're trying to predict the future value or
batchSize = 64 # size of batch of data
epochs = 200 # number epochs to train
verbosity = 1 # how much should we log
validationSplit = 0.2 # what percentage of the data to use to validate training
learningRate = 0.001 # learning rate for optimizer

# Stride
# Basically when we're going to pull from the dataset how often do we want to pull out a window of data
# ie if stride=1 we pull out a window for every datapoint
#    if stride=2 we pull out a window for every other datapoint
# and so on
# you can think of it like:
# [[Window of data at i=0], [Window of data starting at i=1*stride], [Window of data starting at i=2*stride],..]
# The effect is:
# Increasing stride dramatically decreases train time and fights overfitting
# Decreasing stride is "similar" to increasing epochs in that it "gives" you "more data"
sequenceStride = 1


#
#
# WORK BEGINS
#
#

#
# DATA READ IN
#

# first we read in the dataset with pandas 
inFrame = pd.read_csv('../datasets/newAugmented.csv')
# next we want to calculate the size of the raw dataset
rawDataLength = len(inFrame)
# number of values we'll actually get to train on is rawDataLength - timeWindowSizeOut - timeWindowSizeIn
# Why minus 2 * timeWindowSizeIn?
# We won't have future values for the end of the target column once we shift it back by timeWindowSizeIn
# Separately, we won't have non-NaN calculations on our rolling calculations for the first timeWindowSizeIn values of our input data
# Therefore you have to chop off both ends
# This alone would work except if the time frames are of variable size, then you need to chop off more for the in data than the out
# ie we'll have more out windows than in windows, so you want to chop off the in window size from the end
actualTotalDataLength = rawDataLength - timeWindowSizeOut * 2 - timeWindowSizeIn * 2
trainDataLength = int(actualTotalDataLength * (1.0-validationSplit))
validationDataLength = int(actualTotalDataLength * validationSplit)

# our next goal is to time shift the data we want to predict
# To do this with pandas it needs to be a series not a frame
# So we:
# - resolve the series we want to predict
# - time shift it
# - construct a new dataframe of just this time shifted series
# The length of this new frame should be equal to actualTotalDataLength
# if it isn't we fucked up

# resolve series
seriesToPredict = inFrame[predictColumn]
# time shift
timeShiftedSeriesToPredict = seriesToPredict.diff().shift(periods=-timeWindowSizeOut) # shift it back one window
# construct new frame                                                                    remove NaN          Chop off end to line up sets
unNormalizedTimeAdjustedOutFrame = pd.DataFrame({predictColumn:timeShiftedSeriesToPredict[(timeWindowSizeIn * 2):-timeWindowSizeOut]})
# difference the out frame
# differencedOutFrame = unNormalizedTimeAdjustedOutFrame.diff()
# then normalize
timeAdjustedOutFrame = unNormalizedTimeAdjustedOutFrame.divide(normalizationConstant).add(0.5)


#
# METRICS CREATION
#

# This is where we append a shit ton of metrics to the original input data
# We also remove columns from the original in data if they're not valid

# This is the dataframe that's going to store the culled in variables
culledInFrame = inFrame

# remove columns
culledInFrame = culledInFrame.drop(
    ['Timestamp','Volume_(BTC)','Volume_(Currency)'], # column to remove
    axis=1, # means we're removing columns not rows
)


# difference the culled frame
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.diff.html
# why do this?
# https://otexts.com/fpp2/stationarity.html
differencedCulledFrame = culledInFrame.diff()

# Then need to normalize
normalizedCulledFrame = differencedCulledFrame.divide(normalizationConstant).add(0.5)

# First we'll start by declaring our augmented frame contents
# Basically this is a list of all the individual frames we calculate along the way
# At the end this list is going to be unioned into one big frame
augmentedFrameSubframes = [normalizedCulledFrame]

# Create a dataframe of just the main value we're predicting from
mainValueFrame = pd.DataFrame({predictColumn:normalizedCulledFrame[predictColumn]})
# Create rolling frame off of the main value frame
mainValueRolling = mainValueFrame.rolling(timeWindowSizeIn,win_type=None,)

# example of adding a metric (std dev) to the augmented frame
# std = mainValueRolling.std()
# std = std.rename(columns={predictColumn:'StdDev'})
# augmentedFrameSubframes.append(std)
# sma
sma = mainValueRolling.mean()
sma = sma.rename(columns={'Close':'SimpleMovingAverage1Hr'})
augmentedFrameSubframes.append(sma)
# stddev
rstd = mainValueRolling.std()
rstd = rstd.rename(columns={'Close':'StdDev1Hr'})
augmentedFrameSubframes.append(rstd)
# ema
ema = mainValueFrame.ewm(span=timeWindowSizeIn).mean()
ema = ema.rename(columns={'Close':'ExponentialMovingAverage1Hr'})
augmentedFrameSubframes.append(ema)
# bollinger upper
upper_band = sma['SimpleMovingAverage1Hr'] + 2 * rstd['StdDev1Hr']
upper_band = pd.DataFrame({'BollingerUpper1Hr':upper_band})
augmentedFrameSubframes.append(upper_band)
# bollinger lower
lower_band = sma['SimpleMovingAverage1Hr'] - 2 * rstd['StdDev1Hr']
lower_band = pd.DataFrame({'BollingerLower1Hr':lower_band})
augmentedFrameSubframes.append(lower_band)
# temporal sinusoid
# ???
# variance
var = mainValueRolling.var()
var = var.rename(columns={'Close':'Variance1Hr'})
augmentedFrameSubframes.append(var)
# correlation
# corr = mainValueRolling.corr()
# corr = corr.rename(columns={'Close':'Correlation1Hr'})
# augmentedFrameSubframes.append(corr)
# covariance
# cov = mainValueRolling.cov()
# cov = cov.rename(columns={'Close':'Covariance1Hr'})
# augmentedFrameSubframes.append(cov)
# skew
# skew = mainValueRolling.skew()
# skew = skew.rename(columns={'Close':'Skew1Hr'})
# augmentedFrameSubframes.append(skew)
# kurtosis
# kurt = mainValueRolling.kurt()
# kurt = kurt.rename(columns={'Close':'Kurtosis1Hr'})
# augmentedFrameSubframes.append(kurt)
# quantile
# quantile = mainValueRolling.quantile(0.5)
# quantile = quantile.rename(columns={'Close':'Quantile1Hr'})
# augmentedFrameSubframes.append(quantile)
# standard error of mean
# sem = mainValueRolling.sem()
# sem = sem.rename(columns={'Close':'StandardErrorOfMean1Hr'})
# augmentedFrameSubframes.append(sem)


# Compile the augmented frame
# That big ol' list of frames we were compiling up to this point is what this works with
# It unions the list into a single really big dataframe
augmentedFrame = pd.concat(
    augmentedFrameSubframes,
    axis=1,
    join='inner',
)

print(augmentedFrame)

# next for our original dataframe we want to cut off the end by size timeWindowSize
# this way we have matching lengths and the future value index corresponds to the current frame index
timeAdjustedInFrame = augmentedFrame[timeWindowSizeIn:-(timeWindowSizeOut * 2)]

# Lets verify they all are the same
numberInWindows =  len(timeAdjustedInFrame)  - timeWindowSizeIn
numberOutWindows = len(timeAdjustedOutFrame) - timeWindowSizeOut
print("These numbers should be identical: ",actualTotalDataLength," ",numberInWindows," ",numberOutWindows)
if actualTotalDataLength==numberInWindows==numberOutWindows:
    print("Verified!")
else:
    print("In frame")
    print(timeAdjustedInFrame)
    print("Out frame")
    print(timeAdjustedOutFrame)
    print("Divided lengths")
    print("Number in windows ",numberInWindows)
    print("Number out windows ",numberOutWindows)
    print("Data series are not same length or are not the intended length!")
    quit()

print("These are what your in and out datasets look like: ")
print("In frame")
print("\n")
print(timeAdjustedInFrame)
print("\n")
print("Out frame")
print("\n")
print(timeAdjustedOutFrame)
print("\n")

print("In batches  ",numberInWindows/batchSize)
print("Out batches ",numberOutWindows/batchSize)


# Create the dataset object based on those carefully crafted data series
x_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    timeAdjustedInFrame,
    None,
    sequence_length=timeWindowSizeIn, # When we're looking at a single window, this is the number of timestamps in said window
    # It should be obvious that this should be equivalent to the variable timeWindowSizeIn
    start_index=1,
    end_index=trainDataLength + (timeWindowSizeIn - timeWindowSizeOut), # tbh no fuckin clue why we have to add the difference
    sequence_stride=sequenceStride, # [window starts at i],[window starts at i + sequence_stride],..
    sampling_rate=1, # [[i],[i+sampling_rate],[i+sampling_rate*2],..]
    batch_size=batchSize,

)

print("x_train len ",len(x_train))

# CReate train y series
y_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    timeAdjustedOutFrame,
    None,
    sequence_length=timeWindowSizeOut, # When we're looking at a single window, this is the number of timestamps in said window
    # It should be obvious that this should be equivalent to the variable timeWindowSize
    start_index=1,
    end_index=trainDataLength,
    sequence_stride=sequenceStride, # [window starts at i],[window starts at i + sequence_stride],..
    sampling_rate=1, # [[i],[i+sampling_rate],[i+sampling_rate*2],..]
    batch_size=batchSize,
)

print("y_train len ",len(y_train))

# zip the two train series together
trainDataset = tf.data.Dataset.zip((x_train,y_train))


#
# VALIDATION
#
# validation dataset creation
# Create the dataset object based on those carefully crafted data series
x_validate = tf.keras.preprocessing.timeseries_dataset_from_array(
    timeAdjustedInFrame,
    None,
    sequence_length=timeWindowSizeIn, # When we're looking at a single window, this is the number of timestamps in said window
    # It should be obvious that this should be equivalent to the variable timeWindowSizeIn
    start_index=trainDataLength,
    sequence_stride=sequenceStride, # [window starts at i],[window starts at i + sequence_stride],..
    sampling_rate=1, # [[i],[i+sampling_rate],[i+sampling_rate*2],..]
    batch_size=batchSize,

)

print("x_validate len ",len(x_validate))

# CReate train y series
y_validate = tf.keras.preprocessing.timeseries_dataset_from_array(
    timeAdjustedOutFrame,
    None,
    sequence_length=timeWindowSizeOut, # When we're looking at a single window, this is the number of timestamps in said window
    # It should be obvious that this should be equivalent to the variable timeWindowSizeOut
    start_index=trainDataLength,
    sequence_stride=sequenceStride, # [window starts at i],[window starts at i + sequence_stride],..
    sampling_rate=1, # [[i],[i+sampling_rate],[i+sampling_rate*2],..]
    batch_size=batchSize,
)

print("y_validate len ",len(y_validate))


# Validate everything is the same size
if (len(x_train) != len(y_train)) or (len(x_validate) != len(y_validate)):
    print("Datasets are not the same size?!")
    print("x_train shape ",next(iter(x_train)).shape)
    print("y_train shape ",next(iter(y_train)).shape)
    print("x_validate shape ",next(iter(x_validate)).shape)
    print("y_validate shape ",next(iter(y_validate)).shape)
    quit()



# zip the two train series together
validateDataset = tf.data.Dataset.zip((x_validate,y_validate))



#
# CALCULATE INPUT-OUTPUT SHAPES
#

# Dataset objects can be cast to a python iterator using iter()
# This code uses that to get the first element we're going to train on
# It then figures out the shape of that data and uses that to figure out shapes
# for the first and last layers in the model

# Cast to iterator
datasetIter = iter(trainDataset)
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


# Why relu activation?
# https://stackoverflow.com/questions/56022318/lstm-in-python-generating-flat-forecasts


model = keras.Sequential()
# Sequential begins the model
# This is what we will add layers to

model.add(keras.Input(shape=expectedInShape))
# Our input shape defines the size of what we're passing in each run
# If you let the code above do its thing then this should automatically be calculated for you
# and you shouldn't have to mess with it at all

# model.add(layers.LSTM(
#     256,
#     input_shape=expectedInShape,
#     activation='relu',
#     dropout=0.1,
#     recurrent_dropout=0.1,
#     return_sequences=True,
#     ))
# basic LSTM layer - pretty self explanitory
# It uses the past to predict the future

# model.add(layers.LSTM(
#     512,
#     input_shape=expectedInShape,
#     activation='relu',
#     dropout=0.1,
#     recurrent_dropout=0.1,
#     return_sequences=True,
#     ))
# basic LSTM layer - pretty self explanitory
# It uses the past to predict the future


model.add(layers.LSTM(
    60,
    input_shape=expectedInShape,
    activation='relu',
    dropout=0.1,
    recurrent_dropout=0.1,
    return_sequences=True,
    ))
# basic LSTM layer - pretty self explanitory
# It uses the past to predict the future



model.add(layers.Flatten())

model.add(layers.Dense(expectedOutShape[0] * expectedOutShape[1]))

model.add(layers.Reshape(expectedOutShape))
# We need to reshape the data after the work that is done on it
# If the last layer is an LSTM layer:
# We meed to use a dense layer as LSTM will complain otherwise
# We're just looking to reshape data, so there shouldn't be an activation function
# refer to https://www.tensorflow.org/tutorials/structured_data/time_series -> ctrl + f "activation" -> first result
# to give causus beli for why there shouldn't be an activation function
# If the last layer is like a dense or something:
# Use a reshape layer to make things


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
model.compile(
    loss="MSE",
    optimizer=keras.optimizers.Adam(learning_rate=learningRate),
    metrics=[
        # reference 
        # https://keras.io/api/metrics/regression_metrics/
        # for different metrics
        tf.keras.metrics.MeanSquaredError(),
        # tf.keras.metrics.MeanAbsoluteError(),
        # tf.keras.metrics.CosineSimilarity(axis=1),
        # tf.keras.metrics.LogCoshError(),
    ]
    )


# checkpoint code
checkpoint = ModelCheckpoint(
    "../results/best_model.hdf5",
    monitor="val_loss",
    verbose=verbosity,
    save_best_only=True,
    mode="auto",
    person=1
)


#DO NOT SPECIFY batch_size IF YOUR INPUT IS A DATASET: https://keras.io/api/models/model_training_apis/
history = model.fit(
    trainDataset,
    validation_data=validateDataset,
    verbose=verbosity, # 0 for no output, 1 for progress bar, 2 for no progress bar
    epochs=epochs,
    callbacks=[checkpoint],
)

# Save the model
saveName = "../results/more_stats"
model.save(saveName)

#Lets try a prediction
# prediction = model.predict([
#     [
#         [0.605],
#         [0.615],
#         [0.625],
#         [0.635],
#         [0.645],
#         [0.655],
#         [0.665],
#         [0.675],
#         [0.685],
#         [0.695]
#         ]
#     ])
# print(prediction)
