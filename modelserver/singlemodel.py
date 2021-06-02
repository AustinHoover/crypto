from typing import List
from io import StringIO
import csv
import json
from tensorflow import keras
import numpy as np
import pandas as pd
from flask import Flask
from flask import request


#
#Neural Net
#
modelPath = '../analysis/results/best_model.hdf5'
model = keras.models.load_model(modelPath)
model.summary()


#
# Data manipulation functions
#
#Constants we're going to use for this neural network
timeWindowSize = 60 # This is the size of the window we're going to use to try to predict the next value
normalizationConstant = 400 # Constant to scale the dataset by to normalize data for easier training
predictColumn = 'close'
unnescessaryColumns = ['timestamp','volume']
necessaryColumns = ['open','high','low','close','weighted']

# extracts the timestamps of a dataframe to a series
def extractTimestampSeries(inputFrame: pd.DataFrame) -> pd.Series:
    return inputFrame['timestamp']

def extractTargetColumn(inputFrame : pd.DataFrame) -> pd.Series:
    return inputFrame[predictColumn]

# removes unnescessary columns from some input data
def dropUnnescessaryColumns(inputFrame: pd.DataFrame) -> pd.DataFrame:
    #drops all columns that start with names in this array
    return inputFrame.drop(unnescessaryColumns,axis=1)

# if one of our necessary columns is missing, replicate the target column to it
# basically fudging if it doesn't exist
def augmentMissingColumns(inputFrame: pd.DataFrame) -> pd.DataFrame:
    for column in necessaryColumns:
        if column not in inputFrame:
            # ASSUME predictColumn is in dataframe - if not we're fucked anyway lol
            inputFrame[column] = inputFrame[predictColumn]
    return inputFrame

# differentiate, normalize, and center a dataframe
def normalizeData(inputFrame: pd.DataFrame) -> pd.DataFrame:
    # differentiate data
    differentiated = inputFrame.diff()
    # normalize
    normalized = differentiated.divide(normalizationConstant)
    # center around 0.5
    centered = normalized.add(0.5)
    return centered

def transformRawColumns(inputFrame: pd.DataFrame) -> pd.DataFrame:
    # differentiate and normalize culled frame
    normalizedCulled = normalizeData(inputFrame)
    # First we'll start by declaring our augmented frame contents
    # Basically this is a list of all the individual frames we calculate along the way
    # At the end this list is going to be unioned into one big frame
    augmentedFrameSubframes = [normalizedCulled]
    # Create a dataframe of just the main value we're predicting from
    mainValueFrame = pd.DataFrame({predictColumn:normalizedCulled[predictColumn]})
    # Create rolling frame off of the main value frame
    mainValueRolling = mainValueFrame.rolling(timeWindowSize,win_type=None,)
    # example of adding a metric (std dev) to the augmented frame
    # std = mainValueRolling.std()
    # std = std.rename(columns={predictColumn:'StdDev'})
    # augmentedFrameSubframes.append(std)
    # sma
    sma = mainValueRolling.mean()
    sma = sma.rename(columns={predictColumn:'SimpleMovingAverage1Hr'})
    augmentedFrameSubframes.append(sma)
    # stddev
    rstd = mainValueRolling.std()
    rstd = rstd.rename(columns={predictColumn:'StdDev1Hr'})
    augmentedFrameSubframes.append(rstd)
    # ema
    ema = mainValueFrame.ewm(span=timeWindowSize).mean()
    ema = ema.rename(columns={predictColumn:'ExponentialMovingAverage1Hr'})
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
    # variance
    var = mainValueRolling.var()
    var = var.rename(columns={predictColumn:'Variance1Hr'})
    augmentedFrameSubframes.append(var)
    # Compile the augmented frame
    # That big ol' list of frames we were compiling up to this point is what this works with
    # It unions the list into a single really big dataframe
    augmentedFrame = pd.concat(
        augmentedFrameSubframes,
        axis=1,
        join='inner',
    )
    # next for our original dataframe we want to cut off the beginning by size timeWindowSize
    # this is because for the timeWindowSize elements the rolling window was undefined and all calculations are forfeit
    timeAdjustedResults = augmentedFrame[timeWindowSize:]
    return timeAdjustedResults

# transform dataframe to neural network compatable array
def transformDataframeToModelInput(inputFrame: pd.DataFrame) -> np.ndarray:
    # convert frame to numpy array
    numpied = inputFrame.to_numpy()
    if numpied.shape != (None,60,11):
        numpied = np.expand_dims(numpied,axis=0)
    return numpied

# translate return array to frame
def zipResultsToDataframe(prediction: List[float], timestamps: List[float]) -> pd.DataFrame:
    returnFrame = pd.DataFrame(list(zip(timestamps,prediction)),columns=['timestamp','value'])
    return returnFrame

# inverse transform results dataframe
def inverseTransformResultList(inVals : List[float],startValue: float) -> List[float]:
    rVal = []
    currentValue = startValue
    for value in inVals:
        # un center
        unCentered = value - 0.5
        # un normalize
        unNormalized = unCentered * normalizationConstant
        # un differentiate
        currentValue = currentValue + unNormalized
        rVal.append(currentValue)
    return rVal

# generate timestamp array
def generateTimestampArray(startTime: int) -> List[int]:
    rVal = []
    currentTime : int = startTime
    for i in range(0,60):
        currentTime = currentTime + 60 * 1000
        rVal.append(currentTime)
    return rVal
    



#
# Webapp
#

app = Flask(__name__)


@app.route("/")
def helloWorld():
    return 'Data Manipulator!'

@app.route("/eval", methods=["POST"])
def eval():
    # read in data from request
    requestRaw = request.data
    # format data as string
    requestString = requestRaw.decode("utf-8")
    # deserialize striing to list
    inputList = json.loads(requestString)
    # parse list to panda dataframe
    rawInputDataframe = pd.DataFrame(inputList[1:],columns=inputList[0])
    # get the timestamps from the data
    historicalTimeSeries = extractTimestampSeries(rawInputDataframe)
    # get predictcolumn from data
    historicalPredictValues = extractTargetColumn(rawInputDataframe)
    # drop unnescessary columns
    culledInFrame = dropUnnescessaryColumns(rawInputDataframe)
    # fudge missing columns
    augmentedInFrame = augmentMissingColumns(culledInFrame)
    # transform data to NN compatable input
    transformedData = transformRawColumns(augmentedInFrame)
    print(transformedData)
    # transform to numpy array
    NNinput = transformDataframeToModelInput(transformedData)
    # run NN
    result = model.predict(NNinput)
    # get result array
    result = result[0]
    # remove elements from inner array
    result = list(map(lambda x: x[0], result))
    print(result)
    # get last value of target data
    currentValue: float = historicalPredictValues.iloc[-1]
    # inverse transform results
    print(currentValue)
    transformedResults = inverseTransformResultList(result,currentValue)
    # get last timestamp of input
    currentTime: int = historicalTimeSeries.iloc[-1] + 60 * 1000
    # generate timestamps array
    resultTimestamps = generateTimestampArray(currentTime)
    # translate to dataframe
    resultDataframe: pd.DataFrame = zipResultsToDataframe(transformedResults,resultTimestamps)
    # print(resultDataframe)
    # convert final dataframe to list
    finalList = [resultDataframe.columns.values.tolist()] + resultDataframe.values.tolist()
    # parse list to json string
    rVal = json.dumps(finalList) + "\n"
    # return json
    return rVal
    # inverse transform data
    # listResult = np.ndarray.tolist(result)[0]
    # parse inverse transform to json array
    # rVal = json.dumps(listResult) + "\n"
    #return json array
    # return rVal
    # return "[1,2,3]"

