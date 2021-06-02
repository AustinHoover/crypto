import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# rawFile = open('./datasets/test.csv')
# reader = csv.reader(rawFile)
inFrame = pd.read_csv('../datasets/noNaN.csv')

normalizationConstant = 100000

normalizedFrame = inFrame.divide(normalizationConstant)

maxWindowSize = 10 #should be set to the size of the largest rolling window

normalizedCloseSeries = normalizedFrame["Close"]
shiftedNormalizedCloseSeries = normalizedCloseSeries.shift(periods=maxWindowSize)
justCloseFrame = pd.DataFrame({'Close':normalizedFrame["Close"]})
shiftedCloseFrame = pd.DataFrame({'Close':shiftedNormalizedCloseSeries})
rolling = normalizedFrame.rolling(maxWindowSize,win_type=None,)
closeRolling = justCloseFrame.rolling(maxWindowSize,win_type=None,)
#https://towardsdatascience.com/trading-technical-analysis-with-pandas-43e737a17861
#holy shit this is useful
#sma
sma = closeRolling.mean()
sma = sma.rename(columns={'Close':'SimpleMovingAverage1Hr'})
#stddev
rstd = closeRolling.std()
rstd = rstd.rename(columns={'Close':'StdDev1Hr'})
#ema
ema = justCloseFrame.ewm(span=maxWindowSize).mean()
ema = ema.rename(columns={'Close':'ExponentialMovingAverage1Hr'})
#bollinger upper
upper_band = sma['SimpleMovingAverage1Hr'] + 2 * rstd['StdDev1Hr']
upper_band = pd.DataFrame({'BollingerUpper1Hr':upper_band})
#bollinger lower
lower_band = sma['SimpleMovingAverage1Hr'] - 2 * rstd['StdDev1Hr']
lower_band = pd.DataFrame({'BollingerLower1Hr':lower_band})


print(len(ema))
print(len(sma))
print(len(rstd))
print(len(upper_band))
print(len(lower_band))
print(len(inFrame))

finalFrame = pd.concat([normalizedFrame,sma,rstd,ema,upper_band,lower_band],axis=1,join='inner')
# finalFrame = finalFrame.append(sma)
# finalFrame = finalFrame.append(rstd)
# finalFrame = finalFrame.append(upper_band)
# finalFrame = finalFrame.append(lower_band)


print("dataset desc")
print(finalFrame)
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
#https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
#https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy   <----- SECOND answer
#https://stackoverflow.com/questions/47482009/pandas-rolling-window-to-return-an-array
#https://pandas.pydata.org/pandas-docs/stable/reference/window.html
#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/timeseries_dataset_from_array
minStartIndex = maxWindowSize
#read in train data
print(finalFrame)
print(shiftedCloseFrame)
dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    finalFrame,
    shiftedCloseFrame,
    sequence_length=maxWindowSize,
    sequence_stride=1,
    start_index=minStartIndex,
    end_index=100
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

print("datasets")
print("train")
datasetIter = iter(dataset)
# for row in datasetIter:
    # print(row[0])

datasetIter = iter(dataset)
print("test")
# for row in datasetIter:
    # print(row[1])

firstTrain = datasetFirst[0]
firstTest = datasetFirst[1]
print("shapes: ",firstTrain.shape," ",firstTest.shape)
# print("vs")
# print(testShape," ",len(testWindowGenerator))

expectedInShape = (firstTrain.shape[1:])
expectedOutShape = (firstTest.shape[1:])

print("Model translates: ",expectedInShape,"=>",expectedOutShape)


model = keras.Sequential()
model.add(keras.Input(shape=expectedInShape))
model.add(layers.Flatten())
# model.add(layers.LSTM(50,return_sequences=True))
# model.add(layers.LSTM(50,return_sequences=True))
# model.add(layers.LSTM(50,return_sequences=True))
# model.add(layers.SimpleRNN(100))
denseLayer = layers.Dense(maxWindowSize,activation="softmax")
model.add(denseLayer)
# model.add(layers.Dense(maxWindowSize,activation="softmax"))
model.add(layers.Dense(1))
# model.add(layers.Reshape(expectedOutShape))
#     [
#         keras.Input(shape=expectedInShape),
#         # layers.Flatten(),
#         # layers.Dropout(0.5),
#         layers.LSTM(50,return_sequences=True),
#         layers.LSTM(50,return_sequences=True),
#         layers.LSTM(50,return_sequences=True),
#         layers.SimpleRNN(100),
#         layers.Dense(maxWindowSize,activation="softmax"),
#         layers.Reshape(expectedOutShape),
#     ]
# )

model.summary()

epochs = 5000

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    dataset,
    epochs=epochs
)

print(denseLayer.get_weights())