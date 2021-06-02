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

maxWindowSize = 60 #should be set to the size of the largest rolling window


justCloseFrame = pd.DataFrame({'Close':normalizedFrame["Close"]})
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

finalFrame = pd.concat([inFrame,sma,rstd,ema,upper_band,lower_band],axis=1,join='inner')
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
trainWindowGenerator = tf.keras.preprocessing.timeseries_dataset_from_array(finalFrame,None,sequence_length=maxWindowSize,start_index=minStartIndex,end_index=len(finalFrame)-minStartIndex)
trainShape = next(trainWindowGenerator.as_numpy_iterator()).shape
#read in test data
testWindowGenerator = tf.keras.preprocessing.timeseries_dataset_from_array(justCloseFrame,None,sequence_length=maxWindowSize,start_index=minStartIndex*2)
testShape = next(testWindowGenerator.as_numpy_iterator()).shape
#combined
combinedSet = tf.data.Dataset.zip((trainWindowGenerator,testWindowGenerator))
print(combinedSet)
print(trainShape," ",len(trainWindowGenerator))
print("vs")
print(testShape," ",len(testWindowGenerator))

expectedInShape = (trainShape[1:])
expectedOutShape = (testShape[1:])

print("Model translates: ",expectedInShape,"=>",expectedOutShape)


model = keras.Sequential(
    [
        keras.Input(shape=expectedInShape),
        layers.Flatten(),
        # layers.Dropout(0.5),
        layers.Dense(maxWindowSize * 10,activation="softmax"),
        layers.Dense(maxWindowSize,activation="softmax"),
        layers.Reshape(expectedOutShape),
    ]
)

model.summary()

epochs = 500

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    combinedSet,
    batch_size=32,
    epochs=epochs
)
