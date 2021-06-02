import csv
import math
import time


#build data
# Model / data parameters

#Technical Indicators
#https://www.investopedia.com/top-7-technical-analysis-tools-4773275
#https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/list-of-technical-indicators/
#https://en.wikipedia.org/wiki/Moving_horizon_estimation <--- "See Also"
#https://en.wikipedia.org/wiki/Particle_filter#See_also <--- "Other filters" & "See Also"
#https://en.wikipedia.org/wiki/Kalman_filter <--- "See Also"
#https://www.investopedia.com/terms/i/ichimokuchart.asp
numMetrics = 0

#SMA
#EMA
#https://www.investopedia.com/terms/m/movingaverage.asp
numMetrics = numMetrics + 2

#BOLU via SMA @ n=20
#BOLD via SMA @ n=20
#BOLU via EMA @ n=20
#BOLD via EMA @ n=20
#https://www.investopedia.com/terms/b/bollingerbands.asp
numMetrics = numMetrics + 4

#MACD
#https://www.investopedia.com/terms/m/macd.asp
#https://www.investopedia.com/trading/macd/
numMetrics = numMetrics + 1

#RSI
#https://www.investopedia.com/terms/r/rsi.asp
numMetrics = numMetrics + 1

#Stochastic Oscillator @ L14
#https://www.investopedia.com/terms/s/stochasticoscillator.asp
numMetrics = numMetrics + 1

#ROC @ p=current & n=current-10
#https://www.investopedia.com/terms/p/pricerateofchange.asp
numMetrics = numMetrics + 1

#MFI
#https://www.investopedia.com/terms/m/mfi.asp
numMetrics = numMetrics + 1

#Kalman Filter? :)
#https://en.wikipedia.org/wiki/Kalman_filter
numMetrics = numMetrics + 1

#Particle Filter :))))
#https://en.wikipedia.org/wiki/Particle_filter
numMetrics = numMetrics + 1



valueNormalizationMagnitude = 100000
volumeNormalizationMagnitude = 100


#For each row we want
#Open
#High
#Low
#Close
#Volume (BTC)
#Volume (Currency)
#Weighted Price
numPerTimestampValues = 7 #This is the number of per-timestamp things to include
rowItems = numPerTimestampValues + numMetrics
incrementsPerHour = 60
hoursToTrain = 4
numRawDatapoints = incrementsPerHour*hoursToTrain #This is how 'far back' we're going to 'look' 60 mins/hr * 24 hrs/day

datapointFrameWidth = numRawDatapoints * numPerTimestampValues #This is the width of the frame of datapoints per input

rawData = open('./datasets/noNaN.csv')
parsedData = csv.reader(rawData)

fullData = open('./datasets/fulldata.csv','w',newline='')
dataWriter = csv.writer(fullData)


#print("Counting number raw entries")
numRows = 1494677 #uncomment to recalculate sum(1 for row in parsedData) - 1 #-1 for header
#print(numRows)


parsingStart = numRawDatapoints + 1 # Beginning point to start parsing
parsingEnd = numRows # Ending point to stop parsing
print("Beginning analysis")
# Frames
currentFrame = []
frameSize = numRawDatapoints
# basic
timestampFrame = []
exOpenFrame = []
exHighFrame = []
exLowFrame = []
exCloseFrame = []
btcVolFrame = []
usdVolFrame = []
weightedPriceFrame = []
# basic calc
SMAFrame = []
EMAFrame = []
TPFrame = []
SMASum = 0
EMASmoothingFactor = 2 / (frameSize + 1)

#seek and whatnot
rawData.seek(1)
next(parsedData)

for i in range(0,parsingStart):
    currentRow = next(parsedData)
    #basic frames
    #Open
    #High
    #Low
    #Close
    #Volume (BTC)
    #Volume (Currency)
    #Weighted Price
    timestampFrame.append(float(currentRow[0]))
    exOpenFrame.append(float(currentRow[1]))
    exHighFrame.append(float(currentRow[2]))
    exLowFrame.append(float(currentRow[3]))
    exCloseFrame.append(float(currentRow[4]))
    btcVolFrame.append(float(currentRow[5]))
    usdVolFrame.append(float(currentRow[6]))
    weightedPriceFrame.append(float(currentRow[7]))
    #SMA Calc
    SMASum = SMASum + (float(currentRow[1]) + float(currentRow[2]) + float(currentRow[3]) + float(currentRow[4])) / 4 # current close
    SMAFrame.append(SMASum / (i + 1))
    #EMA Frame
    if i == 0:
        EMAy = exCloseFrame[0] * EMASmoothingFactor
        EMAFrame.append(EMAy)
    if i > 0:
        EMAt = exCloseFrame[i] * (EMASmoothingFactor) + EMAy * (1-EMASmoothingFactor)
        EMAy = EMAt
        EMAFrame.append(EMAt)
    #TP Frame
    TPFrame.append((exHighFrame[i] + exLowFrame[i] + exCloseFrame[i])/3)

dataWriter.writerow([
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "vol(btc)",
    "vol(usd)",
    "weighted price",
    "SMA",
    "TP",
    "EMA(today)",
    "SMA_TP",
    "TPStdDev",
    "BOLU",
    "BOLD",
    "MACD"
])
for i in range(parsingStart,parsingEnd):
    if (i % 1000) == 0:
        print("\r",(i/parsingEnd),end='')
    frameStart = i-numRawDatapoints
    frameEnd = i
    currentRow = next(parsedData)
#    currentFrame.pop(0)
#    currentFrame.append(currentRow)
    currentIndex = frameSize - 1
    #append individual values
    timestampFrame.pop(0)
    timestampFrame.append(float(currentRow[0]))
    #alert to timestamp skip
    #if (timestampFrame[currentIndex-1] - timestampFrame[currentIndex - 2]) > 61:
    #    print("Alert! Timestamp mismatch: ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestampFrame[currentIndex-1]))," of size ",(timestampFrame[currentIndex-1] - timestampFrame[currentIndex - 2]),"!\n")
    exOpenFrame.pop(0)
    exOpenFrame.append(float(currentRow[1]))
    exHighFrame.pop(0)
    exHighFrame.append(float(currentRow[2]))
    exLowFrame.pop(0)
    exLowFrame.append(float(currentRow[3]))
    exCloseFrame.pop(0)
    exCloseFrame.append(float(currentRow[4]))
    btcVolFrame.pop(0)
    btcVolFrame.append(float(currentRow[5]))
    usdVolFrame.pop(0)
    usdVolFrame.append(float(currentRow[6]))
    weightedPriceFrame.pop(0)
    weightedPriceFrame.append(float(currentRow[7]))
    #Simple Moving Average
    SMA = (sum(exOpenFrame) + sum(exHighFrame) + sum(exLowFrame) + sum(exCloseFrame)) / 4 / frameSize
    SMAFrame.pop(0)
    SMAFrame.append(SMA)
    #TP
    TP = (exHighFrame[currentIndex] + exLowFrame[currentIndex] + exCloseFrame[currentIndex]) / 3
    TPFrame.pop(0)
    TPFrame.append(TP)
    #Exponential Moving Average
    EMAt = exCloseFrame[currentIndex] * (EMASmoothingFactor) + EMAFrame[currentIndex-1] * (1-EMASmoothingFactor)
    EMAFrame.pop(0)
    EMAFrame.append(EMAt)
    #BollingerBands
    numStdDev = 2
    SMA_TP = sum(TPFrame) / frameSize
    TPVarianceSum = 0
    for j in range(0,frameSize-1):
        TPVarianceSum = TPVarianceSum + (TPFrame[j] - SMA_TP) * (TPFrame[j] - SMA_TP)
    TPvariance = TPVarianceSum / frameSize
    TPStdDev = math.sqrt(TPvariance)
    BOLU = SMA_TP + numStdDev * TPStdDev
    BOLD = SMA_TP - numStdDev * TPStdDev
    #MACD
    MACD = EMAFrame[int(currentIndex - 12)] - EMAFrame[int(currentIndex - 26)]
    #RSI
    #write out
    dataWriter.writerow([
        timestampFrame[currentIndex],
        exOpenFrame[currentIndex],
        exHighFrame[currentIndex],
        exLowFrame[currentIndex],
        exCloseFrame[currentIndex],
        btcVolFrame[currentIndex],
        usdVolFrame[currentIndex],
        weightedPriceFrame[currentIndex],
        SMA,
        TP,
        EMAt,
        SMA_TP,
        TPStdDev,
        BOLU,
        BOLD,
        MACD
    ])

rawData.close()
fullData.close()
print()
print("Done!")