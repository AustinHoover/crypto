# crypto

# What is this repo?
This is a semi-active project I'm using to explore approaches to prediction services for cryptocurrency valuation.




# Does it work?
### No
predicting markets is hard :)




# What all is here?
The project is a monorepo with several major projects

## __BitcoinDataminer__
Passively collects realtime bitcoin valuation data from Binance's API. Written in Java, uses chron to run.

## __DataInterconnect__
Provides a common platform for all other projects within this repo to access data collected from the BitcoinDataminer. Written in Java using Spring Boot.

## __analysis__
Container folder for all datascience related programs used to analyze/predict cryptocurrency pricing. Written in Python using Tensorflow and Pandas.

## __modelserver__
Provides a restful service that exposes a trained neural network to predict data ; used by "vis" to display predicted future values. Written in Python using Flask, Tensorflow, and Python.

## __vis__
The visualizer for the modelserver results ; provides a d3 based web view of realtime data. Written in Typescript using React, Hooks, D3, Bootstrap, and parcelJS.




# Why is there no git history?
This was original developed on a private gitea instance.




# Technologies Used

### Languages
Typescript, Python & Java

### Frameworks / Libraries
React, D3, Bootstrap, parcelJS, Tensorflow, Flask, Pandas, & SpringBoot




# Todo
 - Try more sophisticated models for predicting pricing
 - Secure DataInterconnect such that it can be exposed to the internet
 - Visualizer needs organizational work
 - Construct platform for acting on trading strategy
 - Better documentation
 - Code cleanup