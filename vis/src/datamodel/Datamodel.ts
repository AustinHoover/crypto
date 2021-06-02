
//An individual point on a line graph
export interface LineSeriesDatapoint {
    value : number,
    timestamp : number,
}

//a single line on a graph
export interface LineSeriesModel {
    timestampStart : number,
    timestampEnd : number,
    rangeStart : number,
    rangeEnd : number,
    data : LineSeriesDatapoint[],
    associatedQuery : string,
    queryAppendTimestampParams : boolean,
    realtimeSeries : boolean,
    updateInterval : number,
    color : string,
}

//The collection of all models of a given graph svg
//"Corresponds" to VisCard and VisGraph
export interface GraphModel {
    title : string,
    timestampStart : number,
    timestampEnd : number,
    rangeStart : number,
    rangeEnd : number,
    lineSeries : LineSeriesModel[],
    numValidSeries : number,
}

//A container for all SVGs we plan to draw on the page
//"Corresponds" to Viewer
export interface GraphVisModel {
    models : GraphModel[],
    timestampDomainStart : number,
    timestampDomainEnd : number,
    controls : ControlsModel,
}

export interface ControlsModel {
    autoIncrement : boolean,
}

export interface AppModel {
    graphs : GraphVisModel,
}

let mockEnd = Date.now();
let mockStart = mockEnd - 1000 * 60 * 60 * 12;

let secondMockEnd = Date.now();
let secondMockStart = secondMockEnd - 1000 * 60 * 60 * 1;

let initialMockSeries : LineSeriesModel= {
    timestampStart : mockStart,
    timestampEnd : mockEnd,
    rangeStart : 0,
    rangeEnd : 1000,
    data : null,
    color : "black",
    realtimeSeries : false,
    updateInterval : 60 * 1000,
    associatedQuery : "http://localhost:8080/rawdata",
    queryAppendTimestampParams : true,
};

let secondMockSeries : LineSeriesModel = {
    timestampStart : secondMockStart,
    timestampEnd : secondMockEnd,
    rangeStart : 0,
    rangeEnd : 1000,
    data : null,
    color : "black",
    realtimeSeries : true,
    updateInterval : 60 * 1000,
    associatedQuery : "http://localhost:8080/rawdata",
    queryAppendTimestampParams : true,
}

let unchangingMockSeries : LineSeriesModel = {
    timestampStart : secondMockStart,
    timestampEnd : secondMockEnd + 1000 * 60 * 60,
    rangeStart : 0,
    rangeEnd : 1000,
    data : null,
    color : "red",
    realtimeSeries : false,
    updateInterval : 0,
    associatedQuery : "http://localhost:8080/predict/hour",
    queryAppendTimestampParams : true,
}

let initialMockGraphState : GraphModel = {
    title: "Bitcoin last 12 hours",
    timestampStart : mockStart,
    timestampEnd : mockEnd,
    rangeStart : 100,
    rangeEnd : 1000,
    lineSeries : [initialMockSeries],
    numValidSeries : 0,
}

let secondMockGraphState : GraphModel = {
    title: "Future prediction",
    timestampStart : mockStart,
    timestampEnd : secondMockEnd * 2,
    rangeStart : 100,
    rangeEnd : 1000,
    lineSeries : [unchangingMockSeries, secondMockSeries],
    numValidSeries : 0,
}

let initialControlsState : ControlsModel = {
    autoIncrement : false,
};

let initialGraphVisState : GraphVisModel = {
    models : [initialMockGraphState, secondMockGraphState],
    timestampDomainStart : 0,
    timestampDomainEnd : 100,
    controls : initialControlsState,
}

export let initialState : AppModel = {
    graphs : initialGraphVisState,
};