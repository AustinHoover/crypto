import { AppModel } from "./datamodel"

export enum ActionType {
    GRAPH_CONTROL_TOGGLE_AUTO_INCREMENT,
    GRAPH_UPDATE_SERIES_DATA,
    GRAPH_SERIES_INCREASE_TIMESCALE,
}

export const actionGraphControlToggleAutoIncrement = (position : number) => {
    return {
        position : position,
        type : ActionType.GRAPH_CONTROL_TOGGLE_AUTO_INCREMENT,
    }
}

export const actionGraphUpdateSeriesData = (graph : number, series : number, data : any[]) => {
    return {
        graph : graph,
        series : series,
        data : data,
        type : ActionType.GRAPH_UPDATE_SERIES_DATA,
    }
}

export const actionGraphSeriesIncreaseTimescale = (graph : number, series : number, increment : number) => {
    return {
        graph : graph,
        series : series,
        increment : increment,
        type : ActionType.GRAPH_SERIES_INCREASE_TIMESCALE,
    }
}