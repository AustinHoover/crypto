import { calculateNewBounds } from "../util/ModelUtil";
import { ActionType } from "./actions";
import { AppModel, GraphModel, GraphVisModel, LineSeriesModel } from "./datamodel";


export function graphReducer(state : AppModel, action : any){
    if(action.type === ActionType.GRAPH_CONTROL_TOGGLE_AUTO_INCREMENT){
        
    } else if(action.type === ActionType.GRAPH_UPDATE_SERIES_DATA){
        let graph = action.graph;
        let series = action.series;
        let newSeriesRangeLower = -1;
        let newSeriesRangeUpper = -1;
        let setNewRange = false;
        console.log(action.data);
        action.data.forEach((el)=>{
            if(!setNewRange){
                setNewRange = true;
                newSeriesRangeLower = el.value;
                newSeriesRangeUpper = el.value;
            } else {
                if(el.value < newSeriesRangeLower){
                    newSeriesRangeLower = el.value;
                }
                if(el.value > newSeriesRangeUpper){
                    newSeriesRangeUpper = el.value;
                }
            }
        });
        // let oldSeries = state.graphs.models[graph].lineSeries[series];
        let newstate : AppModel = {...state, graphs : {...state.graphs,
            models : state.graphs.models.map((el,i)=>{
                if(i === action.graph){
                    let newModel = {...el,
                        lineSeries : el.lineSeries.map((ser,j)=>{
                            if(j === action.series){
                                return {...ser, 
                                    rangeStart : newSeriesRangeLower,
                                    rangeEnd : newSeriesRangeUpper,
                                    data : action.data
                                };
                            } else {
                                return ser;
                            }
                        }),
                        numValidSeries : el.numValidSeries + 1
                    }
                    newModel = calculateNewBounds(newModel);
                    return newModel;
                } else {
                    return el;
                }
            })
        }};
        return newstate;
        // newstate.graphs.models[graph].lineSeries[series].data = data;
    } else if(action.type === ActionType.GRAPH_SERIES_INCREASE_TIMESCALE){
        let newstate : AppModel = {...state, graphs : {...state.graphs,
            models : state.graphs.models.map((el,i)=>{
                if(i === action.graph){
                    let newModel = {...el,
                        lineSeries : el.lineSeries.map((ser,j)=>{
                            if(j === action.series){
                                return {
                                    ...ser,
                                    timestampEnd : ser.timestampEnd + 60 * 1000,
                                    data : null
                                };
                            } else {
                                return ser;
                            }
                        }),
                        numValidSeries : el.numValidSeries - 1
                    }
                    return newModel;
                } else {
                    return el;
                }
            })
        }};
        return newstate;
    } else {
        return state;
    }
}

