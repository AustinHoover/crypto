import 'regenerator-runtime/runtime'
import * as React from "react";
import "react-dom";
import * as d3 from "d3";
import { AppModel, initialState } from '../../datamodel/datamodel';
import { graphReducer } from '../../datamodel/reducers';
import VisContext from '../../components/VisCard/VisCard';
import { actionGraphUpdateSeriesData } from '../../datamodel/actions';

export const datasetStart = 1612588920000;

const Viewer = () => {
    // let timeCurrent = + new Date();
    // let oneHourMilliseconds = 3600000;
    // let [timeStart,setTimeStart] = React.useState(timeCurrent - 3600000);
    let [state,dispatch] = React.useReducer(graphReducer,initialState);

    let graphs : JSX.Element[] = [];
    
    state.graphs.models.forEach((el,i)=>{
        graphs.push(
            <div className="col" key={i + "c" + el.numValidSeries + " "}>
                <VisContext model={el} position={i} dispatch={dispatch} state={state}/>
            </div>
        );
        if(i % 2 == 1){
            graphs.push(<div className="w-100" key={i + "nl"}></div>);
        }
    });

    React.useEffect(()=>{
        async function checkSeries(){
            state.graphs.models.forEach((model, modelnum)=>{
                model.lineSeries.forEach((series, seriesnum)=>{
                    if(series.data === null){
                        // let fetchUrl = series.associatedQuery;
                        // d3.csv(fetchUrl,
                        //     (d)=>{
                        //         return {
                        //             timestamp: +d.timestamp,
                        //             open: +d.open,
                        //             high: +d.high,
                        //             low: +d.low,
                        //             close: +d.close,
                        //         }
                        //     })
                        // .then((data)=>{
                        //     dispatch(actionGraphUpdateSeriesData(modelnum,seriesnum,data));
                        // }).catch((err)=>{
                        //     console.log("err loading data ",err);
                        // })
                        // ;
                    }
                });
            });
        }
        checkSeries();
    },[]);

    return(
        <div className="container-xlg">
            <div className="row">
                {graphs}
            </div>
        </div>
    );
}

export default Viewer;