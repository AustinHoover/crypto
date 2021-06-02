import 'regenerator-runtime/runtime'
import * as React from "react";
import "react-dom";
import * as d3 from "d3";
import { dispatch, json, xml } from 'd3';
import { getPredictiveModel, mapModelToTimestamp } from './Requests';
import { GraphModel } from '../../datamodel/datamodel';
import { actionGraphSeriesIncreaseTimescale, actionGraphUpdateSeriesData } from '../../datamodel/actions';


export interface VisGraphInterface {
    dispatch : React.Dispatch<any>,
    model : GraphModel,
    position : number,
}

const VisGraph = (props : VisGraphInterface) => {

    let graphMountRef = React.useRef();
    let model = props.model;
    let seriesArray = model.lineSeries;
    let timeStart = props.model.timestampStart;
    let timeEnd = props.model.timestampEnd;

    

    React.useEffect(()=>{
        /*
        //https://www.d3-graph-gallery.com/graph/line_basic.html
        */
        async function drawGraph() {
            // set the dimensions and margins of the graph
            var margin = {top: 10, right: 30, bottom: 30, left: 60},
            width = 670 - margin.left - margin.right,
            height = 300 - margin.top - margin.bottom;
            //the svg we're going to draw :)
            var svg = d3.select("#graphMountPoint" + props.position)
            .append("svg")
            // .attr("viewBox","0 0 " + width + " " + height)
            // .attr("preserveAspectRatio", "xMinYMin meet")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");
            svg.classed("img-fluid");
            svg.style("max-width: 100%; height: auto;");

            // Add X axis --> it is a date format
            var x = d3.scaleTime()
            .domain([model.timestampStart,model.timestampEnd])
            .range([ 0, width ]);
            svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x));

            // Add Y axis
            var y = d3.scaleLinear()
            .domain([
                model.rangeStart - 100,
                model.rangeEnd + 100
            ])
            .range([ height, 0 ]);
            svg.append("g")
            .call(d3.axisLeft(y));

            if(seriesArray !== null){
                seriesArray.forEach((series,i)=>{
                    if(series.data !== null){
                        console.log("data",series.data)
                        svg.append("path")
                            .datum(series.data)
                            .attr("fill","none")
                            .attr("stroke", series.color)
                            .attr("stroke-width", 1.5)
                            //@ts-ignore
                            .attr("d", d3.line()
                                //@ts-ignore
                                .x(d=>x(d.timestamp))
                                //@ts-ignore
                                .y(d=>y(d.value))
                            );
                        if(series.realtimeSeries){
                            setTimeout(()=>{
                                props.dispatch(actionGraphSeriesIncreaseTimescale(props.position,i,series.updateInterval));
                            },series.updateInterval);
                        }
                    } else {
                        let fetchUrl = series.associatedQuery;
                        if(series.queryAppendTimestampParams){
                            fetchUrl = fetchUrl + "?timestampStart=" + series.timestampStart + "&timestampEnd=" + series.timestampEnd;
                        }
                        d3.csv(fetchUrl,
                            (d)=>{
                                return {
                                    timestamp: +d.timestamp,
                                    value: +d.value,
                                }
                            })
                        .then((data)=>{
                            props.dispatch(actionGraphUpdateSeriesData(props.position,i,data));
                        }).catch((err)=>{
                            console.log("err loading data ",err);
                        })
                        ;
                    }
                });
            }
            // let fetchUrl = "http://localhost:8080/mock?timestampStart=" + timeStart + "&timestampEnd=" + timeEnd;
                    // Add the area
                    // svg.append("path")
                    // .datum(data)
                    // .attr("fill", "red")
                    // .attr("fill-opacity", .3)
                    // .attr("stroke", "none")
                    // //@ts-ignore
                    // .attr("d", d3.area()
                    //     //@ts-ignore
                    //     .x(d=>x(d.timestamp))
                    //     //@ts-ignore
                    //     .y0(d=>y(d.low))
                    //     //@ts-ignore
                    //     .y1(d=>y(d.high))
                    //     );

            
            // let sampleData = "[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]";
            
            // let predictedData = await getPredictiveModel(sampleData);
            
            // let intervalFutureNext = 0;
            // let intervalFutureFinal = 100;
            // let timestampedPredictedData = await mapModelToTimestamp(predictedData,intervalFutureNext,intervalFutureFinal);
            
            // await svg.append("path")
            // .datum(timestampedPredictedData)
            // .attr("fill","none")
            // .attr("stroke", "black")
            // .attr("stroke-width", 1.5)
            // //@ts-ignore
            // .attr("d", d3.line()
            //     //@ts-ignore
            //     .x(d=>x(d.timestamp))
            //     //@ts-ignore
            //     .y(d=>y(d.close))
            // )

            // svg.classed("img-fluid");
            // svg.style("max-width: 100%; height: auto;");

        }

        drawGraph();
    },[]);
    let mountPointID = "graphMountPoint" + props.position;
    return(
        <div className="dataVisPanel" key={timeStart}>
            <div id={mountPointID} ref={graphMountRef}></div>
        </div>
    );
}

export default VisGraph;