import * as React from 'react';
import { AppModel, GraphModel } from '../../datamodel/datamodel';
import VisGraph from '../VisGraph/VisGraph';


export interface VisContextProps {
    dispatch : React.Dispatch<any>,
    state : AppModel,
    model : GraphModel,
    position : number,
}

const VisContext = (props : VisContextProps) => {
    

    let graph : JSX.Element[] = [];
    let controls : JSX.Element[] = [];

    // graph.push(<VisGraph model={props.model} position={props.position} dispatch={props.dispatch}/>);

    // graph.push(
    // <VisGraph 
    //     datasetStart={datasetStart}
    //     timeStart={timeStart}
    //     timeCurrent={timeCurrent}
    // />
    // );
    // controls.push(
    // <VisControl
    //     datasetStart={datasetStart}
    //     timeStart={timeStart}
    //     timeCurrent={timeCurrent}
    //     setTimeStart={setTimeStart}
    // />
    // );


    return (
        <div className="card m-2 shadow">
            <h4 className="m-2 text-center">{props.model.title}</h4>
            <VisGraph model={props.model} position={props.position} dispatch={props.dispatch}/>
            {controls}
        </div>
    );
}

export default VisContext;