import * as React from "react";
import "react-dom";


export interface VisControlInterface {
    datasetStart : number,
    timeStart : number,
    timeCurrent : number,
    setTimeStart,
}


const VisControl = (props : VisControlInterface) => {

    let datasetStart = props.datasetStart;
    let timeStart = props.timeStart;
    let timeCurrent = props.timeCurrent;
    let setTimeStart = props.setTimeStart;
    let oneHourMilliseconds = 3600000;

    let [newTimeVal,setNewTimeVal] = React.useState(timeStart);

    let rangeModifyCallback = (e) => {
        setNewTimeVal(+e.currentTarget.value);
    }

    let confirmRangeChangeCallback = () => {
        setTimeStart(newTimeVal);
    }

    let confirmRangePastHour = () => {
        let currentTimestamp = + new Date();
        currentTimestamp = currentTimestamp;
        let newTime = currentTimestamp - oneHourMilliseconds;
        setNewTimeVal(newTime);
        setTimeStart(newTime);
    }

    React.useEffect(() => {
        setTimeout(
            () => {
                confirmRangePastHour()
            },
            60000
        );
    }, [timeStart])

    return(
        <div className="controlsPanel">
            <input 
            min={datasetStart}
            max={timeCurrent}
            value={newTimeVal}
            onChange={rangeModifyCallback}
            type="range"
            />
            <input
            value="Confirm"
            onClick={confirmRangeChangeCallback}
            type="button"
            />
            <input
            value="Past Hour"
            onClick={confirmRangePastHour}
            type="button"
            />
            <div>{newTimeVal}</div>
        </div>
    );
}

export default VisControl;