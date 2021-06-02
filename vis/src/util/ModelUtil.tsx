import { GraphModel } from "../datamodel/datamodel";



export function calculateNewBounds(model : GraphModel) : GraphModel{
    let rVal : GraphModel = {...model};
    let domainStart = -1;
    let domainEnd = -1;
    let setDomain = false;
    let rangeStart = -1;
    let rangeEnd = -1;
    let setRange = false;
    rVal.lineSeries.forEach((el)=>{
        if(!setDomain){
            setDomain = true;
            domainStart = el.timestampStart;
            domainEnd = el.timestampEnd;
        } else {
            if(el.timestampStart < domainStart){
                domainStart = el.timestampStart;
            }
            if(el.timestampEnd > domainEnd){
                domainEnd = el.timestampEnd;
            }
        }
        if(!setRange){
            setRange = true;
            rangeStart = el.rangeStart;
            rangeEnd = el.rangeEnd;
        } else {
            if(el.rangeStart < rangeStart){
                rangeStart = el.rangeStart
            }
            if(el.rangeEnd > rangeEnd){
                rangeEnd = el.rangeEnd;
            }
        }
    });
    rVal.timestampStart = domainStart;
    rVal.timestampEnd = domainEnd;
    rVal.rangeStart = rangeStart;
    rVal.rangeEnd = rangeEnd;
    return rVal;
}