

export async function getPredictiveModel(inputData){
    let predictedData = await fetch(
        "http://localhost:500/eval",
        {
            method: 'POST',
            body: JSON.stringify(inputData)
        }
    ).then(response => response.json());
    return predictedData;
}

export async function mapModelToTimestamp(inputData, timeStart, timeEnd){
    let dataLength = inputData.length;
    let timeLength = timeEnd - timeStart;
    let increment = timeLength / dataLength;
    let i = 0;
    let rVal = inputData.map(
        input => {
            i++;
            return {
                value : input,
                timestamp : (timeStart + (i * increment))
            }
        }
    );
    return rVal;
}