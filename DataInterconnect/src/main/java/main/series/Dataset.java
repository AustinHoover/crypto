package main.series;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.List;
import main.series.datapoint.Datapoint;
import main.series.datapoint.Ticker;

/**
 *
 * @author Austin Hoover
 */
public class Dataset {
    List<String> columns;
    List<Datapoint> data;
    
    public Dataset(){
        columns = new ArrayList();
        data = new ArrayList();
    }
    
    public static Dataset datasetFromTickerSeries(TickerSeries series){
        Dataset rVal = new Dataset();
        rVal.columns.add("timestamp");
        rVal.columns.add("open");
        rVal.columns.add("high");
        rVal.columns.add("low");
        rVal.columns.add("close");
        rVal.columns.add("volume");
        for(Ticker currentInput : series.getData()){
            Datapoint newDatapoint = new Datapoint(
                    currentInput.getTimestamp(),
                    currentInput.getOpen(),
                    currentInput.getHigh(),
                    currentInput.getLow(),
                    currentInput.getClose(),
                    currentInput.getVolume()
            );
            rVal.addDatapoint(newDatapoint);
        }
        return rVal;
    }
    
    public void addDatapoint(Datapoint d){
        data.add(d);
    }
    
    public List<Datapoint> getData(){
        return data;
    }
    
    public List<String> getColumns(){
        return columns;
    }
    
    public static Dataset deserializeFromList(String listString){
        Dataset rVal = new Dataset();
        //some stuff
        JsonElement parsed = JsonParser.parseString(listString);
        boolean parsedColumns = false;
        for(JsonElement currentRow : parsed.getAsJsonArray()){
            if(!parsedColumns){
                parsedColumns = true;
                for(JsonElement entry : currentRow.getAsJsonArray()){
                    rVal.getColumns().add(entry.getAsString());
                }
            } else {
                JsonArray currentRowArray = currentRow.getAsJsonArray();
                Datapoint currentDatapoint = new Datapoint(
                        currentRowArray.get(0).getAsLong(), // timestamp
                        currentRowArray.get(1).getAsDouble() // value
                );
                rVal.addDatapoint(currentDatapoint);
            }
        }
        return rVal;
    }
    
    public static String serializeDatasetToList(Dataset dataset){
        String rVal = "[[";
        for(String currentColumn : dataset.getColumns()){
            rVal = rVal + "\"" + currentColumn + "\",";
        }
        //chop off comma
        rVal = rVal.substring(0, rVal.length() - 1);
        rVal = rVal + "],";
        for(Datapoint data : dataset.getData()){
            rVal = rVal + "[" + data.getTimestamp() + ",";
            for(double currentVal : data.getValues()){
                rVal = rVal + currentVal + ",";
            }
            rVal = rVal.substring(0, rVal.length() - 1);
            rVal = rVal + "],";
        }
        rVal = rVal.substring(0, rVal.length() - 1);
        rVal = rVal + "]";
        return rVal;
    }
    
    public static String serializeDatasetToCSV(Dataset dataset){
        String rVal = "";
        for(String currentColumn : dataset.getColumns()){
            rVal = rVal + currentColumn + ",";
        }
        rVal = rVal.substring(0, rVal.length() - 1) + "\n";
        for(Datapoint data : dataset.getData()){
            rVal = rVal + data.getTimestamp() + ",";
            for(double value : data.getValues()){
                rVal = rVal + value + ",";
            }
            rVal = rVal.substring(0, rVal.length() - 1) + "\n";
        }
        return rVal;
    }
}
