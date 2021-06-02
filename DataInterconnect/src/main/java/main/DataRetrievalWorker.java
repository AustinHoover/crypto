package main;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import main.series.datapoint.Ticker;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import main.series.Dataset;
import main.series.TickerSeries;
import main.series.datapoint.Datapoint;

/**
 *
 * @author Austin Hoover
 */
public class DataRetrievalWorker {
    
    static final String modelURL = "http://localhost:5000/eval";
    
    public static TickerSeries fetchData(long startTime, long endTime){
        List<Ticker> tickerList = new ArrayList();
        try {
            DatabaseManager dbMgr = new DatabaseManager();
            dbMgr.connectToDB();
            Statement statement = dbMgr.getConnect().createStatement();
            String fetchQuery = "SELECT `open`, high, low, `close`, volume, `timestamp` FROM crypto.btcRealtimeData WHERE `timestamp` > " + startTime + " AND `timestamp` < " + endTime + ";";
            System.out.println(fetchQuery);
            ResultSet rs = statement.executeQuery(fetchQuery);
            while(rs.next()){
                //indexed starting at 1 for some fucked up reason
                Ticker currentTicker = new Ticker(
                    rs.getLong(6),
                    rs.getDouble(1),
                    rs.getDouble(2),
                    rs.getDouble(3),
                    rs.getDouble(4),
                    rs.getDouble(5)
                );
                tickerList.add(currentTicker);
            }
            dbMgr.disconnectFromDB();
        } catch (SQLException ex) {
            ex.printStackTrace();
        }
        TickerSeries rVal = new TickerSeries(tickerList);
        return rVal;
    }
    
    
    public static Dataset getPrediction(Dataset inData){
        Dataset rVal = new Dataset();
        String inputList = Dataset.serializeDatasetToList(inData);
        //request prediction
        String address = modelURL;
        String response = Utilities.restfulPost(address, inputList);
        System.out.println(response);
        //parse response & construct returned data
        rVal = Dataset.deserializeFromList(response);
        return rVal;
    }
    
    
    public static List<Datapoint> getPredictionHour(long startTime){
        List<Datapoint> data = new LinkedList();
        long hourInMillis = 1000 * 60 * 60;
        long endTime = startTime + hourInMillis * 1;
        long predictionStart = endTime + 1000 * 60;
        TickerSeries rawData = fetchData(startTime,endTime);
        String outVal = "[";
        for(Ticker current : rawData.getData()){
            outVal = outVal + "[" + current.getOpen() + "," + current.getHigh() + "," + current.getLow() + "," + current.getClose() + "," + current.getVolume() + "," + current.getVolume() * current.getClose() + "," + current.getClose() + "],";
        }
        outVal = outVal.substring(0, outVal.length() - 1);
        outVal = outVal + "]";
        //request prediction
        String address = modelURL;
        String response = Utilities.restfulPost(address, outVal);
        //parse response & construct returned data
        long incrementingTime = predictionStart;
        JsonElement jsonEl = JsonParser.parseString(response);
        for(JsonElement currentElement : jsonEl.getAsJsonArray()){
            data.add(new Datapoint(
                    incrementingTime,
                    currentElement.getAsDouble()
            ));
            incrementingTime = incrementingTime + 60 * 1000;
        }
        return data;
    }
    
    public static void runPredictionNoReply(Dataset inData){
        String inputList = Dataset.serializeDatasetToList(inData);
        //request prediction
        String address = modelURL;
        String response = Utilities.restfulPost(address, inputList);
        System.out.println(response);
    }
    
    
    
    public static List<Ticker> generateData(long startTime, long endTime){
        List<Ticker> rVal = new LinkedList();
        long increment = 1000 * 60;//1000 milli/sec * 60sec/min
        long currentTime = startTime;
        Random rand = new Random();
        while(currentTime < endTime){
            Ticker newTicker = new Ticker(
                    currentTime,
                    rand.nextDouble() * 1000,
                    rand.nextDouble() * 1000,
                    rand.nextDouble() * 1000,
                    rand.nextDouble() * 1000,
                    rand.nextDouble()
            );
            rVal.add(newTicker);
            currentTime = currentTime + increment;
        }
        return rVal;
    }
    
}
