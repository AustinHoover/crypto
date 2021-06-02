package main;

import main.series.datapoint.Ticker;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import main.series.Dataset;
import main.series.TickerSeries;
import main.series.datapoint.Datapoint;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

/**
 *
 * @author Austin Hoover
 */



@RestController
public class DataRetrievalController {
    
    @CrossOrigin(origins = {"http://localhost:8080","http://localhost:1234"})
    @RequestMapping("/rawdata")
    @ResponseBody
    public String rawData(
            @RequestParam String timestampStart,
            @RequestParam String timestampEnd
    ){
        String rVal = "\"timestamp\",\"open\",\"high\",\"low\",\"value\",\"volume\",\n";
        long startTime = Long.parseLong(timestampStart);
        long endTime = Long.parseLong(timestampEnd);
        TickerSeries series = DataRetrievalWorker.fetchData(startTime, endTime);
        Iterator<Ticker> tickerIterator = series.getData().iterator();
        while(tickerIterator.hasNext()){
            Ticker currentTicker = tickerIterator.next();
            rVal = rVal +
                    currentTicker.getTimestamp() + "," +
                    currentTicker.getOpen() + "," +
                    currentTicker.getHigh() + "," +
                    currentTicker.getLow() + "," +
                    currentTicker.getClose() + "," +
                    currentTicker.getVolume() + "," + 
                    "\n"
                    ;
        }
        return rVal;
    }
    
    @CrossOrigin(origins = {"http://localhost:8080","http://localhost:1234"})
    @RequestMapping("/predict/hour")
    @ResponseBody
    public String predict(){
        String rVal = "";
        long endTime = System.currentTimeMillis();
        long startTime = endTime - 2 * 60 * 60 * 1000;
        TickerSeries historicalTickers = DataRetrievalWorker.fetchData(startTime, endTime);
        Dataset historicalDataset = Dataset.datasetFromTickerSeries(historicalTickers);
//        System.out.println(Dataset.serializeDatasetToList(historicalDataset));
        Dataset predictionSet = DataRetrievalWorker.getPrediction(historicalDataset);
        String predictionCSV = Dataset.serializeDatasetToCSV(predictionSet);
        rVal = predictionCSV;
        
        
//        List<Datapoint> data = DataRetrievalWorker.getPredictionHour(startTime);
//        rVal = "\"timestamp\",";
//        for(int i = 0; i < data.get(0).getValues().size(); i++){
//            rVal = rVal + "\"value" + (i+1) + "\","; 
//        }
//        rVal = rVal + "\n";
//        for(Datapoint datapoint : data){
//            rVal = rVal + datapoint.getTimestamp() + ",";
//            for(double value : datapoint.getValues()){
//                rVal = rVal + value + ",";
//            }
//            rVal = rVal + "\n";
//        }
        return rVal;
    }
    
    
    
    @CrossOrigin(origins = {"http://localhost:8080","http://localhost:1234"})
    @RequestMapping("/mock")
    @ResponseBody
    public String mockData(
            @RequestParam String timestampStart,
            @RequestParam String timestampEnd
    ){
        String rVal = "\"timestamp\",\"open\",\"high\",\"low\",\"value\",\"volume\",\n";
        long startTime = Long.parseLong(timestampStart);
        long endTime = Long.parseLong(timestampEnd);
        List<Ticker> tickerList = DataRetrievalWorker.generateData(startTime, endTime);
        Iterator<Ticker> tickerIterator = tickerList.iterator();
        while(tickerIterator.hasNext()){
            Ticker currentTicker = tickerIterator.next();
            rVal = rVal +
                    currentTicker.getTimestamp() + "," +
                    currentTicker.getOpen() + "," +
                    currentTicker.getHigh() + "," +
                    currentTicker.getLow() + "," +
                    currentTicker.getClose() + "," +
                    currentTicker.getVolume() + "," + 
                    "\n"
                    ;
        }
        return rVal;
    }
    
    
    @CrossOrigin(origins = {"http://localhost:8080","http://localhost:1234"})
    @RequestMapping("/predict/hour/noreply")
    @ResponseBody
    public String testModelNoReply(){
        String rVal = "It works!";
        long endTime = System.currentTimeMillis();
        long startTime = endTime - 2 * 60 * 60 * 1000;
        TickerSeries historicalTickers = DataRetrievalWorker.fetchData(startTime, endTime);
        Dataset historicalDataset = Dataset.datasetFromTickerSeries(historicalTickers);
        DataRetrievalWorker.runPredictionNoReply(historicalDataset);
        return rVal;
    }
    
    
    @CrossOrigin(origins = "http://localhost:8080")
    @RequestMapping("/shutdown")
    @ResponseBody
    public String shutdown(){
        String rVal = "exit";
        Thread shutdownThread = new Thread(){
            @Override
            public void run(){
                try {
                    TimeUnit.MILLISECONDS.sleep(5000);
                } catch (InterruptedException ex) {
                }
                System.exit(0);
            }
        };
        shutdownThread.start();
        return rVal;
    }
}
