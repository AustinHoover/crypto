package main;

import java.util.Date;

/**
 *
 * @author Austin Hoover
 */
public class Ticker {
    long timestamp;
    double open;
    double high;
    double low;
    double close;
    double volume;

    public long getTimestamp() {
        return timestamp;
    }

    public double getOpen() {
        return open;
    }

    public double getHigh() {
        return high;
    }

    public double getLow() {
        return low;
    }

    public double getClose() {
        return close;
    }

    public double getVolume() {
        return volume;
    }
    
    
    
    public Ticker(long timestamp, double open, double high, double low, double close, double volume){
        this.timestamp = timestamp;
        this.open = open;
        this.high = high;
        this.low = low;
        this.close = close;
        this.volume = volume;
    }
    
    @Override
    public String toString(){
        String rVal = "";
        Date date = new Date(timestamp);
        rVal = date + " " + open + " " + high + " " + low + " " + close + " " + volume;
        return rVal;
    }
}
