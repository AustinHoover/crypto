package main.series.datapoint;

import java.util.LinkedList;
import java.util.List;

/**
 *
 * @author Austin Hoover
 */
public class Datapoint {
    long timestamp;
    List<Double> values;
    
    public Datapoint(long timestamp, double... values){
        this.timestamp = timestamp;
        this.values = new LinkedList();
        for(double currentVal : values){
            this.values.add(currentVal);
        }
    }
    
    public long getTimestamp(){
        return timestamp;
    }
    
    public List<Double> getValues(){
        return values;
    }
    
}
