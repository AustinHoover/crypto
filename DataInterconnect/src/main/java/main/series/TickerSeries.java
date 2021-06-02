package main.series;

import java.util.List;
import main.series.datapoint.Ticker;

/**
 *
 * @author Austin Hoover
 */
public class TickerSeries {
    List<Ticker> data;
    
    public TickerSeries(List<Ticker> tickerList){
        data = tickerList;
    }
    
    public List<Ticker> getData(){
        return data;
    }
}
