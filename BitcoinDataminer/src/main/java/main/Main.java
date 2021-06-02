package main;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.List;

/**
 *
 * @author Austin Hoover
 */
public class Main {
    public static void main(String args[]) throws IOException {
        String configFilePath = "./config.json";
        Gson gson = new Gson();
        Config appConfig = gson.fromJson(Files.newBufferedReader(new File(configFilePath).toPath()), Config.class);
        String requestComplete = appConfig.baseUrl + appConfig.requestPath + "?symbol=" + appConfig.symbol + "&interval=" + appConfig.interval + "&limit=" + appConfig.limit;
        System.out.println(requestComplete);
        String resultRaw = restfulParseString(requestComplete);
        resultRaw = "{results:" + resultRaw + "}";
        JsonArray upperLevelArray = JsonParser.parseString(resultRaw).getAsJsonObject().get("results").getAsJsonArray();
        ArrayList<Ticker> tickerList = new ArrayList();
        Iterator upperLevelArrayIterator = upperLevelArray.iterator();
        while(upperLevelArrayIterator.hasNext()){
            JsonArray current = (JsonArray)upperLevelArrayIterator.next();
            long timestamp = current.get(0).getAsLong();
            double open = current.get(1).getAsDouble();
            double high = current.get(2).getAsDouble();
            double low = current.get(3).getAsDouble();
            double close = current.get(4).getAsDouble();
            double volume = current.get(5).getAsDouble();
            Ticker newTicker = new Ticker(timestamp, open, high, low, close, volume);
            tickerList.add(newTicker);
            Date date = new Date(timestamp);
//            System.out.println(newTicker);
        }
        Ticker tickerToWriteOut = tickerList.get(tickerList.size() - 1);
        System.out.println(tickerToWriteOut.getTimestamp());
        System.out.println(tickerToWriteOut);
        int resultValue = writeToDB(tickerToWriteOut, appConfig);
        System.out.println("DB insert result status: " + resultValue);
    }
    public static String restfulParseString(String serverAddress){
        String rVal = "";
        try {
            URL url = new URL(serverAddress);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");
            conn.connect();
//            DataOutputStream out = new DataOutputStream(conn.getOutputStream());
//            out.writeBytes("{\"target\":\"" + input + "\"}");
//            out.flush();
//            out.close();
            BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            String currentLine = "";
            while((currentLine = reader.readLine()) != null){
                rVal = rVal + currentLine;
            }
            reader.close();
            conn.disconnect();
        } catch (MalformedURLException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        return rVal;
    }
    public static int writeToDB(Ticker ticker, Config config){
        int rVal = 0;
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
            Connection con = DriverManager.getConnection(
                    "jdbc:mysql://" + config.dbAddress + ":3306/" + config.dbDatabase, config.dbUsername, config.dbPassword);
            //here sonoo is database name, root is username and password  
            Statement stmt = con.createStatement();
            String insertText = "INSERT INTO " + config.dbDatabase + "." + config.dbTable + " (`open`, high, low, `close`, volume, `timestamp`) VALUES(" + 
                    ticker.open + ", " + 
                    ticker.high + ", " + 
                    ticker.low + ", " + 
                    ticker.close + ", " + 
                    ticker.volume + ", " + 
                    ticker.timestamp + ");";
            int status = stmt.executeUpdate(insertText);
            rVal = status;
//            while (rs.next()) {
//                System.out.println(rs.getInt(1) + "  " + rs.getString(2) + "  " + rs.getString(3));
//            }
            con.close();
        } catch (Exception e) {
            System.out.println(e);
        }
        return rVal;
    }
}
