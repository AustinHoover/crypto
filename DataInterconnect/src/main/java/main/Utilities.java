package main;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

/**
 *
 * @author Austin Hoover
 */
public class Utilities {
    
    public static String restfulPost(String address, String body){
        String rVal = "";
        try {
            URL url = new URL(address);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "text/plain");
            conn.connect();
            //write out request body
            OutputStream os = conn.getOutputStream();
            os.write(body.getBytes());
            //read response
            BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            String currentLine = "";
            while((currentLine = reader.readLine()) != null){
                rVal = rVal + currentLine;
            }
            os.close();
            reader.close();
            conn.disconnect();
        } catch (MalformedURLException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        return rVal;
    }
    
}
