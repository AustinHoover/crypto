package main;

import com.google.gson.Gson;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 *
 * @author Austin Hoover
 */
@SpringBootApplication
public class Main {
    public static Config appConfig;
    public static void main(String args[]) throws IOException{
        String configFilePath = "./config.json";
        Gson gson = new Gson();
        appConfig = gson.fromJson(Files.newBufferedReader(new File(configFilePath).toPath()), Config.class);
        SpringApplication.run(Main.class, args);
    }
}
