package main;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

/**
 *
 * @author Austin Hoover
 */
public class DatabaseManager {
    public Connection dbCon;
    public void connectToDB(){
        int rVal = 0;
        String username = Main.appConfig.dbUsername;
        String password = Main.appConfig.dbPassword;
        String address = Main.appConfig.dbAddress;
        String database = Main.appConfig.dbDatabase;
        String table = Main.appConfig.dbTable;
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
            dbCon = DriverManager.getConnection(
                    "jdbc:mysql://" + address + ":3306/" + database, username, password);
        } catch (Exception e) {
            e.printStackTrace();
//            System.out.println(e);
        }
    }
    
    public Connection getConnect(){
        return dbCon;
    }
    
    public void disconnectFromDB(){
        try {
            dbCon.close();
        } catch (SQLException ex) {
            ex.printStackTrace();
        }
    }
}
