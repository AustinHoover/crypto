#
# The purpose of this program is to preprocess data we receive from the DataInterconnect
# to a format more usable by the datascience models we're constructing.
#
import csv

parsed = csv.reader(open('../rawdata.csv'))

writer = csv.writer(open('../datasets/Augmented.csv', 'w', newline=''))


#new format
# "timestamp","open","high","low","Close","volume",
# 1612588920000,38984.04,38984.05,38966.36,38972.2,2.4257,

previousTimestamp = 0
prevRow = []
for row in parsed:
    print(row)
    if(row[0]=="timestamp"):
        newRow = ["Timestamp","Open","High","Low","Close","Volume_(BTC)","Volume_(Currency)","Weighted_Price"]
        writer.writerow(newRow)
    if(row[0]!="timestamp"):
        newRow = [row[0],row[1],row[2],row[3],row[4],"0.9",str(float(row[4])*0.9),row[4]]
        writer.writerow(newRow)
