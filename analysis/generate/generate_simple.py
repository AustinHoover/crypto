#
# Purpose: The purpose of this program is to generate a dummy dataset
#
import csv

outFile = open('../datasets/simple.csv','w')

writer = csv.writer(outFile)
writer.writerow(["timestamp","value"])
for i in range(0,100):
    writer.writerow([i,i/2])