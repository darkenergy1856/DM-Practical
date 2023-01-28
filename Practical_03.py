# Load the data from wine dataset. Check whether all attributes are standardized or not (mean
# is 0 and standard deviation is 1). If not, standardize the attributes. Do the same with Iris dataset.

import csv
import statistics

#### WINE ####

data = []
mean = []
standardDeviation = []
tempNew = []

with open('wine.csv', mode='r') as file:

    csvFile = csv.reader(file)

    for lines in csvFile:
        data.append(lines)

for x in range(0, 13):
    temp = []
    for l in data:
        temp.append(float(l[x+1]))
    mean.append(statistics.mean(temp))
    standardDeviation.append(statistics.stdev(temp))

isMeanZero = True
isStandardDeviationOne = True

for x in mean:
    if (round(x) != 0):
        isMeanZero = False

for x in standardDeviation:
    if (round(x) != 1):
        isStandardDeviationOne = False

if ((isMeanZero and isStandardDeviationOne) != True):
    for x in data:
        tempNew.append([x[0], (float(x[1])-mean[0])/standardDeviation[0], (float(x[2])-mean[1])/standardDeviation[1], (float(x[3])-mean[2])/standardDeviation[2], (float(x[4])-mean[3])/standardDeviation[3], (float(x[5])-mean[4])/standardDeviation[4], (float(x[6])-mean[5])/standardDeviation[5], (float(x[7])-mean[6]) /
                        standardDeviation[6], (float(x[8])-mean[7])/standardDeviation[7], (float(x[9])-mean[8])/standardDeviation[8], (float(x[10])-mean[9])/standardDeviation[9], (float(x[11])-mean[10])/standardDeviation[10], (float(x[12])-mean[11])/standardDeviation[11], (float(x[13])-mean[12])/standardDeviation[12]])

    with open('wine.csv' , mode='w' , newline='') as file :
        writer = csv.writer(file)
        writer.writerows(tempNew)


#### DIRT IRIS ####

data = []
mean = []
standardDeviation = []
tempNew = []

with open('dirty_iris.csv' , mode = 'r') as file :
    csvFile = csv.reader(file)

    for lines in csvFile:
        data.append(lines)

for x in range(0 , 4):
    temp = []
    for y in range ( 1 , len(data)):
        if(data[y][x] != 'NA'):
            temp.append(float(data[y][x]))
    mean.append(statistics.mean(temp))
    standardDeviation.append(statistics.stdev(temp))

isMeanZero = True
isStandardDeviationOne = True

for x in mean:
    if (round(x) != 0):
        isMeanZero = False

for x in standardDeviation:
    if (round(x) != 1):
        isStandardDeviationOne = False

if ((isMeanZero and isStandardDeviationOne) != True):
    tempNew.append(data[0])
    for x in range( 1 , len(data)):
        row = []
        for y in range(0 , 5):
            if(y == 4):
                row.append(data[x][y])
            else:
                if(data[x][y] != 'NA'):
                    row.append((float(data[x][y]) - mean[y])/standardDeviation[y])
                else:
                    row.append('NA')
        tempNew.append(row)

    with open('dirty_iris.csv' , mode = 'w', newline='') as file :
        writer = csv.writer(file)
        writer.writerows(tempNew)