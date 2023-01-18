#  Q2. Perform the following preprocessing tasks on the dirty_iris datasetii.
# i) Calculate the number and percentage of observations that are complete.
# ii) Replace all the special values in data with NA.
# iii) Define these rules in a separate text file and read them.
# (Use editfile function in R (package editrules). Use similar function in Python).
# Print the resulting constraint object.
# – Species should be one of the following values: setosa, versicolor or virginica.
# – All measured numerical properties of an iris should be positive.
# – The petal length of an iris is at least 2 times its petal width.
# – The sepal length of an iris cannot exceed 30 cm.
# – The sepals of an iris are longer than its petals.
# iv)Determine how often each rule is broken (violatedEdits). Also summarize and plot the
# result.
# v) Find outliers in sepal length using boxplot and boxplot.stats


# Solution

class iris:
    def __init__(self, sepalLength, sepalWidth, petalLength, petalWidth, species):
        self.sepalLength = sepalLength
        self.sepalWidth = sepalWidth
        self.petalWidth = petalWidth
        self.petalLength = petalLength
        self.species = species

    def value(self):
        temp = self.sepalLength + ',' + self.sepalWidth + ',' + self.petalLength + ',' + self.petalWidth + ',' + self.species
        return temp


irisList = []
fileLine = []

file = open("dirty_iris.csv", "r")
for x in file:
    fileLine.append(x)

file.close()

for x in range(0, len(fileLine)):
    temp = fileLine[x].split(',')
    irisList.append(iris(temp[0], temp[1], temp[2], temp[3], temp[4]))

irisLength = len(irisList)-1

incompleteData = 0

for x in irisList:
    if (x.sepalLength == "NA" or x.sepalWidth == "NA" or x.petalLength == "NA" or x.petalWidth == "NA" or x.species == "NA"):
        incompleteData = incompleteData + 1

print("Total no. of Data : ", irisLength)
print("No. of Incomplete Data : ", incompleteData)

try:
    print("Percentage of Incomplete Data : ", (incompleteData/irisLength)*100)
except:
    print("File is Empty")   

# (ii) part starts Here.

for l in range(0, len(irisList)):
    try:
        if (float(irisList[l].sepalLength) <= 0.000):
            irisList[l].sepalLength = "NA"
        if (float(irisList[l].sepalWidth) <= 0.0000):
            irisList[l].sepalWidth = "NA"
        if (float(irisList[l].petalLength) <= 0.0000):
            irisList[l].petalLength = "NA"           
        if (float(irisList[l].petalWidth) <= 0.0000):
            irisList[l].petalWidth = "NA"           
    except:
        continue

fileWrite = open("dirty_iris.csv", "w")

for l in range(0, len(irisList)):
    fileWrite.write(irisList[l].value())


