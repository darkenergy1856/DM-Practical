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
import matplotlib.pyplot as plt
import copy

class iris:
    def __init__(self, sepalLength, sepalWidth, petalLength, petalWidth, species):
        self.sepalLength = sepalLength
        self.sepalWidth = sepalWidth
        self.petalWidth = petalWidth
        self.petalLength = petalLength
        self.species = species

    def value(self):
        temp = self.sepalLength + ',' + self.sepalWidth + ',' + \
            self.petalLength + ',' + self.petalWidth + ',' + self.species
        return temp

    def checkConstraiont(self):
        temp = [0, 0, 0, 0, 0]
        speciesValue = ['"setosa"\n', '"versicolor"\n', '"virginica"\n']

        if (self.sepalLength == 'NA' or self.sepalWidth == 'NA' or self.petalLength == 'NA' or self.petalWidth == 'NA'):
            return temp
        if (self.species in speciesValue):
            temp[0] = 0
        else:
            temp[0] = 1

        if (float(self.sepalLength) < 0 or float(self.sepalWidth) < 0 or float(self.petalLength) < 0 or float(self.petalWidth) < 0):
            temp[1] = 1
        if (float(self.petalLength) < 2*float(self.petalWidth)):
            temp[2] = 1
        if (float(self.sepalLength) > 30.0):
            temp[3] = 1
        if (float(self.sepalLength) < float(self.petalLength) or float(self.sepalWidth) < float(self.petalWidth)):
            temp[4] = 1

        return temp


irisList = []
fileLine = []
originalIrisList = []

file = open("dirty_iris.csv", "r")
for x in file:
    fileLine.append(x)

file.close()

for x in range(0, len(fileLine)):
    temp = fileLine[x].split(',')
    irisList.append(iris(temp[0], temp[1], temp[2], temp[3], temp[4]))

originalIrisList = copy.deepcopy(irisList)
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
        if (float(irisList[l].sepalLength) <= 0.000 or irisList[l].sepalLength == 'Inf'):
            irisList[l].sepalLength = "NA"
        if (float(irisList[l].sepalWidth) <= 0.0000 or irisList[l].sepalWidth == 'Inf'):
            irisList[l].sepalWidth = "NA"
        if (float(irisList[l].petalLength) <= 0.0000 or irisList[l].petalLength == 'Inf'):
            irisList[l].petalLength = "NA"
        if (float(irisList[l].petalWidth) <= 0.0000 or irisList[l].petalWidth == 'Inf'):
            irisList[l].petalWidth = "NA"
    except:
        continue

fileWrite = open("dirty_iris.csv", "w")

for l in range(0, len(irisList)):
    fileWrite.write(irisList[l].value())

fileWrite.close()

ruleVoilation = [0, 0, 0, 0, 0]

for l in range(1, len(irisList)):
    result = originalIrisList[l].checkConstraiont()
    for k in range(0, 5):
        ruleVoilation[k] = ruleVoilation[k] + result[k]

print("Species Rule Voilation : " , ruleVoilation[0])
print("Positive Value Rule Voilation : " , ruleVoilation[1])
print("Petal Length Rule Voilation : " , ruleVoilation[2])
print("Sepal Length Rule Voilation : " , ruleVoilation[3])
print("Sepal and Petal Length Rule Voilation : " , ruleVoilation[4])

Options = ["Species Rule Voilation" ,"Positive Value Rule Voilation","Petal Length Rule Voilation","Sepal Length Rule Voilation","Sepal and Petal Length Rule Voilation"]


plt.bar(Options, ruleVoilation, color ='maroon',
        width = 0.4)
plt.ylabel('No. of Rule Voilation')
plt.title("No. of Record : " + str(len(irisList)-1))

plt.show()

sepalLengthList = []

for l in range(1 , len(irisList)):
    try:
     sepalLengthList.append(float(irisList[l].sepalLength))
    except:
        continue

graph = plt.boxplot(sepalLengthList)
plt.title("Sepal Length")
plt.show()

print(graph)
