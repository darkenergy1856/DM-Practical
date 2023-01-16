import matplotlib.pyplot as plt

class people:
    def __init__(self, Age, ageGroup, height, status, yearsMarried):
        self.Age = Age
        self.ageGroup = ageGroup
        self.height = height
        self.status = status
        self.yearsMarried = yearsMarried

    def checkAge(self):
        if (self.Age >= 1 and self.Age <= 150):
            return True
        return False

    def checkYearMarried(self):
        if self.Age > self.yearsMarried:
            return True
        return False

    def checkStatus(self):
        temp = ["single", "married", "widowed"]
        if self.status.lower() in temp:
            return True
        return False

    def checkHeight(self):
        if self.height <= 0:
            return False
        return True

    def checkAgeGroup(self):
        if self.Age < 18 and self.ageGroup.lower() == "child":
            return True
        elif self.Age >= 18 and self.Age <= 65 and self.ageGroup.lower() == "adult":
            return True
        elif self.Age > 65 and self.ageGroup.lower() == "elderly":
            return True
        else:
            return False

    def check(self):
        if self.checkAge() and self.checkStatus() and self.checkAgeGroup() and self.checkYearMarried() and self.checkHeight():
            return True
        return False


file = open('people.txt', "r")
fileLine = []
peopleList = []
for line in file:
    fileLine.append(line)

file.close()

for x in range(1, len(fileLine)):
    temp = fileLine[x].split()
    peopleList.append(
        people(int(temp[0]), temp[1], float(temp[2]), temp[3], int(temp[4])))

validData = 0   

for x in peopleList:
    if x.check() == True:
        validData = validData+1


print("No. of Valid Data Set : " , validData)
print("No. of Invalid Data Set : " , len(peopleList) - validData)
Options = ['Valid Data' ,  'Invalid Data']
Value = [validData ,len(peopleList) - validData]

plt.bar(Options, Value, color ='maroon',
        width = 0.4)
plt.ylabel('No. of Data')
plt.show()

