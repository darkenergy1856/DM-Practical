# Run Apriori algorithm to find frequent itemsets and association rules
# 1.1 Use minimum support as 50% and minimum confidence as 75%
# 1.2 Use minimum support as 60% and minimum confidence as 60 %


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori

dataset = [['A', 'B', 'C', 'D', 'F', 'H'],['B', 'E', 'F', 'H'],['A', 'C', 'E'],['B', 'C', 'D', 'F', 'H'],
['A', 'B', 'C', 'D', 'E'],['C','D','F','H'],['A','C','D','H'],['E','H']]

data_file = "Grocery.csv"

file = open(data_file , "r")
fileLine = []
dataList = []

for line in file:
    fileLine.append(line)

file.close()

for x in range(0 , len(fileLine)):
    temp = fileLine[x].split(",")
    temp.pop()
    dataList.append(temp)

# print(dataList)

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
dataFrame = pd.DataFrame(te_ary , columns = te.columns_) # type: ignore
print(dataFrame)

frequent_itemsets = apriori(dataFrame, min_support=0.5, use_colnames=True)
print(frequent_itemsets)
print(association_rules(frequent_itemsets, metric="confidence", min_threshold=0.75))

frequent_itemsets = apriori(dataFrame, min_support=0.6, use_colnames=True)
print(frequent_itemsets)
print(association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6))

