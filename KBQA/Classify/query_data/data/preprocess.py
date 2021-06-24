import openpyxl
import random

source = openpyxl.load_workbook('train.xlsx')
sh = source['sheet']

query_result = []

query_class = {'属性值': 0, '并列句': 1, '比较句': 2}

for cases in list(sh.rows)[1:]:
	query = cases[0].value
	category = query_class[cases[1].value]
	query_result.append([query, category])
	# print(query, category)

print(len(query_result))

train_ratio = 0.8
dev_test_ratio = 0.1

random.shuffle(query_result)
train_data = query_result[ : int(train_ratio*len(query_result))]
dev_data = query_result[int(train_ratio*len(query_result)) : int((train_ratio+dev_test_ratio)*len(query_result))]
test_data = query_result[int((train_ratio+dev_test_ratio)*len(query_result)) :]

with open('train.txt', 'w') as f:
	for data in train_data:
		f.write(data[0]+'\t'+str(data[1])+'\n')

with open('dev.txt', 'w') as f:
	for data in dev_data:
		f.write(data[0]+'\t'+str(data[1])+'\n')

with open('test.txt', 'w') as f:
	for data in dev_data:
		f.write(data[0]+'\t'+str(data[1])+'\n')