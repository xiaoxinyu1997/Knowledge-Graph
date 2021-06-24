from LAC import LAC
import openpyxl
import numpy as np
import re

# 装载LAC模型
lac = LAC(mode='lac')
lac.load_customization('synonyms.txt', sep = '|')

# 单个样本输入，输入为Unicode编码的字符串
# text = u"LAC是个优秀的分词工具"
# lac_result = lac.run(text)

# 批量样本输入, 输入为多个句子组成的list，平均速率更快
# texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
# lac_result = lac.run(texts)

source = openpyxl.load_workbook('train.xlsx')
sh = source['sheet']

query_result = []
entity_name = []

for cases in list(sh.rows)[1:] :
    query = cases[0].value
    query_result.append(lac.run(query))
    entity_name.append(cases[3].value) #得到每个问题的实体

# 得到NER结果 query_result
# print(query_result)
# print(entity_name)

entity_possible = []
# hmm_n = ['n' 'nr' 'nz' 'ns' 'nt' 'an' 'nw' 'vn']
for cases in query_result:
    add_entity = []
    for i in range(len(cases[0])):
        if cases[1][i] == 'n' or cases[1][i] == 'nr' or cases[1][i] == 'nz' or cases[1][i] == 'ns' or cases[1][i] == 'nt' or cases[1][i] == 'an' or cases[1][i] == 'nw' or cases[1][i] == 'vn' or cases[1][i] == 'PER' or cases[1][i] == 'v' or cases[1][i] == 'm':
            add_entity.append(cases[0][i])
    if add_entity == []:
        add_entity = ['空']
    #print(add_entity)

    entity_possible.append(add_entity)

#print(entity_possible)

#for i in range(entity_possible.count(['空'])):
#    index = entity_possible.index(['空'])
#    print('index:', index+i)
#    entity_possible.remove(['空'])
#    print(query_result[index+i])

words = []
label_index = []
for row in entity_possible:
    words = words + row

words = list(set(words))
label_index = list(set(entity_name))

#print(len(words))

features = np.zeros([len(entity_possible), len(words)], dtype = int)
labels = np.zeros([features.shape[0], 1], dtype = int)
i = 0

for entity in entity_possible:
    for word in entity:
        features[i][words.index(word)] = 1
    i = i+1
    
i = 0
for label in entity_name:
    labels[i][0] = label_index.index(label)
    i = i+1

features_labels = np.concatenate((features, labels), axis = 1)
#print(features_labels)
np.savetxt("train_for_entity.csv", features_labels, delimiter=',', fmt='%d')