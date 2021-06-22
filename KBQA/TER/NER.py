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

for cases in list(sh.rows)[1:] :
    query = cases[0].value
    query_result.append(lac.run(query))

# 得到NER结果 query_result
#print(query_result)

entity_possible = []
hmm_n = ['n' 'nr' 'nz' 'ns' 'nt' 'an' 'nw' 'vn']
for cases in query_result:
    add_entity = []
    for i in range(len(cases[0])):
        if cases[1][i] == 'n' or cases[1][i] == 'nr' or cases[1][i] == 'nz' or cases[1][i] == 'ns' or cases[1][i] == 'nt' or cases[1][i] == 'an' or cases[1][i] == 'nw' or cases[1][i] == 'vn':
            add_entity.append(cases[0][i])
    if add_entity == []:
        add_entity = ['空']
    # print(add_entity)

    entity_possible.append(add_entity)

print(entity_possible)
