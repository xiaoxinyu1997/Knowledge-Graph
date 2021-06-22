from LAC import LAC
import openpyxl

# 装载LAC模型
lac = LAC(mode='lac')

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
print(query_result)