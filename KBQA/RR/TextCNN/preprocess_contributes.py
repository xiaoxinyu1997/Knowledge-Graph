import openpyxl
import numpy as np

train_excel = openpyxl.load_workbook('../../data/train.xlsx')
sheet = train_excel['sheet']
# 获取每一行的label和所有的label
labels = []
all_labels = []
for row in list(sheet.rows)[1:]:
    labels.append(row[2].value.split('|'))
    all_labels += row[2].value.split('|')
all_labels = list(set(all_labels))
print('length of labels one_hot_coding: ', len(all_labels))
# 将标签转换为one_hot
def label2one_hot(label, all_labels):
    label_one_hot = np.zeros([len(all_labels)], dtype = int) 
    for word in label:
        if all_labels.count(word):
            label_one_hot[all_labels.index(word)] = 1
    return label_one_hot
labels_one_hot = []
for label in labels:
    print(label)
    print(label2one_hot(label, all_labels))
    labels_one_hot.append(label2one_hot(label, all_labels))
print('length of questions: ', len(labels_one_hot))
# print(labels_one_hot)
# 保存文件
np.save('labels.npy', labels_one_hot)



