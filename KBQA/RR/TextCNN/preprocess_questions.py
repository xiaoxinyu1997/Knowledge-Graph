import openpyxl
from LAC import LAC
import numpy as np

data_path = '../../data/'
dictionary = np.load('dictionary.npy', allow_pickle=True).item()
# print(dictionary)
# print(len(dictionary['专会']))
lac = LAC(mode = 'lac')
lac.load_customization(data_path + 'synonyms_dense.txt', sep = '|')
excel = openpyxl.load_workbook(data_path + 'train.xlsx')
sheet = excel['sheet']
sentences = []
for row in list(sheet.rows)[1:]:
    sentences.extend(lac.run(row[0].value)[0])
max = len(sentences[0])
for sentence in sentences:
    if max < len(sentence):
        max = len(sentence)

embeddings = []
for sentence in sentences:
    words = []
    for i in range(max-1):
        if i-1 < len(sentence) and sentence[i-1] in dictionary:
            words.append(dictionary[sentence[i-1]])
        else:
            words.append(np.zeros([len(dictionary['专会'])], dtype = 'float32'))
    print(words)
    embeddings.append(words)

np.save('embeddings.npy', embeddings)