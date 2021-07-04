import openpyxl
from LAC import LAC
import numpy as np

data_path = '../../data/'
train_data = data_path + 'train.xlsx'
test_data = data_path + 'test1.xlsx'
synonyms = data_path + 'synonyms_dense.txt'

lac = LAC(mode = 'lac')
lac.load_customization(synonyms, sep = '|')
excel = openpyxl.load_workbook(train_data)
sheet = excel['sheet']
sentences = []
relations = []
all_relations = []
dictionary = []
constraints = []
all_constraints = []
for row in list(sheet.rows)[1:]:
    words = lac.run(row[0].value)[0]
    relation = row[2].value.split('|')
    leafs = str(row[5].value)
    if None == leafs:
        leaf = 'None'
    elif leafs.find('｜') > -1:
        leaf = leafs.split('｜')
    else:
        leaf = leafs.split('|')
    sentences.append(words)
    relations.append(relation)
    constraints.append(leaf)
    dictionary += words
    all_relations += relation
    all_constraints += leaf
# print(sentences)
# print(relations)
excel = openpyxl.load_workbook(test_data)
sheet = excel['sheet']
test_queries = []
for row in list(sheet.rows)[1:]:
    words = lac.run(row[1].value)[0]
    test_queries.append(words)
    dictionary += words
dictionary = list(set(dictionary))
all_relations = list(set(all_relations))
all_constraints = list(set(all_constraints))
np.savetxt('dictionary.txt', dictionary, delimiter = '\n', fmt = '%s')
np.savetxt('relations.txt', all_relations, delimiter = '\n', fmt = '%s')
np.savetxt('constraints.txt', all_constraints, delimiter = '\n', fmt = '%s')
features_box = np.zeros([len(sentences), len(dictionary)], dtype = int)
labels_box = np.zeros([len(relations), len(all_relations)], dtype = int)
print('Length of feature: ', len(dictionary))
print('Length of property: ', len(all_relations))
print('Length of constraint: ', len(all_constraints))
for i in range(len(sentences)):
    # print(sentences[i])
    for word in sentences[i]:
        features_box[i][dictionary.index(word)] = 1
    for relation in relations[i]:
        labels_box[i][all_relations.index(relation)] = 1
features_labels = np.concatenate((features_box, labels_box), axis = 1)
np.savetxt("train_for_relation.csv", features_labels, delimiter=',', fmt='%d')

constraints_box = np.zeros([len(constraints), len(all_constraints)], dtype = int)
for i in range(len(constraints)):
    for constraint in constraints[i]:
        constraints_box[i][all_constraints.index(constraint)] = 1
features_constraint = np.concatenate((features_box, constraints_box), axis = 1)
np.savetxt("train_for_leaf.csv", features_constraint, delimiter=',', fmt='%d')

test_box = np.zeros([len(test_queries), len(dictionary)], dtype = int)
for i in range(len(test_queries)):
    for word in test_queries[i]:
        test_box[i][dictionary.index(word)] = 1
np.savetxt("test.csv", test_box, delimiter=',', fmt='%d')
