# Relation Recognition

from LAC import LAC
import openpyxl
import numpy as np
import re

data_path = '../data/'
# Subgraph Extraction: extract the subgraphs within its two-hop 
# in the knowledge base, and each relationships in the subgraph 
# could be the target relationship.
def Subgraph(entity) :
    file = open(data_path + 'triples.txt', 'r+', encoding = 'utf-8')
    triples = []
    while True:
        line = file.readline()
        if line:
            if line.find(entity) >= 1 and line.find(entity) <= len(entity) + 1:
                # triple = ()
                bowls = line.split('\t')
                # triples.append(triple)
                elements = []
                elements.append(bowls[0][1:-1]) 
                elements.append(bowls[1][1:-1]) 
                elements.append(bowls[2][1:-2]) 
                triples.append(elements)
        else:
            break
    file.close()
    return triples

# Scoring Module: After getting allthe candidate relationships, 
# we constructed a scoring strategy.
# def scoring(question, relations) :
    # Score_relation_similarity
    # s1 = score_relation_similarity(question, relations)
    # Score_object_similarity
    # Score_char_overlap
# def score_relation_similarity():
# def score_object_similarity():
# def score_char_overlap():
# def divide_words(sentence) :
#     lac = LAC(mode = 'lac')
#     lac.load_customization(data_path + 'synonyms.txt', sep = '|')
#     return lac.run(sentence)


def excel2csv(file_path) :
    lac = LAC(mode = 'lac')
    lac.load_customization(data_path + 'synonyms.txt', sep = '|')
    source = openpyxl.load_workbook(file_path)
    sh = source['sheet']
    queries = []
    attributes = []
    for case in list(sh.rows)[1:] :
        queries.append(lac.run(case[0].value)[0])
        attributes.append(case[2].value.split('|'))
    
    features = []
    for query in queries :
        features.extend(query)

    # print(features)

    labels = []
    for attribute in attributes :
        labels.extend(attribute)

    features_vector = list(set(features))
    lables_vector = list(set(labels))

    features_box = np.zeros([len(queries), len(features_vector)], dtype = int)
    labels_box = np.zeros([len(queries), len(lables_vector)], dtype = int)

    for i in range(len(queries)) :
        for case in queries[i] :
            features_box[i][features_vector.index(case)] = 1
        for case in attributes[i] :
            labels_box[i][lables_vector.index(case)] = 1

    features_labels = np.concatenate((features_box, labels_box), axis = 1)

    np.savetxt("train_for_relations.csv", features_labels, delimiter=',', fmt='%d')


# question = '贴心流量券我为啥办理不了呀，怎么开通'

# result = Subgraph('139邮箱')
# result = divide_words(question)
# result = excel2csv(data_path + 'train.xlsx')
# print(result)
# print(len(result))

# excel2csv(data_path + 'train.xlsx')