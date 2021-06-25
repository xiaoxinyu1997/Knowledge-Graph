# Relation Recognition
from LAC import LAC
import openpyxl
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import torch

data_path = '../data/'
dict_path = 'dict.txt'
file = open(dict_path, 'r')
dictionary = file.read().split('\n')
lac = LAC(mode = 'lac')
lac.load_customization(data_path + 'synonyms.txt', sep = '|')
bert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
alpha = 0.5
beta = 2.0
# Subgraph Extraction: extract the subgraphs within its two-hop 
# in the knowledge base, and each relationships in the subgraph 
# could be the target relationship.
def Subgraph(entity) :
    file = open(data_path + 'triples.txt', 'r+', encoding = 'utf-8')
    triples = []
    while True:
        line = file.readline()
        if line:
            if line.find('<' + entity + '>') >= 0 and line.find(entity) <= line.find('>'):
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
def sentence_similarity(s1, s2):
    vector1 = torch.tensor(bert_model.encode(s1)) 
    vector2 = torch.tensor(bert_model.encode(s2))
    # vector1 = sentence2vector(s1, dictionary)
    # vector2 = sentence2vector(s2, dictionary)
    cos = cosine(vector1, vector2)
    if np.isnan(cos):
        return 0.0
    else:
        return cos

def sentence_overlap(s1, s2):
    vector1 = sentence2vector(s1, dictionary)
    vector2 = sentence2vector(s2, dictionary)
    cos = cosine(vector1, vector2)
    if np.isnan(cos):
        return 0.0
    else:
        return cos

def cosine(a, b):
    return a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))

def sentence2vector(sentence, diction):
    words = lac.run(sentence)[0]
    vector = np.zeros([len(diction)], dtype = int)
    for word in words:
        if diction.count(word):
            vector[diction.index(word)] = 1
    return vector

def excel2csv(file_path) :
    # lac = LAC(mode = 'lac')
    # lac.load_customization(data_path + 'synonyms.txt', sep = '|')
    source = openpyxl.load_workbook(file_path)
    sh = source['sheet']
    queries = []
    attributes = []
    objects = []
    for case in list(sh.rows)[1:] :
        queries.append(lac.run(case[0].value)[0])
        attributes.append(case[2].value.split('|'))
        # print(case[5].value)
        if case[5].value == None:
            objects.append(['None'])
        # elif isinstance(case[5].value, str):
        #     if case[5].value.count('|') >= 0:
        #         objects.append(case[5].value.split('|'))
        #     elif case[5].value.count('｜') >= 0:
        #         objects.append(case[5].value.split('｜'))
        #     elif case[5].value.count('丨') >= 0:
        #         objects.append(case[5].value.split('丨'))
        else:
            objects.append(str(case[5].value).split('|'))
    features = []
    for query in queries :
        features.extend(query)
    print(objects)
    # print(features)
    labels = []
    for attribute in attributes :
        labels.extend(attribute)
    labels2 = []
    for case in objects:
        labels2.extend(case)
    # print(labels2)
    features_vector = list(set(features))
    lables_vector = list(set(labels))
    lables2_vector = list(set(labels2))
    # print(len(lables_vector))
    features_box = np.zeros([len(queries), len(features_vector)], dtype = int)
    labels_box = np.zeros([len(queries), len(lables_vector)], dtype = int)
    labels_box2 = np.zeros([len(queries), len(lables2_vector)], dtype = int)
    for i in range(len(queries)) :
        for case in queries[i] :
            features_box[i][features_vector.index(case)] = 1
        for case in attributes[i] :
            labels_box[i][lables_vector.index(case)] = 1
        for case in objects[i]:
            labels_box2[i][lables2_vector.index(case)] = 1
    features_labels = np.concatenate((features_box, labels_box), axis = 1)
    features_labels2 = np.concatenate((features_box, labels_box2), axis = 1)
    np.savetxt("dict.txt", features_vector + lables_vector + lables2_vector, delimiter = '\n', fmt = '%s')
    np.savetxt("train_for_relations.csv", features_labels, delimiter=',', fmt='%d')
    np.savetxt("train_for_leaf.csv", features_labels2, delimiter=',', fmt='%d')

def relation_reg(entity, question):
    one_hop_subgraph = Subgraph(entity)
    # for item in one_hop_subgraph:
    #     print(item)
    two_hop_subgraph = one_hop_subgraph.copy()
    for triple in one_hop_subgraph:
        two_hop_subgraph += Subgraph(triple[2])
    # for item in two_hop_subgraph:
    #     print(item)
    SRQ = []
    SOQ = []
    for triple in two_hop_subgraph:
        SRQ.append(alpha*sentence_similarity(triple[1], question) + beta*sentence_overlap(triple[1], question))
        SOQ.append(alpha*sentence_similarity(triple[2], question) + beta*sentence_overlap(triple[2], question))
        # print(triple[1])
    # print(SRQ)
        # print(SRQ[triple[1]])
    # for triple in two_hop_subgraph:
    #     SOQ += sentence_similarity(triple[2], question)
    #     # print(triple[2])
    #     # print(SOQ[triple[2]])
    # for item in SRQ:
    #     print(item)
    # print('______________________________________________________________')
    # print('______________________________________________________________')
    # for item in SOQ:
    #     print(item)
    for i in range(len(two_hop_subgraph)):
        print(two_hop_subgraph[i][1], SRQ[i] + SOQ[i])
    return SRQ, SOQ

question = '9元百度专属定向流量包如何取消'
entity = '专属定向流量包'
relation_reg(entity, question)
# result = Subgraph('139邮箱')
# result = divide_words(question)
# result = excel2csv(data_path + 'train.xlsx')
# print(result)
# print(len(result))
# excel2csv(data_path + 'train.xlsx')
# print(sentence_similarity('今天天气很好', '天气不错啊今天'))
# file = open('dict.txt', 'r')
# dictionary = file.read().split('\n')
# lac = LAC(mode = 'lac')
# lac.load_customization(data_path + 'synonyms.txt', sep = '|')
# words = lac.run('9元百度专属定向流量包如何取消')
# print(sentence2vector(words[0], dictionary))