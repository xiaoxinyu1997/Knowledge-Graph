import pandas as pd
import json
import pickle
import csv
import openpyxl

if __name__ == '__main__':
    leaf_model_pickle = open('leaf_model.pickle', 'rb')
    relations_model_pickle = open('relations_model.pickle', 'rb')
    leaf_model = pickle.load(leaf_model_pickle)
    relations_model = pickle.load(relations_model_pickle)
    data = pd.read_csv('test.csv').iloc[:]
    relationtxt = open('relations.txt', 'r')
    relation_dict = str(relationtxt.read()).split('\n')
    leaftxt = open('constraints.txt', 'r')
    leaf_dict = str(leaftxt.read()).split('\n')
    excel = openpyxl.load_workbook('test1.xlsx')
    sheet = excel['sheet']

    X = data
    relation = relations_model.predict(X)
    relation_box = []
    for row in relation:
        row_relation = ''
        for i in range(len(row)):
            if row[i] == 1:
                row_relation = row_relation + relation_dict[i] + '|'
        relation_box.append(row_relation[:-1]) 
    # print(relation_box)
    for i in range(len(relation_box)):
        sheet['D' + str(i + 2)] = relation_box[i - 1]
    sheet['D' + str(len(relation_box) + 2)] = relation_box[len(relation_box) - 1]

    leaf = leaf_model.predict(X)
    leaf_box = []
    for row in leaf:
        row_leaf = ''
        for i in range(len(row)):
            if row[i] == 1:
                row_leaf = row_leaf + leaf_dict[i] + '|'
        leaf_box.append(row_leaf[:-1])
    for i in range(len(leaf_box)):
        sheet['F' + str(i + 2)] = leaf_box[i - 1]
    sheet['F' + str(len(leaf_box) + 2)] = leaf_box[len(leaf_box) - 1]

    excel.save('test.xlsx')


