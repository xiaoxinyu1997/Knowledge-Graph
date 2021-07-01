f = open("triples.txt","r",encoding='utf-8')   #设置文件对象
data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
f.close()             #关闭文件
# print(type(data))
entity2entity=[]
for line in data:
    line = line[:-1]
    entity2entity.append(line.split("\t")) #实体关系列表
for entity in entity2entity:
    entity[0]=entity[0][1:-1]
    entity[1]=entity[1][1:-1]
    entity[2]=entity[2][1:-1]
# print(entity2entity)

import csv


def remover_constraint(entity2entity,constraint):
    remover_entity=set()
    for constr in constraint:
        for entity in entity2entity:
            if entity[2] != constr:
                remover_entity.add(entity[0])
    fit_entity=[]
    for entity in entity2entity:
        if entity[0] not in remover_entity:
            fit_entity.append(entity)
    return fit_entity
def select_simple2(entity2entity,entity,relation1,relation2):
    for x in entity2entity:
        # print(x[0],relation1,x[1])
        if entity == x[0] and relation1 in x[1]:
            # print('2-hop')
            select_simple(entity2entity,x[2],relation2)


def select_simple(entity2entity,entity,relation):
    for x in entity2entity:
        if entity == x[0] and relation in x[1]:
            print(entity,relation,x[2])
def select_compound(entity2entity,entity1,relation1,relation2):
    middleentity=set()
    for entity in entity2entity:
        if entity1 == entity[0] and (relation1 in entity[1] or relation2 in entity[1]):
            middleentity.add(entity[0])
            middleentity.add(entity[2])
    for midentity in middleentity:
        select_simple(entity2entity,midentity,relation1)
        select_simple(entity2entity,midentity,relation2)
def select_compare(entity2entity,entity,relation1,relation2):
    print("no")




if __name__ == '__main__':
    # print(entity2entity)
    # entity='139邮箱'
    # relation1='档位介绍表'
    # relation2='产品名称'
    # # select_simple(entity2entity,entity,relation1)
    # select_compound(entity2entity,entity,relation1,relation2)
    entity=[]
    relation=[]
    sentence_class=[]
    file = "train.csv"
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            sentence_class.append(i[1])
            relation.append(i[2])
            entity.append(i[3])
    for i in range(1,100):
        print(i)
        relation1=relation[i].split('-')[0] #业务简介
        relation2=relation[i].split('-')[-1] #业务简介
        # print(entity[i],relation1)
        if(sentence_class[i]=='属性值'):
            if relation1 == relation2:
                print('简单1')
                select_simple(entity2entity,entity[i],relation1)
            else:
                print('简单2')
                select_simple2(entity2entity,entity[i],relation1,relation2)