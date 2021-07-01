import codecs
import jieba
import openpyxl
import re
import numpy as np
import torch
from gensim.models import word2vec, KeyedVectors
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


def clean_data(sentence):
    ## 去除数字，空格和符号
    # r = u'[a-zA-Z0-9’!"#$%&，；：！？\'（）()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~+ +]'   # 单标签分类
    r = u'[’!"#$%&，；：！？\'（）()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~+ +]'    # 多标签分类

    sentence = re.sub(r, '', sentence)
    ## 分词
    words = jieba.cut(sentence)
    ## 去除停用词
    # stopwords = ['word1', 'word2', ...]
    stopwords = frozenset((line.rstrip() for line in codecs.open('alidata/data/cn_stopwords.txt', 'r', 'utf-8')))
    words = [w for w in words if w not in stopwords]
    return words


# get vocab & class
def build_vocab(data_path):
    source = openpyxl.load_workbook(data_path)  # train.xlsx
    sh = source['sheet']
    word_list = set()
    class_list = set()

    for cases in list(sh.rows)[1:]:
        query = cases[0].value  # 取第一列所有文本
        words = clean_data(query)
        with open('alidata/multidata/all_words.txt', 'a', encoding='utf-8') as output:
            output.write(' ' + ' '.join(words))
        for word in words:
            word_list.add(word)

        label = str(cases[5].value)
        multi_label = False
        # if '｜' in label:
        #     label = label.split('｜')
        #     multi_label = True
        if label == None:
            label = 'None'
        elif '｜' in label:  # label == '子业务｜子业务':
            label = label.split('｜')
            multi_label = True
        elif '|' in label:
            label = label.split('|')
            multi_label = True
        print(label)

        if multi_label == False:
            class_list.add(label)
        else:
            for lab in label:
                class_list.add(lab)
        # class_list.add(label) if multi_label == False else (class_list.add(lab) for lab in label)

    all_word = list(word_list)
    class_list = list(class_list)

    print(len(class_list))

    with open('alidata/multidata/class.txt', 'w', encoding='utf-8') as f:
        for cla in class_list:
            f.write(cla + '\n')

    return all_word


# get train_data
def get_train_data(data_path):
    source = openpyxl.load_workbook(data_path)  # train.xlsx
    sh = source['sheet']

    train_data_x = []
    train_data_y = []
    categories = {}
    label = 0

    with open('alidata/multidata/class.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line:
                categories[line.strip()] = label
                label += 1
    print('len of categories: ', len(categories))

    ## 单标签分类
    # for cases in list(sh.rows)[1:]:
    #     query = cases[0].value  # 取第一列所有文本
    #     answer = categories[cases[1].value]  # label
    #     words = clean_data(query)
    #
    #     train_data_x.append(words)
    #     train_data_y.append(answer)

    ## 多标签分类
    for cases in list(sh.rows)[1:]:
        query = cases[0].value  # 取第一列所有文本
        answer = str(cases[5].value)  # multi-label

        # words = clean_data(query)

        multi_label = False
        if answer == None:
            answer = 'None'
        elif '|' in answer:
            answer = answer.split('|')
            multi_label = True
        elif '｜' in answer:  # answer == '子业务｜子业务':
            answer = answer.split('｜')
            multi_label = True

        # train_data_x.append(words)
        train_data_x.append(query)
        train_data_y.append([categories[answer]] if multi_label == False else [categories[ans] for ans in answer])

    train_ratio = 0.8
    dev_test_ratio = 0.1
    with open('alidata/multidata/train.txt', 'w', encoding='utf-8') as tf:
        train_x = train_data_x[:int(len(train_data_x)*train_ratio)]
        train_y = train_data_y[:int(len(train_data_y)*train_ratio)]
        for i in range(len(train_x)):
            tf.write(train_x[i]+'\t'+' '.join([str(y) for y in train_y[i]])+'\n')
    with open('alidata/multidata/dev.txt', 'w', encoding='utf-8') as td:
        dev_x = train_data_x[int(len(train_data_x) * train_ratio):int(len(train_data_x)*(train_ratio+dev_test_ratio))]
        dev_y = train_data_y[int(len(train_data_y) * train_ratio):int(len(train_data_x)*(train_ratio+dev_test_ratio))]
        for i in range(len(dev_x)):
            td.write(dev_x[i] + '\t' + ' '.join([str(y) for y in dev_y[i]])+'\n')
    with open('alidata/multidata/test.txt', 'w', encoding='utf-8') as tt:
        test_x = train_data_x[int(len(train_data_x)*(train_ratio+dev_test_ratio)):]
        test_y = train_data_y[int(len(train_data_x)*(train_ratio+dev_test_ratio)):]
        for i in range(len(test_x)):
            tt.write(test_x[i] + '\t' + ' '.join([str(y) for y in test_y[i]])+'\n')
    return train_data_x, train_data_y


## 训练Word2Vec模型
# num_features = 100  # Word vector dimensionality
# min_word_count = 3  # Minimum word count
# num_workers = 16  # Number of threads to run in parallel
# context = 3  # Context window size
# downsampling = 1e-3  # Downsample setting for frequent words
#
# def word2vec_model():
#     sentences = word2vec.Text8Corpus("./multidata/all_words.txt")
#
#     print(sentences)
#
#     model = word2vec.Word2Vec(sentences, workers=num_workers, \
#                               vector_size=num_features, min_count=min_word_count, \
#                               window=context, sg=1, sample=downsampling)
#     model.init_sims(replace=True)
#     model.save("./saved_model/word2vec")
#
# # categories = {'属性值': 0, '比较句': 1, '并列句': 2}
#
# # all_word = build_vocab('./data/train.xlsx')
# with open('alidata/multidata/all_words.txt', 'r') as f:
#     all_word = f.read().split(' ')
#
# ## 训练word2vec模型
# word2vec_model()


datapath = 'alidata/multidata/train.xlsx'
build_vocab(datapath)
get_train_data(data_path=datapath)









# model_hasTrain = word2vec.Word2Vec.load('./saved_model/word2vec')  # 模型讀取方式
# print(model_hasTrain.wv.most_similar('流量', topn=10))
# # store the words + their trained embeddings.
# word_vectors = model_hasTrain.wv
# word_vectors.save('./saved_model/word2vec.wordvectors')
# # load
# wv = KeyedVectors.load("./saved_model/word2vec.wordvectors", mmap='r')
# print(wv['百度'])  # 获取单词的词向量
#
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(all_word)
# word2idx = tokenizer.word_index  # eg. {'百度':1, ...}
# idx2word = tokenizer.index_word  # {1: '百度', ...}
#
# print(word2idx)
#
# train_data_x, train_data_y = get_train_data('alidata/data/train.xlsx')
# # 对每个句子的每个词进行编号
# train_x_id = tokenizer.texts_to_sequences(train_data_x)
# # pad for sequence
# maxlen = 30
# train_x = pad_sequences(train_x_id, maxlen=maxlen)
# # label one-hot encoding
# train_y = to_categorical(train_data_y, num_classes=3)
#
# # embedding_matrix = np.zeros(len(vocab) + 1, 100)  # num_features
#
# sentence = train_x[0]
# print(sentence)
# sentence_embedding = np.zeros((maxlen, num_features), dtype=float)
# for i in range(len(sentence)):
#     try:
#         word_embedding = wv[idx2word[sentence[i]]]
#         sentence_embedding[i] = word_embedding
#     except KeyError:
#         continue
# print(sentence_embedding)