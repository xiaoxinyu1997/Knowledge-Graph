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
    r = u'[a-zA-Z0-9’!"#$%&，；：！？\'（）()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~+ +]'
    sentence = re.sub(r, '', sentence)
    ## 分词
    words = jieba.cut(sentence)
    ## 去除停用词
    # stopwords = ['word1', 'word2', ...]
    stopwords = frozenset((line.rstrip() for line in codecs.open('alidata/data/cn_stopwords.txt', 'r', 'utf-8')))
    words = [w for w in words if w not in stopwords]
    return words


# get vocab
def build_vocab(data_path):
    source = openpyxl.load_workbook(data_path)  # train.xlsx
    sh = source['sheet']
    word_list = set()

    for cases in list(sh.rows)[1:]:
        query = cases[0].value  # 取第一列所有文本
        words = clean_data(query)
        with open('alidata/data/all_words.txt', 'a', encoding='utf-8') as output:
            output.write(' ' + ' '.join(words))
        for word in words:
            word_list.add(word)

    all_word = list(word_list)
    return all_word


# get train_data
def get_train_data(data_path):
    source = openpyxl.load_workbook(data_path)  # train.xlsx
    sh = source['sheet']

    train_data_x = []
    train_data_y = []

    for cases in list(sh.rows)[1:]:
        query = cases[0].value  # 取第一列所有文本
        answer = categories[cases[1].value]  # label
        words = clean_data(query)

        train_data_x.append(words)
        train_data_y.append(answer)

    return train_data_x, train_data_y


# BOW词袋模型(词频计数)
def bag_of_word(all_word, sentence):
    words = clean_data(sentence)

    word2int = dict((w, i) for i, w in enumerate(all_word))
    print(word2int)

    onehot_encoded = []
    for word in words:
        one_hot = [0 for _ in range(len(all_word))]
        if word in word2int:
            one_hot[word2int[word]] = 1
        onehot_encoded.append(one_hot)
    # print(onehot_encoded)
    # 按列求和
    sentence_encoded = np.array(onehot_encoded).sum(axis=0)
    return sentence_encoded


# all_word = '赤道 的 边境 万里无云 天 很 清'.split(' ')
# sentence = '赤道 的 天 非常 清'
# print(bag_of_word(all_word, sentence))

# 训练Word2Vec模型
num_features = 100  # Word vector dimensionality
min_word_count = 3  # Minimum word count
num_workers = 16  # Number of threads to run in parallel
context = 3  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

def word2vec_model():
    sentences = word2vec.Text8Corpus("./data/all_words.txt")

    print(sentences)

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                              vector_size=num_features, min_count=min_word_count, \
                              window=context, sg=1, sample=downsampling)
    model.init_sims(replace=True)
    model.save("./saved_model/word2vec")

categories = {'属性值': 0, '比较句': 1, '并列句': 2}

# all_word = build_vocab('./data/train.xlsx')
with open('alidata/data/all_words.txt', 'r') as f:
    all_word = f.read().split(' ')

## 训练word2vec模型
word2vec_model()

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