# coding: UTF-8
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tqdm import tqdm
import time
from datetime import timedelta
from gensim.models import word2vec, KeyedVectors
from preprocess import clean_data
import torch.utils.data as Data


def get_embedding_matrix(max_len):
    # load pre-trained word2vec
    wv = KeyedVectors.load("./saved_model/word2vec.wordvectors", mmap='r')

    with open('alidata/data/all_words.txt', 'r') as f:
        all_word = f.read().split(' ')
    # word2idx & idx2word
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_word)
    word2idx = tokenizer.word_index  # eg. {'百度':1, ...}

    embedding_matrix = np.zeros((len(word2idx) + 1, 100))
    for word, i in word2idx.items():
        try:
            embedding_vector = wv[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue
    return embedding_matrix


def build_dataset(config):
    def load_dataset(path, max_len=30):
        with open('alidata/data/all_words.txt', 'r') as f:
            all_word = f.read().split(' ')

        # categories
        categories = {'属性值': 0, '比较句': 1, '并列句': 2}

        # word2idx & idx2word
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_word)
        word2idx = tokenizer.word_index  # eg. {'百度':1, ...}
        idx2word = tokenizer.index_word  # {1: '百度', ...}

        data_x, data_y = [], []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                data_x.append(clean_data(content))
                data_y.append(label)

        # 对每个句子的每个词进行编号
        data_x_id = tokenizer.texts_to_sequences(data_x)
        # pad for sequence
        X = torch.Tensor(pad_sequences(data_x_id, maxlen=max_len))
        # label one-hot encoding
        Y = torch.Tensor(to_categorical(data_y, num_classes=len(categories)))

        return X, Y

    train = load_dataset(config.train_path, config.max_len)
    dev = load_dataset(config.dev_path, config.max_len)
    test = load_dataset(config.test_path, config.max_len)
    return train, dev, test


def build_loader(dataset, config):
    torch_dataset = Data.TensorDataset(dataset[0], dataset[1])

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1
    )
    return loader


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
