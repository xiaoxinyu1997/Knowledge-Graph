import numpy as np
import torch
from importlib import import_module
import torch.utils.data as Data

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import openpyxl
from preprocess import clean_data

path = 'saved_model/saved_dict/TextCNN.ckpt'
dataset = 'alidata'

x = import_module('models.' + 'textCNN')
config = x.Config(dataset)
model = x.Model(config).to(config.device)
# load model
model_ckpt = torch.load(path)
model.load_state_dict(model_ckpt)

with open('alidata/data/all_words.txt', 'r') as f:
    all_word = f.read().split(' ')

categories = {'属性值': 0, '比较句': 1, '并列句': 2}

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_word)

data_x, data_y = [], []
test_source = openpyxl.load_workbook('alidata/data/test1.xlsx')
sh_1 = test_source['sheet']
for cases in list(sh_1.rows)[1:]:
    query = cases[1].value
    words = clean_data(query)
    data_x.append(words)

# 对每个句子的每个词进行编号
data_x_id = tokenizer.texts_to_sequences(data_x)
# pad for sequence
X = torch.Tensor(pad_sequences(data_x_id, maxlen=30))

torch_dataset = Data.TensorDataset(X)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=1
)

for i, query in enumerate(loader):
    print(i, query)
    query = query.to(config.device)  # torch.from_numpy(np.asarray(query)).to(config.device)   # query.to(config.device)
    outputs = model(query)
    predic = torch.max(outputs.data, 1)[1].cpu()
    print(predic)