# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_loader, get_time_dif

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--saved_model', type=str, required=True, help='choose a saved_model: Bert, ERNIE')
# args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'alidata/data'  # 'query_data'  # 数据集

    # model_name = args.saved_model  # bert
    # x = import_module('models.' + model_name)

    x = import_module('models.' + 'textCNN')
    config = x.Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_loader(train_data, config)
    dev_iter = build_loader(dev_data, config)
    test_iter = build_loader(test_data, config)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
