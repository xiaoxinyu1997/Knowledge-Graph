"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os
import argparse
import logging
from collections import OrderedDict

# import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# from nni.nas.pytorch.mutables import LayerChoice, InputChoice
# from nni.algorithms.nas.pytorch.classic_nas import get_and_apply_next_architecture

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger('mnist_AutoML')


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.conv3 = nn.Conv2D(1, 1, (3, config.word_embedding_dimension)) # in_channel, out_channel, size
        self.conv4 = nn.Conv2D(1, 1, (4, config.word_embedding_dimension))
        self.conv5 = nn.Conv2D(1, 1, (5, config.word_embedding_dimension))
        self.Max3_pool = nn.MaxPool2d((self.config.sentence_max_size-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((self.config.sentence_max_size-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((self.config.sentence_max_size-5+1, 1))
        self.linear1 = nn.Linear(3, config.label_num)
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # in_channel, out_channel, size, stride, padding
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(stride=2, kernel_size=2),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(stride=2, kernel_size=2)
        # )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(7 * 7 * 128, hidden_size), # in_features, out_features
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5)
        # )
        # self.fc2 = LayerChoice([
        #     nn.Sequential(
        #         nn.Linear(hidden_size, 128),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5)
        #     ),
        #     nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, 128),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5)
        #     ),
        #     nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, 128),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5)
        #     ),
        #     nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, 128),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5)
        #     ),
        #     nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, 128),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5)
        #     ),
        #     nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5),
        #         nn.Linear(hidden_size, 128),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.5)
        #     )
        # ], key = 'fc2')

        #self.fc3 = nn.Sequential(
        #    nn.Linear(hidden_size, 128),
        #    nn.ReLU(),
        #    nn.Dropout(p=0.5)
        #)
        # self.fc_out = nn.Linear(128, 10)
        # skip connection over fc
        #self.input_switch = InputChoice(6,1) # n_chosen (int) -- Recommended inputs to choose. If None, mutator is instructed to select any.

        # # two options of conv1
        # self.conv1 = LayerChoice(OrderedDict([
        #     ("conv5x5", nn.Conv2d(1, 20, 5, 1)), 
        #     ("conv3x3", nn.Conv2d(1, 20, 3, 1))
        # ]), key='first_conv')
        # # two options of mid_conv
        # self.mid_conv = LayerChoice([
        #     nn.Conv2d(20, 20, 3, 1, padding=1),
        #     nn.Conv2d(20, 20, 5, 1, padding=2)
        # ], key='mid_conv')
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4*4*50, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, 10)
        # # skip connection over mid_conv
        # self.input_switch = InputChoice(n_candidates=2,
        #                                 n_chosen=1,
        #                                 key='skip')

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # old_x = x
        # x = F.relu(self.mid_conv(x))
        # zero_x = torch.zeros_like(old_x)
        
        # x = torch.add(x, skip_x)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4*4*50)
        # x = F.relu(self.fc1(x))
        # x = self.features(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.fc1(x)
        # old_x = x
        # x = self.fc2(x)
        #zero_x = torch.zeros_like(old_x)
        #x = self.input_switch([zero_x, old_x])
        #skip_x = self.input_switch([zero_x, old_x])
        #x = torch.add(x, skip_x)
        #x = self.fc3(x)
        # x = self.fc_out(x)
        # return F.log_softmax(x, dim=1)
        batch = x.shape[0]
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, self.config.label_num)

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def getdata(excel):
    excel


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_dir = args['data_dir']

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(data_dir, train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args['batch_size'], shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=1000, shuffle=True, **kwargs)

    # print('train_loader: ', train_loader)
    # print('test_loader: ', test_loader)

    hidden_size = args['hidden_size']

    # model = Net(hidden_size=hidden_size).to(device)
    # get_and_apply_next_architecture(model)
    # optimizer = optim.SGD(model.parameters(), lr=args['lr'],
    #                       momentum=args['momentum'])

    # for epoch in range(1, args['epochs'] + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     test_acc = test(args, model, device, test_loader)

    #     if epoch < args['epochs']:
    #         # report intermediate result
    #         nni.report_intermediate_result(test_acc)
    #         logger.debug('test accuracy %g', test_acc)
    #         logger.debug('Pipe send intermediate result done.')
    #     else:
    #         # report final result
    #         nni.report_final_result(test_acc)
    #         logger.debug('Final result is %g', test_acc)
    #         logger.debug('Send final result done.')


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Text CNN')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
