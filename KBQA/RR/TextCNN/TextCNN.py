"""
A deep text classifier using TextCNN.

"""
import os
import argparse
import logging
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

logger = logging.getLogger('TextCNN')

class RelationConfig(object):
    word_embedding_dimension = 128
    sentence_max_size = 11
    label_num = 19
    dropout = 0.1

# define dataset
class TextDataset(Dataset):
    def __init__(self, label_file):
        self.labels = np.load(label_file, allow_pickle=True)
        logger.debug('Loading dataset...')
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        all_sentence_vector = np.load('embeddings.npy', allow_pickle=True)
        sentence_vector = all_sentence_vector[idx]
        # sentence_vector = np.expand_dims(sentence_vector, 0)
        # print(sentence_vector)
        # sentence_vector = np.expand_dims(sentence_vector, 0)
        logger.debug('Loading batch...')
        label = self.labels[idx]
        return sentence_vector, label

# define torch NN
class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__() 
        self.config = config
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = (2, config.word_embedding_dimension), stride = 1)
        self.conv3 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = (3, config.word_embedding_dimension), stride = 1)
        self.conv4 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = (4, config.word_embedding_dimension), stride = 1)
        self.Max2_pool = nn.MaxPool2d((self.config.sentence_max_size-2+1, 1))
        self.Max3_pool = nn.MaxPool2d((self.config.sentence_max_size-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((self.config.sentence_max_size-4+1, 1))
        self.linear = nn.Linear(3 * 3, config.label_num)
        self.dropout = nn.Dropout(p=config.dropout)
        logger.debug('Creating neural net...')

    def forward(self, x):
        x = x.unsqueeze(1).to(dtype=torch.float)
        x1 = F.relu(self.conv2(x))
        x1 = self.Max2_pool(x1)
        x2 = F.relu(self.conv3(x))
        x2 = self.Max3_pool(x2)
        x3 = F.relu(self.conv4(x))
        x3 = self.Max4_pool(x3)
        x = torch.cat([x1, x2, x3], 1)
        x = self.linear(x)
        x = self.dropout(x)
        return F.Sigmoid(x)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        print(data.size())
        output = model(data)
        # loss = F.nll_loss(output, target)
        # target = torch.unsqueeze(target, dim=1)
        # target = torch.unsqueeze(target, dim=1)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

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
    train_loader = DataLoader(TextDataset(label_file = 'labels.npy'), batch_size = 20, shuffle = True)
    test_loader = DataLoader(TextDataset(label_file = 'labels.npy'), batch_size = 20, shuffle = True)
    model = Net(RelationConfig).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
                          momentum=args['momentum'])
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)
        if epoch < args['epochs']:
            # report intermediate result
            logger.debug('test accuracy %g', test_acc)
            logger.debug('Pipe send intermediate result done.')
        else:
            # report final result
            logger.debug('Final result is %g', test_acc)
            logger.debug('Send final result done.')

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
