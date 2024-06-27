import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
from tqdm import tqdm

from dataset.datasets import C100Dataset, DataLoader
from models.models import *
from loss import TruncatedLoss

parser = argparse.ArgumentParser(
    description='PyTorch TruncatedLoss')

parser.add_argument('--decay', default=1e-4, type=float,
                    help='weight decay (default=1e-4)')
parser.add_argument('--lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--batch-size', '-b', default=5,
                    type=int, help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--gamma', type = float, default = 0.1)
parser.add_argument('--schedule', nargs='+', type=int)
parser.add_argument('--sess', default='default')

args = parser.parse_args()

def main():
    train_dataset, val_dataset = C100Dataset('/content/Yonsei-vnl-coding-assignment-vision-48hrs/dataset/data/cifar100_nl.csv','/content/Yonsei-vnl-coding-assignment-vision-48hrs/dataset/data/cifar100_nl_test.csv').getDataset()
    trainloader = DataLoader(train_dataset, args.batch_size)
    valloader = DataLoader(val_dataset, args.batch_size)

    num_classes = 100

    net = ResNet18(num_classes)

    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    logname = result_folder + net.__class__.__name__ + '_' + args.sess + '.csv'

    net.cuda()
    net = torch.nn.DataParallel(net)

    criterion = TruncatedLoss(trainset_size=len(train_dataset[0])).cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

    for epoch in range(args.epochs):

        train_loss, train_acc = train(epoch, trainloader, net, criterion, optimizer)
        test_loss, test_acc = test(epoch, valloader, net, criterion)

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

    scheduler.step()

def train(epoch, trainloader, net, criterion, optimizer):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # print(len(trainloader))
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        tensor_inputs, tensor_targets = torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
        tensor_inputs, tensor_targets = tensor_inputs.cuda(), tensor_targets.cuda()

        # print(inputs.shape, targets.shape)

        outputs = net(tensor_inputs)
        loss = criterion(outputs, tensor_targets, indexes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += len(targets)

        correct += predicted.eq(torch.argmax(tensor_targets, dim=1).data).cpu().sum()
        correct = correct.item()

        # print(targets)
        # print(total)
        print(f'Epoch: {epoch} | Batch: {batch_idx} / {len(trainloader)} | Loss {loss.item()} | Acc: {100. * correct / total} | Correct : {correct} | Total : {total}')

    print(train_loss / len(trainloader))
    print(100. * correct / total)

    return (train_loss / len(trainloader), 100. * correct / total)

def test(epoch, testloader, net, criterion):
    test_loss = 0
    correct = 0
    total = 0

    batch_idx = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            inputs, targets = torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets, indexes)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            if batch_idx != len(testloader) - 1:
                total += args.batch_size
            else:
                total = 9999
            correct += predicted.eq(torch.argmax(targets, dim=1).data).cpu().sum()
            correct = correct.item()

            print(f'Epoch: {epoch} | Batch: {batch_idx} / {len(testloader)} | Loss {loss.item()} | Acc: {100. * correct / total} | Correct : {correct} | Total : {total}')

    return (test_loss / len(testloader), 100 * correct / total)

if __name__ == '__main__':
    main()