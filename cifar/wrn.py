from __future__ import absolute_import, print_function
import argparse
import json
import sys
import os.path as osp

try:
    from time import perf_counter as tic
except ImportError:
    from time import time as tic

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import init, DataParallel
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

this_dir = osp.dirname(osp.abspath(__file__))
if osp.join(this_dir, '..') not in sys.path:
    sys.path.insert(0, osp.join(this_dir, '..'))

from evaluation_metrics import accuracy
from utils.logging import Logger
from utils.meters import AverageMeter
from utils.osutils import mkdir_if_missing
from utils.serialization import load_checkpoint, save_checkpoint


def margin_loss(x):
    mat = x.data.permute(1,0,2,3).contiguous()
    mat = mat.view(mat.size(0), -1)
    margin = mat.std(dim=1) * 0.6
    margin = Variable(margin.view(1, -1, 1, 1).expand_as(x))
    mask = (x >= 0).float() * 2 - 1
    loss = (margin - mask * x).clamp(min=0).mean()
    return loss


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout_rate=0):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, padding=0, bias=False)

    def forward(self, x):
        loss = 0
        if self.shortcut is not None:
            x = self.relu1(self.bn1(x))
            res = x
            x = self.shortcut(x)
            loss = loss + margin_loss(x)
        else:
            res = self.relu1(self.bn1(x))
        res = self.conv1(res)
        loss = loss + margin_loss(res)
        res = self.relu2(self.bn2(res))
        if self.dropout is not None:
            res = self.dropout(res)
        res = self.conv2(res)
        loss = loss + margin_loss(res)
        return x + res, loss


class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, stride,
                 dropout_rate=0):
        super(NetworkBlock, self).__init__()
        layers = nn.ModuleList()
        layers.append(BasicBlock(in_channels, out_channels, stride,
                                 dropout_rate=dropout_rate))
        for _ in range(1, num_layers):
            layers.append(BasicBlock(out_channels, out_channels, 1,
                                     dropout_rate=dropout_rate))
        self.layers = layers

    def forward(self, x):
        total_loss = 0
        for layer in self.layers:
            x, loss = layer(x)
            total_loss = total_loss + loss
        return x, total_loss


class WideResNet(nn.Module):
    def __init__(self, depth, width, num_classes, dropout_rate=0):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'Depth should be 6n+4'
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, 16, 16 * width, 1, dropout_rate)
        self.block2 = NetworkBlock(n, 16 * width, 32 * width, 2, dropout_rate)
        self.block3 = NetworkBlock(n, 32 * width, 64 * width, 2, dropout_rate)
        self.bn = nn.BatchNorm2d(64 * width)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(64 * width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        total_loss = margin_loss(x)
        x, loss = self.block1(x)
        total_loss = total_loss + loss
        x, loss = self.block2(x)
        total_loss = total_loss + loss
        x, loss = self.block3(x)
        total_loss = total_loss + loss
        x = self.relu(self.bn(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, total_loss


def get_datasets(dataset, data_dir):
    if dataset == 'cifar10':
        cls = CIFAR10
        num_classes = 10
    else:
        cls = CIFAR100
        num_classes = 100
    normalizer = T.Normalize(mean=[0.491, 0.482, 0.446],
                             std=[0.247, 0.244, 0.262])
    train_dataset = cls(data_dir, download=True, train=True,
                        transform=T.Compose([
                            T.RandomHorizontalFlip(),
                            T.RandomCrop(32, 4),
                            T.ToTensor(),
                            normalizer,
                        ]))
    test_dataset = cls(data_dir, download=True, train=False,
                       transform=T.Compose([
                           T.ToTensor(),
                           normalizer,
                       ]))
    return train_dataset, test_dataset, num_classes


def evaluate(epoch, data_loader, model, criterion, cpu_only=False):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prec1_meter = AverageMeter()

    end = tic()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(tic() - end)

        if not cpu_only:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)

        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        prec1, = accuracy(outputs.data, targets.data, topk=(1,))

        batch_size = inputs.size(0)
        loss_meter.update(loss.data[0], batch_size)
        prec1_meter.update(prec1[0], batch_size)

        batch_time.update(tic() - end)
        end = tic()

        print('Eval: [{}][{}/{}]  '
              'Time {:.3f} ({:.3f})  '
              'Data {:.3f} ({:.3f})  '
              'Loss {:.3f} ({:.3f})  '
              'Prec1 {:.3f} ({:.3f})  '
              .format(epoch, i, len(data_loader),
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg,
                      loss_meter.val, loss_meter.avg,
                      prec1_meter.val, prec1_meter.avg))

    return prec1_meter.avg


def train(epoch, data_loader, model, criterion, optimizer, margin_loss_weight, cpu_only=False):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prec1_meter = AverageMeter()

    end = tic()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(tic() - end)

        if not cpu_only:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)

        outputs, margin_loss = model(inputs)
        margin_loss = margin_loss.mean()
        loss = criterion(outputs, targets)
        prec1, = accuracy(outputs.data, targets.data, topk=(1,))

        batch_size = inputs.size(0)
        loss_meter.update(loss.data[0], batch_size)
        prec1_meter.update(prec1[0], batch_size)

        optimizer.zero_grad()
        loss = loss + margin_loss_weight * margin_loss
        loss.backward()
        optimizer.step()

        batch_time.update(tic() - end)
        end = tic()

        print('Epoch: [{}][{}/{}]  '
              'Time {:.3f} ({:.3f})  '
              'Data {:.3f} ({:.3f})  '
              'Loss {:.3f} ({:.3f})  '
              'Prec1 {:.3f} ({:.3f})  '
              .format(epoch, i, len(data_loader),
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg,
                      loss_meter.val, loss_meter.avg,
                      prec1_meter.val, prec1_meter.avg))


def main(args):
    if not torch.cuda.is_available():
        args.cpu_only = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.cpu_only:
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True

    # Logs directory
    mkdir_if_missing(args.logs_dir)
    if not args.eval_only:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Data
    train_dataset, test_dataset, num_classes = get_datasets(args.dataset,
                                                            args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.workers, shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, shuffle=False,
                             pin_memory=True)

    # Model
    model = WideResNet(args.depth, args.width, num_classes,
                       dropout_rate=args.dropout)
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_prec1 = 0, 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_prec1 = checkpoint['best_prec1']
        print("=> Load from {}, start epoch {}, best prec1 {:.2%}"
              .format(args.resume, start_epoch, best_prec1))

    if not args.cpu_only:
        model = DataParallel(model).cuda()
        criterion = criterion.cuda()

    # Optimizer
    if args.optim_method == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, nesterov=True,
                        momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)

    # Evaluation only
    if args.eval_only:
        evaluate(start_epoch - 1, test_loader, model, criterion, args.cpu_only)
        return

    # Training
    epoch_steps = json.loads(args.epoch_steps)[::-1]
    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        power = 0
        for i, step in enumerate(epoch_steps):
            if epoch >= step:
                power = len(epoch_steps) - i
        lr = args.lr * (args.lr_decay_ratio ** power)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Training
        train(epoch, train_loader, model, criterion, optimizer, args.margin_loss_weight, args.cpu_only)
        prec1 = evaluate(epoch, test_loader, model, criterion, args.cpu_only)
        is_best = prec1 > best_prec1
        best_prec1 = max(best_prec1, prec1)

        # Save checkpoint
        checkpoint = {'epoch': epoch, 'best_prec1': best_prec1}
        if args.cpu_only:
            checkpoint['model'] = model.state_dict()
        else:
            checkpoint['model'] = model.module.state_dict()
        save_checkpoint(checkpoint, is_best,
                        osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {}  Prec1: {:.2%}  Best: {:.2%}{}\n'.
              format(epoch, prec1, best_prec1, ' *' if is_best else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Wide Residual Net (WRN) on CIFAR')
    # Common
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    # Model
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--margin-loss-weight', type=float, default=0.01)
    # Training
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--eval-only', action='store_true')
    # Optimization
    parser.add_argument('--optim-method', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epoch-steps', type=str, default='[60,120,160]')
    parser.add_argument('--lr-decay-ratio', type=float, default=0.2)
    # Misc
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(this_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(this_dir, 'logs'))
    args = parser.parse_args()
    main(args)
