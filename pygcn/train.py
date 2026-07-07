"""
解析训练参数
-> 加载 Cora 数据
-> 创建 GCN 模型
-> 创建优化器
-> 训练模型
-> 验证模型
-> 测试模型
"""


from __future__ import division
from __future__ import print_function

import time
import argparse             # 解析命令行参数
import numpy as np          

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy     # 加载Cora数据，计算准确率
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()            # 创建命令行参数解析器
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,             # 权重衰减，L2 正则化，限制参数过大，缓解过拟合
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()          # 读取命令行参数，保存到args对象中
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data，加载数据
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],        # 输入特征维度，1433
            nhid=args.hidden,
            nclass=labels.max().item() + 1, # 类别数量
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),              # 更新GCN中所有可训练参数，weight和bias
                       lr=args.lr, weight_decay=args.weight_decay)

# 如有GPU，把模型和数据转到GPU上
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()       # 清空上一轮的梯度
    output = model(features, adj)       # 前向传播
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])       # 计算训练损失，只取训练集的节点，半监督节点分类的做法
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()        # 优化器根据梯度更新模型参数

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model，主循环训练
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
