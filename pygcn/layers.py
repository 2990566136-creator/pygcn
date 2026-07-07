"""
GCN的一层图卷积：输入为邻接矩阵和特征矩阵，输出为特征矩阵。
核心公式：H' = A X W + b
"""

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):       # 输入特征维度，输出特征维度
        # super(GraphConvolution, self).__init__()
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))       # 权重矩阵，可训练参数
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))      # 加到每个节点的输出特征
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()             # 调用参数初始化函数，给权重和偏置初始随机值

    # 参数初始化函数
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))      # 计算初始化范围
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向传播函数
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)      # 普通矩阵乘法，support = XW，节点数和输出特征维度，先对每个节点自己的做线性变换
        output = torch.spmm(adj, support)           # 稀疏矩阵乘dense矩阵，output = AXW，最后也是[节点数，输出特征维度]，聚合邻居节点的信息到当前节点上
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 定义对象的字符串表示
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
