"""
GCN模型：把两层图卷积层组合成一个神经网络
"""

import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution       # 导入自定义卷积层


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        # super(GCN, self).__init__()
        super().__init__()

        # 第一层卷积层，输入节点特征维度1433，输出隐藏层特征维度16，聚合邻居节点的信息
        self.gc1 = GraphConvolution(nfeat, nhid)
        # 第二层卷积层，输入隐藏层特征维度16，输出类别数7
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout      # dropout概率，训练时随机丢弃一部分神经元，防止过拟合

    # 前向传播
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)  # 对隐藏层特征进行dropout，只在训练时使用
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)      # 对概率取对数，输出的是log概率
