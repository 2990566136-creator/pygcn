"""
1、读取Cora数据集
2、把标签转成one-hot
3、构造并归一化邻接矩阵和特征矩阵
4、计算准确率、把scipy稀疏矩阵转成torch稀疏张量
"""

import numpy as np
import scipy.sparse as sp       # 导入稀疏矩阵模块
import torch


def encode_onehot(labels):          # 字符串类别标签转成one-hot编码
    classes = set(labels)           # 获取不重复的标签类别
    # 为每个类别创建一个one-hot向量
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in      # 生成单位矩阵，然后会给每一个类别分配一行one-hot
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),       # 把每个原始标签替换成它对应的 one-hot 向量
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):                # 数据加载函数
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # 读取节点特征和标签，读取cora.content文件
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)        # 取出节点特征，构造稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])      # 取最后一列作为类别标签，并转成 one-hot

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)       # 取出每个节点的原始ID
    idx_map = {j: i for i, j in enumerate(idx)}                     # 创建一个字典，键为原始ID，为了后面构造邻接矩阵
    
    # 读取节点关系，读取cora.cites文件
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),     # 原始id组成的边列表
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),     # 把每个原始论文ID转换成连续编号
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 构造邻接矩阵，使用稀疏矩阵表示
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix，无向图，对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)     # 有向图转成无向图

    features = normalize(features)          # 特征归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))     # 给邻接矩阵加自环

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))      # 把scipy稀疏矩阵转成dense矩阵，再转成PyTorch张量
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)     # 把scipy稀疏矩阵转成PyTorch张量

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# 定义归一化函数，按行归一化
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()      # 计算每一行和的倒数
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)                 # 倒数向量变成对角矩阵
    mx = r_mat_inv.dot(mx)                      # 用对角矩阵左乘原矩阵，实现行归一化
    return mx


# 定义准确率计算函数
def accuracy(output, labels):
    """Calculate accuracy."""
    preds = output.max(1)[1].type_as(labels)        # 对每个节点，取预测分数最大的类别作为预测结果 
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# 定义函数，把 scipy 稀疏矩阵转成 PyTorch 稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))     # 构造 PyTorch 稀疏张量需要的索引
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
