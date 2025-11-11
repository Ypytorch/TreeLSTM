import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch

class CNNmodel(nn.Module):
    def __init__(self, dropout_rate=0.5, input_size=4000, num_classes=128):
        super(CNNmodel, self).__init__()

        # 降低卷积层的通道数
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3)  # 输入通道1，输出通道4
        self.conv2 = nn.Conv1d(4, 8, kernel_size=3)  # 输入通道4，输出通道8

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 重新计算输出尺寸
        conv1_out_size = (input_size - 3) + 1  # Conv1
        pool1_out_size = conv1_out_size // 2  # Pool1
        conv2_out_size = (pool1_out_size - 3) + 1  # Conv2
        pool2_out_size = conv2_out_size // 2  # Pool2

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * pool2_out_size, 256)  # 调整为 8 个通道
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # Shape: (batch_size, 4, 3998)
        x = self.pool(x)   # Shape: (batch_size, 4, 1999)

        x = self.conv2(x)  # Shape: (batch_size, 8, 1997)
        x = self.dropout(x)
        x = self.pool(x)   # Shape: (batch_size, 8, 998)
        x = self.flatten(x)  # Shape: (batch_size, 8 * 998)

        x = self.fc1(x)  # Shape: (batch_size, 300)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Shape: (batch_size, num_classes)
        return x


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim=2050, output_dim=256, dropout=0.3, hidden_dim=512):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout( dropout )
        self.fc2 = nn.Linear( 512,  256)
        self.fc3 = nn.Linear( 256, 256 )
        # self.fc2 = nn.Linear( 1000,  500)
        # self.fc3 = nn.Linear( 500, 250 )
        # self.fc4 = nn.Linear( 250, 1 )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.float()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout( x )
        x = self.fc2( x )
        x = self.relu1(x)
        x = self.fc3(x)
        # x = self.relu1(x)
        # x = self.fc4(x)
        # x = self.sigmoid( x )

        return x

class FeedForwardNN1(nn.Module):
    def __init__(self, input_dim=2050, output_dim=256, dropout=0.3, hidden_dim=512):
        super(FeedForwardNN1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout( dropout )
        # self.fc2 = nn.Linear( 1000,  500)
        # self.fc3 = nn.Linear( 500, 2 )
        self.fc2 = nn.Linear( 1000,  500)
        self.fc3 = nn.Linear( 500, 250 )
        self.fc4 = nn.Linear( 250, 1 )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.float()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout( x )
        x = self.fc2( x )
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.relu1(x)
        x = self.fc4(x)
        x = self.sigmoid( x )

        return x

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim = 10, hidden_dim = 256, output_dim = 1322):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # 第一层 GCN
        self.conv2 = GCNConv(hidden_dim, 2000)  # 第二层 GCN
        self.global_pool = global_mean_pool  # 全局池化
        self.fc = torch.nn.Linear(2000, output_dim)  # 全局特征输出

        # 可选的改进：dropout和batch normalization
        self.dropout = torch.nn.Dropout(p=0.5)  # dropout层
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(2000)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)  # GCN 第一层
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.conv2(x, edge_index)  # GCN 第二层
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.global_pool(x, batch)  # 全局池化，提取全局特征
        x = self.fc(x)  # 输出全局特征

        return x

class SVMModel(torch.nn.Module):
    def __init__(self, input_dim = 10, hidden_dim = 256, output_dim = 1375):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # 第一层 GCN
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # 第二层 GCN
        self.global_pool = global_mean_pool  # 全局池化
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # 全局特征输出

        # 可选的改进：dropout和batch normalization
        self.dropout = torch.nn.Dropout(p=0.5)  # dropout层
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)  # GCN 第一层
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.conv2(x, edge_index)  # GCN 第二层
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.global_pool(x, batch)  # 全局池化，提取全局特征
        x = self.fc(x)  # 输出全局特征

        return x


def eval_test(model, data):


        x = model(data)# 输出全局特征
        # x = model(torch.tensor(data[0]).unsqueeze(0).unsqueeze(0).float())# 输出全局特征
        x = nn.Sigmoid()(x)

        # return x,data[1]
        return x

