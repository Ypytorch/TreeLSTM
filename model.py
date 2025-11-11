from .eval_model import CNNmodel,FeedForwardNN
from model.TreeLSTMChild import TreeLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
# class Model( nn.Module ):
#     def __init__(self, config):
#
#         super( Model, self ).__init__()
#         self.device = config.get("device")
#         self.dropout = config.get("dropout")
#         self.input_size = config.get( "input_size" )
#         self.hidden_size = config.get("hidden_size")
#         self.output_size = config.get("output_size")
#         self.tree_lstm = TreeLSTM(32,self.hidden_size,self.device,self.dropout)
#         self.CNN = CNNmodel(self.dropout, self.input_size, self.output_size)
#         self.W1 = nn.Linear( self.hidden_size,self.output_size )
#         self.W2 = nn.Linear( self.output_size * 2,self.output_size )
#         self.dropout = nn.Dropout( self.dropout )
#
#     def forward(self, MS2, trees):
#
#         ms_feat_1 = self.CNN( MS2 )
#         tree_feat_1 = self.W1(self.tree_lstm( trees ))
#         # ms_feat_1 = standardize( ms_feat_1 )
#         # tree_feat_1 = standardize( tree_feat_1 )
#         fused_feat_1 = self.W2(torch.cat([ms_feat_1, tree_feat_1], dim=1))
#
#         return fused_feat_1

class Model(nn.Module):
    def __init__(self, config):

        super( Model, self ).__init__()
        self.device = config.get("device")
        self.dropout = config.get("dropout")
        self.input_size = config.get( "input_size" )
        self.hidden_size = config.get("hidden_size")
        self.output_size = config.get("output_size")

        self.tree_lstm = TreeLSTM(input_size=10, hidden_size=64, device=self.device,
                                  dropout_rate=self.dropout)
        # self.CNN = CNNmodel(dropout_rate=self.dropout, input_size=self.input_size, num_classes=256)

        # self.W1 = nn.Linear(self.hidden_size, self.output_size)
        # self.final_fc = nn.Linear(self.output_size, self.output_size)
        # self.CNN = CNNmodel(dropout_rate=self.dropout, input_size=self.input_size, num_classes=128)
        self.ANN = FeedForwardNN(2000, 256)
        self.W2 = nn.Linear(320, 256)
        self.final_fc = nn.Linear(256, 1322)
        self.dropout1 = nn.Dropout(self.dropout)

    def forward(self, MS2, trees):
        ms_feat_1 = self.ANN(MS2)
        # tree_feat_1 = self.self.tree_lstm(trees)
        tree_feat_1 = self.tree_lstm(trees)

        # 特征融合
        fused_feat_1 = torch.cat([ms_feat_1, tree_feat_1], dim=1)
        fused_feat_1 = self.W2(fused_feat_1)

        # 可选：加入激活函数或者 dropout
        fused_feat_1 = F.relu(fused_feat_1)
        # fused_feat_1 = self.dropout1(fused_feat_1)
        output = self.final_fc(fused_feat_1)

        return output


class ModelTree( nn.Module ):

    def __init__(self, config):
        super( ModelTree, self ).__init__()
        self.device = config.get("device")
        self.dropout = config.get("dropout")
        self.input_size = config.get( "input_size" )
        self.hidden_size = config.get("hidden_size")
        self.output_size = config.get("output_size")
        self.tree_lstm = TreeLSTM(34,self.output_size,self.device,self.dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.output_size, 1845)

    def forward(self, trees):

        tree_feat_1 = self.tree_lstm( trees )
        tree_feat_1 = self.fc( tree_feat_1 )
        # tree_feat_1 = self.sigmoid( tree_feat_1 )

        return tree_feat_1

    def test_eval(self,trees):

        tree_feat_1 = self.tree_lstm(trees)
        tree_feat_1 = self.fc(tree_feat_1)
        tree_feat_1 = self.sigmoid(tree_feat_1)
        return tree_feat_1

class ModelC( nn.Module ):
    def __init__(self, config):

        super( ModelC, self ).__init__()
        self.device = config.get("device")
        self.dropout = config.get("dropout")
        self.input_size = config.get( "input_size" )
        self.hidden_size = config.get("hidden_size")
        # self.output_size = config.get("output_size")
        self.tree_lstm = TreeLSTM(10,64,self.device,self.dropout)
        # self.CNN = CNNmodel(self.dropout, self.input_size, self.hidden_size)
        # self.W1 = nn.Linear( self.hidden_size,self.output_size )
        self.ANN = FeedForwardNN(self.input_size,self.hidden_size)
        self.W2 = nn.Linear( 320,self.hidden_size )
        self.dropout1 = nn.Dropout( 0.1 )
        self.final_fc = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, MS2, trees):

        # ms_feat_1 = self.CNN(MS2)
        ms_feat_1 = self.ANN(MS2)

        tree_feat_1 = self.tree_lstm(trees)
        # tree_feat_1 = self.W1(self.tree_lstm(trees))

        # 特征融合
        fused_feat_1 = torch.cat([ms_feat_1, tree_feat_1], dim=1)
        fused_feat_1 = self.W2(fused_feat_1)

        # 可选：加入激活函数或者 dropout
        fused_feat_1 = F.relu(fused_feat_1)
        fused_feat_1 = self.dropout1(fused_feat_1)

        output = self.final_fc(fused_feat_1)
        return output


class ModelGTree( nn.Module ):

    def __init__(self, config):
        super( ModelGTree, self ).__init__()
        self.device = config.get("device")
        self.dropout = config.get("dropout")
        self.input_size = config.get( "input_size" )
        self.hidden_size = config.get("hidden_size")
        self.output_size = config.get("output_size")
        self.tree_lstm = TreeLSTM(10,64,self.device,self.dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(64, 1322)

    def forward(self, trees):

        tree_feat_1 = self.tree_lstm( trees )
        tree_feat_1 = self.fc( tree_feat_1 )
        # tree_feat_1 = self.sigmoid( tree_feat_1 )

        return tree_feat_1

    def test_eval(self,trees):

        tree_feat_1 = self.tree_lstm(trees)
        tree_feat_1 = self.fc(tree_feat_1)
        tree_feat_1 = self.sigmoid(tree_feat_1)
        return tree_feat_1

class ModelTree1( nn.Module ):
    def __init__(self, config):
        super( ModelTree1, self ).__init__()
        self.device = config.get("device")
        self.dropout = config.get("dropout")
        self.input_size = config.get( "input_size" )
        self.hidden_size = config.get("hidden_size")
        self.output_size = config.get("output_size")
        self.tree_lstm = TreeLSTM(13,self.output_size,self.device,self.dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.output_size, 512)
        self.fc1 = nn.Linear(512, 1162)

    def forward(self, trees):

        tree_feat_1 = self.tree_lstm( trees )
        tree_feat_1 = self.fc( tree_feat_1 )
        tree_feat_1 = self.fc1( tree_feat_1 )
        # tree_feat_1 = self.sigmoid( tree_feat_1 )

        return tree_feat_1

    def test_eval(self,trees):

        tree_feat_1 = self.tree_lstm(trees)
        feature1_norm = tree_feat_1 / tree_feat_1.norm(dim=1, keepdim=True)
        feature2_norm = tree_feat_1 / tree_feat_1.norm(dim=1, keepdim=True)
        smility = torch.matmul(feature1_norm, feature2_norm.T)
        tree_feat_1 = self.fc(tree_feat_1)
        feature1_norm = tree_feat_1 / tree_feat_1.norm(dim=1, keepdim=True)
        feature2_norm = tree_feat_1 / tree_feat_1.norm(dim=1, keepdim=True)
        smility = torch.matmul(feature1_norm, feature2_norm.T)
        tree_feat_1 = self.fc1(tree_feat_1)
        feature1_norm = tree_feat_1 / tree_feat_1.norm(dim=1, keepdim=True)
        feature2_norm = tree_feat_1 / tree_feat_1.norm(dim=1, keepdim=True)
        smility = torch.matmul(feature1_norm, feature2_norm.T)
        # tree_feat_1 = self.sigmoid(tree_feat_1)
        return tree_feat_1

# class FocalLoss(nn.Module):
#
#     def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
#
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, y_pred,y_true ):
#
#         # 确保输入的预测值是介于0和1之间的概率值
#         y_pred = torch.sigmoid(y_pred)
#
#         # 防止log(0)的情况
#         epsilon = 1e-7
#         y_pred = torch.clamp(y_pred, min=epsilon, max=1. - epsilon)
#
#         # 计算交叉熵损失
#         cross_entropy_loss = -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
#
#         # 计算Focal Loss
#         focal_loss =  self.alpha * (1 - y_pred) ** self.gamma * cross_entropy_loss
#
#         # 根据 reduction 参数来处理损失
#         if self.reduction == 'mean':
#             return torch.mean(focal_loss)
#         elif self.reduction == 'sum':
#             return torch.sum(focal_loss)
#         elif self.reduction == 'none':
#             return focal_loss
#         else:
#             raise ValueError("Invalid value for reduction. Choose 'mean', 'sum' or 'none'.")
#     def compute_loss(self, y_pred, y_true ):
#
#
#         # y_pred = torch.sigmoid(y_pred)
#
#         # 防止log(0)的情况
#         epsilon = 1e-7
#         y_pred = torch.clamp(y_pred, min=epsilon, max=1. - epsilon)
#
#         # 计算交叉熵损失
#         cross_entropy_loss = -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
#
#         # 计算Focal Loss
#         focal_loss = self.alpha * (1 - y_pred) ** self.gamma * cross_entropy_loss
#
#         # 根据 reduction 参数来处理损失
#         if self.reduction == 'mean':
#             return torch.mean(focal_loss)
#         elif self.reduction == 'sum':
#             return torch.sum(focal_loss)
#         elif self.reduction == 'none':
#             return focal_loss
#         else:
#             raise ValueError("Invalid value for reduction. Choose 'mean', 'sum' or 'none'.")

class ModelANN( nn.Module ):
    def __init__(self):
        super(ModelANN, self).__init__()
        self.W1 = nn.Linear(128, 64)
        self.W2 = nn.Linear(64, 64)
        self.W3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, feat):
        feat = self.W1(feat)
        feat = self.relu(feat)
        feat = self.W2(feat)
        feat = self.relu(feat)
        feat = self.W3(feat)
        feat = self.sigmoid(feat)

        return feat


def test_eval(model,batch):
    spec, tree, fp = batch
    # spec, fp = batch
    x = model(torch.tensor(spec,dtype=torch.float32).unsqueeze(0),[tree])
    x = nn.Sigmoid()(x)
    return x, fp

def test_eval1(model,batch):
    # spec, tree, fp = batch
    spec, fp = batch
    x = model(torch.tensor(spec,dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    x = nn.Sigmoid()(x)
    return x, fp

def standardize(features):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True) + 1e-6  # 避免除以0
    return (features - mean) / std


import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 平衡正负样本
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction  # 损失的计算方式 ('mean' 或 'sum')

    def forward(self, logits, labels):
        eps = 1e-7  # 防止数值溢出
        preds = torch.sigmoid(logits)  # 将 logits 转换为概率
        loss = -self.alpha * labels * torch.pow(1 - preds, self.gamma) * torch.log(preds + eps) - \
               (1 - self.alpha) * (1 - labels) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps)

        # 根据 reduction 返回
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

