import torch
import torch.nn as nn
import torch.nn.init as init

class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.U_f = nn.Conv1d( hidden_size, hidden_size, 1 )
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_u = nn.Linear(input_size, hidden_size)
        self.U_u = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.fc = nn.Linear(input_size, hidden_size, bias=False)
        # self.dropout = nn.Dropout( dropout_rate )

    def forward(self, x, child_c, child_h):
        h_sum = torch.sum(child_h, dim=0)
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
        f = torch.sigmoid(self.W_f(x) + self.U_f(child_h))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
        u = torch.tanh(self.W_u(x) + self.U_u(h_sum))
        c = i * u + torch.sum(f * child_c, dim=0)
        h = o * torch.tanh(c)
        # h = self.dropout( h )
        return c, h

class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device,dropout_rate=0.5):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.cell = TreeLSTMCell(input_size, hidden_size, dropout_rate)
        self.dropout = nn.Dropout( dropout_rate )

    def forward(self, batch_trees):
        global_features = []
        total_loss = 0
        for tree in batch_trees:
            _, tree_embedding = self.forward_tree( tree.root )
            global_features.append( tree_embedding )
        global_features = torch.stack( global_features,dim=0 )
        # global_features = self.dropout( global_features )
        return global_features

    def forward_tree(self, tree_node):
        if tree_node.is_leaf():
            child_c, child_h = self.init_hidden_state(torch.tensor(tree_node.input, dtype=torch.float).to(self.device))
        else:
            child_c, child_h = zip(*[self.forward_tree(child) for child in tree_node.children])
            child_c, child_h = torch.stack(child_c), torch.stack(child_h)
        c, h = self.cell(torch.tensor(tree_node.input, dtype=torch.float).to(self.device), child_c, child_h)
        return c, h


    def init_hidden_state(self, x):
        h = torch.zeros(1, self.hidden_size).to(x.device)
        c = torch.zeros(1, self.hidden_size).to(x.device)
        init.xavier_normal_(h)  # 使用 Xavier 正态分布初始化
        init.xavier_normal_(c)  # 使用 Xavier 正态分布初始化
        return (h, c)


