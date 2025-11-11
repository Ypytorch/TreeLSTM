import torch
import torch.nn as nn

class TreeLSTM(nn.Module):
    def __init__(self, config):
        super(TreeLSTM, self).__init__()
        self.in_dim = config['in_dim']
        if self.in_dim is None:
            raise ValueError('input dimension must be specified')
        self.mem_dim = config.get('mem_dim', 150)
        self.mem_zeros = torch.zeros(self.mem_dim)
        self.train_mode = False

    def forward(self, tree, inputs):
        raise NotImplementedError

    def backward(self, tree, inputs, grad):
        raise NotImplementedError

    def training(self):
        self.train_mode = True

    def evaluate(self):
        self.train_mode = False

    def allocate_module(self, tree, module):
        modules = getattr(self, f'{module}s')
        if len(modules) == 0:
            setattr(tree, module, getattr(self, f'new_{module}')())
        else:
            setattr(tree, module, modules.pop())
        if self.train_mode:
            getattr(tree, module).train()
        else:
            getattr(tree, module).eval()

    def free_module(self, tree, module):
        if hasattr(tree, module):
            getattr(self, f'{module}s').append(getattr(tree, module))
            delattr(tree, module)
