import torch
from torch.nn.functional import one_hot


class Flatten(torch.nn.Module):
    def __call__(self, tensor):
        return tensor.reshape(-1)


class OneHot(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, label):
        return one_hot(torch.Tensor([label]).to(torch.int64), self.n_classes).to(torch.float32).reshape(-1)