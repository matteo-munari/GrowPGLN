import torch
from torch import nn
from torch.nn import Parameter


class BaseGLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = None

    def forward(self, x):
        side_info = x

        x = x.unsqueeze(-1).expand(-1, -1, self.n_classes)

        for layer in self.layers:
            x = layer.forward(x, side_info)

        return torch.sigmoid(x)

    def backward(self, loss_fn, target):
        for layer in self.layers:
            layer.backward(loss_fn, target)

    def propagate_updates(self):
        for layer in self.layers:
            layer.propagate_updates()


# Base multi-class layer for GLN
class BaseGLNLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 n_classes,
                 side_info_dim):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.side_info_dim = side_info_dim

        # weights init
        self.weights = torch.Tensor()
        self.bias = torch.Tensor()

        self.index_tensor = None
        self.current_index = None

        # data for backward/update
        self.last_output = None
        self.current_weights = Parameter()
        self.current_bias = Parameter()

    def forward(self, x, side_info):
        raise NotImplementedError('Implement this method')

    def backward(self, loss_fn, target) -> None:
        """
        Compute and store gradient in .grad attribute of parameters

        :param loss_fn: loss function
        :param target: one_hot encoding of target label, shape [batch_size, n_classes]
        :return: None
        """
        target = target.unsqueeze(1)
        target = target.expand(-1, self.out_features, -1)

        loss = loss_fn(self.last_output.to(torch.float32), target)
        loss.backward(gradient=torch.ones(loss.shape), inputs=(self.current_weights, self.current_bias))

    def propagate_updates(self):
        self.weights.data[self.current_index[0], :] = self.current_weights
        self.bias.data[self.current_index[0]] = self.current_bias
