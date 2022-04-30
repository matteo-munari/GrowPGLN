import torch
from torch import nn

from models.base_gln import BaseGLNLayer, BaseGLN


class ProtoGLN(BaseGLN):
    def __init__(self, input_dim, outputs_list, n_classes, n_prototypes, prot_bias_std=0.05):
        super(ProtoGLN, self).__init__()
        self.input_dim = input_dim
        self.outputs_list = outputs_list
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes

        inputs_list = [input_dim] + outputs_list[:-1]
        self.layers = nn.ModuleList([ProtoGLNLayer(in_features=in_dim,
                                                  out_features=out_dim,
                                                  n_classes=n_classes,
                                                  side_info_dim=input_dim,
                                                  n_prototypes=n_prototypes,
                                                  prot_bias_std=prot_bias_std)
                                     for in_dim, out_dim in zip(inputs_list, outputs_list)])


class ProtoGLNLayer(BaseGLNLayer):
    def __init__(self, n_prototypes, prot_bias_std=0.05, *args, **kwargs):
        super(ProtoGLNLayer, self).__init__(*args, **kwargs)
        self.n_prototypes = n_prototypes

        # weights and bias init
        self.weights = torch.zeros(self.n_classes * self.out_features * self.n_prototypes, self.in_features)
        self.bias = torch.zeros(self.n_classes * self.out_features * self.n_prototypes)

        self.index_tensor = torch.zeros(self.n_classes * self.out_features, dtype=torch.int64)
        for i in range(self.n_classes * self.out_features):
            self.index_tensor[i] += i * n_prototypes

        # prototypes init
        self.prototypes = torch.normal(0.0, prot_bias_std, size=(self.n_classes, self.out_features, n_prototypes, self.side_info_dim))

    def forward(self, x, side_info):
        """
        Works only with batch size of 1 (online)

        :param x: input of shape [batch_size, in_features, n_classes]
        :param side_info: side information of shape [batch_size, side_info_dim]
        :return:
        """
        # input clipping
        x = torch.sigmoid(x)
        x = torch.logit(x, 0.01)

        # gating computation
        # compute euclidean distance
        dist = (self.prototypes - side_info).pow(2).sum(-1).sqrt()
        gate_tensor = torch.argmin(dist, dim=-1)

        # build gate index
        gate_tensor_new = gate_tensor.view(1, -1)
        self.current_index = gate_tensor_new + self.index_tensor
        # apply to bias and weights
        self.current_bias.data = self.bias[self.current_index[0]]
        bias = self.current_bias.view(self.n_classes, self.out_features)
        self.current_weights.data = self.weights[self.current_index[0], :]
        weights = self.current_weights.view(self.n_classes, self.out_features, self.in_features)

        # Output computation
        output = torch.bmm(weights, x.T) + bias.unsqueeze(2)
        output = output.permute(2, 1, 0)  # dim = [batch_size, out_features, n_classes]

        # prediction clipping
        self.last_output = torch.clip(torch.sigmoid(output), 0.01, 0.99)

        return output

    def propagate_updates(self):
        self.weights.data[self.current_index[0], :] = torch.clip(self.current_weights, -200, 200)  # gradient clipping
        self.bias.data[self.current_index[0]] = torch.clip(self.current_bias, -200, 200)  # gradient clipping
