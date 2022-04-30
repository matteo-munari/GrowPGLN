import torch
from torch import nn
from torch.nn.functional import normalize

from dev_models.base_gln import BaseGLN, BaseGLNLayer


class HypGLN(BaseGLN):
    def __init__(self, input_dim, outputs_list, n_classes, n_hyperplanes, hyp_bias_std=0.05):
        super(HypGLN, self).__init__()
        self.input_dim = input_dim
        self.outputs_list = outputs_list
        self.n_classes = n_classes
        self.n_hyperplanes = n_hyperplanes

        inputs_list = [input_dim] + outputs_list[:-1]
        self.layers = nn.ModuleList([HypGLNLayer(in_features=in_dim,
                                                 out_features=out_dim,
                                                 n_classes=n_classes,
                                                 side_info_dim=input_dim,
                                                 n_hyperplanes=n_hyperplanes,
                                                 hyp_bias_std=hyp_bias_std)
                                     for in_dim, out_dim in zip(inputs_list, outputs_list)])


class HypGLNLayer(BaseGLNLayer):
    def __init__(self, n_hyperplanes, hyp_bias_std=0.05, *args, **kargs):
        super(HypGLNLayer, self).__init__(*args, **kargs)
        self.n_hyperplanes = n_hyperplanes

        # weights and bias init
        self.weights = torch.zeros(self.n_classes * self.out_features * 2**self.n_hyperplanes, self.in_features)
        self.bias = torch.zeros(self.n_classes * self.out_features * 2**self.n_hyperplanes)

        self.index_tensor = torch.zeros(self.n_classes * self.out_features, dtype=torch.int64)
        for i in range(self.n_classes * self.out_features):
            self.index_tensor[i] += i * 2**n_hyperplanes

        # hyperplanes init
        self.hyp_w = torch.normal(0.0, 1.0, size=(self.n_classes, self.out_features, n_hyperplanes, self.side_info_dim))
        self.hyp_w = normalize(self.hyp_w, dim=2)
        self.hyp_b = torch.normal(0.0, hyp_bias_std, size=(self.n_classes, self.out_features, n_hyperplanes, 1))

        self.bit_importance = torch.zeros(n_hyperplanes)
        for i in range(n_hyperplanes):
            self.bit_importance[i] = 2**(n_hyperplanes - i - 1)  # [2**n_hyp, ... , 1]

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
        projection = torch.matmul(self.hyp_w, side_info.T)
        projection = torch.gt(projection, self.hyp_b).to(torch.float32)
        projection = torch.reshape(projection, (self.n_classes, self.out_features, -1))

        # build gate index
        projection = torch.matmul(projection, self.bit_importance)
        gating_tensor = projection.view(1, -1)
        gating_tensor += self.index_tensor
        self.current_index = gating_tensor.to(torch.int64)

        self.current_bias.data = self.bias.data[self.current_index[0]]
        bias = self.current_bias.view(self.n_classes, self.out_features)

        self.current_weights.data = self.weights[self.current_index[0], :]
        weights = self.current_weights.view(self.n_classes, self.out_features, self.in_features)

        # output computation
        output = torch.bmm(weights, x.T) + bias.unsqueeze(2)
        output = output.permute(2, 1, 0)  # dim = [batch_size, out_features, n_classes]

        # prediction clipping
        self.last_output = torch.clip(torch.sigmoid(output), 0.01, 0.99)

        return output

    def propagate_updates(self):
        self.weights.data[self.current_index[0], :] = torch.clip(self.current_weights, -200, 200)  # gradient clipping
        self.bias.data[self.current_index[0]] = torch.clip(self.current_bias, -200, 200)  # gradient clipping
