import torch
from torch import nn

from models.base_gln import BaseGLN, BaseGLNLayer


class GrowPGLN(BaseGLN):
    def __init__(self, input_dim, outputs_list, n_classes, d, dropout=0.):
        super(GrowPGLN, self).__init__()
        self.input_dim = input_dim
        self.outputs_list = outputs_list
        self.n_classes = n_classes

        # grow hyperparameter
        self.d = d

        inputs_list = [input_dim] + outputs_list[:-1]
        in_drop_list = [0.] + [dropout] * (len(outputs_list) - 1)
        out_drop_list = [dropout] * (len(outputs_list) - 1) + [0.]
        self.layers = nn.ModuleList([GrowPGLNLayer(d=d,
                                                     in_dropout=in_drop,
                                                     out_dropout=out_drop,
                                                     in_features=in_dim,
                                                     out_features=out_dim,
                                                     n_classes=n_classes,
                                                     side_info_dim=input_dim)
                                     for in_dim, out_dim, in_drop, out_drop
                                     in zip(inputs_list, outputs_list, in_drop_list, out_drop_list)])

    def remove_unused_prototypes(self, assignments_threshold):
        for layer in self.layers:
            layer.remove_unused_prototypes(assignments_threshold)


class GrowPGLNLayer(BaseGLNLayer):
    def __init__(self, d, in_dropout=0., out_dropout=0., *args,
                 **kwargs):
        super(GrowPGLNLayer, self).__init__(*args, **kwargs)
        self.d = d
        self.in_drop = in_dropout
        self.out_drop = out_dropout

        self.idx1 = torch.tensor([cl for cl in range(self.n_classes) for _ in range(self.out_features)])
        self.idx2 = torch.tensor([i for i in range(self.out_features)] * self.n_classes)

        # define weights and bias
        init_size = 1
        self.weights = torch.zeros(self.n_classes * self.out_features * init_size, self.in_features)
        self.bias = torch.zeros(self.n_classes * self.out_features * init_size, 1)
        self.index_tensor = torch.tensor([i * init_size for i in range(self.n_classes * self.out_features)],
                                         dtype=torch.int64)

        # init prototypes
        self.max_regions = init_size
        self.prototypes = torch.full(size=(1, self.side_info_dim), fill_value=float('inf'))
        self.neuron_proto_count = torch.zeros(size=(self.n_classes, self.out_features), dtype=torch.int64)
        self.neuron_proto_map = torch.zeros(size=(self.n_classes, self.out_features, init_size), dtype=torch.int64)
        self.neuron_proto_map_assignments = torch.zeros(size=(self.n_classes, self.out_features, init_size), dtype=torch.int64)

        self.output_mask = None
        self.current_distances = None

    def _compute_distances(self, side_info):
        # compute distance between side info and prototypes
        dist = (self.prototypes - side_info).pow(2).sum(-1).sqrt()

        # build distance tensor selecting the prototypes assigned to each neuron
        neuron_index = self.neuron_proto_map.reshape(-1)
        self.current_distances = dist[neuron_index].reshape(self.n_classes, self.out_features, -1)

    def _expand_weights(self):
        new_weight = torch.zeros(self.n_classes, self.out_features, 1, self.in_features)
        new_bias = torch.zeros(self.n_classes, self.out_features, 1, 1)

        # expand weights and bias
        if self.weights is None or self.bias is None:
            self.weights = new_weight
            self.bias = new_bias
        else:
            self.weights = torch.cat((self.weights.reshape(self.n_classes, self.out_features, -1, self.in_features), new_weight), dim=2)
            self.bias = torch.cat((self.bias.reshape(self.n_classes, self.out_features, -1, 1), new_bias), dim=2)

        self.weights = self.weights.reshape(-1, self.in_features)
        self.bias = self.bias.reshape(-1, 1)
        self.index_tensor = torch.tensor([i * self.max_regions for i in range(self.n_classes * self.out_features)], dtype=torch.int64)

    def _add_prototype(self, prototype, index):
        # add new prototype to prototypes
        appending_position = self.prototypes.shape[0]
        self.prototypes = torch.cat((self.prototypes, prototype), dim=0)

        # if needed, expand assignment map and weights
        if torch.max(self.neuron_proto_count[index]) >= self.max_regions:
            new_map_row = torch.zeros(size=(self.n_classes, self.out_features, 1), dtype=torch.int64)
            new_map_assignment_row = torch.zeros(size=(self.n_classes, self.out_features, 1), dtype=torch.int64)
            self.neuron_proto_map = torch.cat((self.neuron_proto_map, new_map_row), dim=-1)
            self.neuron_proto_map_assignments = torch.cat((self.neuron_proto_map_assignments, new_map_assignment_row), dim=-1)
            self.max_regions += 1
            self._expand_weights()

        # update neuron map
        new_index = index + (self.neuron_proto_count[index].reshape(-1),)
        self.neuron_proto_map[new_index] = appending_position
        self.neuron_proto_count[index] += 1

    def remove_unused_prototypes(self, n_assignments):
        remove_mask = torch.le(self.neuron_proto_map_assignments, n_assignments)
        remove_index = torch.nonzero(remove_mask, as_tuple=True)

        self.neuron_proto_map[remove_index] = 0
        self.neuron_proto_map_assignments[remove_index] = 0

    @staticmethod
    def _mask_index(mask, index):
        relevant_mask = mask[index]
        masked_index = tuple([idx[relevant_mask] for idx in index])
        return masked_index

    def _grow(self, side_info, mask):
        side_info = side_info.reshape(1, -1)

        # compute nearest neighbors
        nn_dist, nn_index = torch.kthvalue(self.current_distances, k=1, dim=-1)

        # compute indexes of neuron to update, based on d
        far_from_nn = torch.gt(nn_dist, self.d)
        far_from_nn_idx = torch.nonzero(far_from_nn, as_tuple=True)

        # update neurons
        masked_idx = self._mask_index(mask, far_from_nn_idx)
        if masked_idx[0].numel() > 0:
            self._add_prototype(side_info, masked_idx)

    def forward(self, x, side_info):
        if self.training:
            self._compute_distances(side_info[0])

            # compute dropout binary output mask
            mask = torch.bernoulli(
                torch.full(size=(x.shape[0], self.out_features, self.n_classes), fill_value=1. - self.out_drop))
            self.output_mask = mask

            # grow prototypes
            self._grow(side_info[0], torch.gt(mask[0], 0).permute(1, 0))

        # recompute distances on new set of prototypes
        self._compute_distances(side_info[0])

        # gating computation
        gate_tensor = torch.argmin(self.current_distances, dim=-1)

        if self.training:  # count training samples assigned to prototypes
            m = torch.gt(self.output_mask[0].permute(1, 0).reshape(-1), 0.0)
            index = (self.idx1[m], self.idx2[m], gate_tensor.view(-1)[m])
            self.neuron_proto_map_assignments[index] += 1

        # build gate index
        gate_tensor_new = gate_tensor.reshape(-1)
        self.current_index = gate_tensor_new + self.index_tensor

        # apply gating to bias and weights
        self.current_bias.data = self.bias[self.current_index, :]
        bias = self.current_bias.reshape(self.n_classes, self.out_features)
        self.current_weights.data = self.weights[self.current_index, :]
        weights = self.current_weights.reshape(self.n_classes, self.out_features, self.in_features)

        # output computation
        output = torch.bmm(weights, x.T) + bias.unsqueeze(2)
        output = output.permute(2, 1, 0)  # dim = [batch_size, out_features, n_classes]

        self.last_output = torch.sigmoid(output)

        if self.training:
            output = output * self.output_mask
        else:
            output = output * (1. - self.in_drop)  # scale weights for inference

        return output

    def propagate_updates(self):
        m = self.output_mask.permute(0, 2, 1).reshape(-1)  # dim = [n_classes * out_features]
        m = torch.gt(m, 0)
        masked_index = self.current_index[m]

        self.weights[masked_index, :] = self.current_weights[m, :]
        self.bias[masked_index, :] = self.current_bias[m, :]
