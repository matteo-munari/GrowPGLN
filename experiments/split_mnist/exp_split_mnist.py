import argparse
import os
from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torch import nn

from utils.loggers import Logger
from models.hyperplanes_gln import HypGLN
from models.grow_pgln import GrowPGLN
from models.prototypes_gln import ProtoGLN

from utils.mean_std_estimators import MeanStdEstimator
from utils.transforms import Flatten
from torch.nn.functional import one_hot


def build_config(main_args):
    config = dict()
    config['exp_seed'] = main_args.exp_seed
    config['gln_type'] = main_args.gln_type
    config['model_arch'] = main_args.model_arch
    config['lr'] = main_args.lr
    config['norm'] = main_args.norm
    config['log_dir'] = main_args.log_dir

    if config['gln_type'] == 'hyperplanes':
        config['n_hyperplanes'] = main_args.n_hyperplanes
    if config['gln_type'] == 'prototypes':
        config['n_prototypes'] = main_args.n_prototypes
        config['init'] = main_args.init
        config['prot_bias_std'] = main_args.prot_bias_std
    if config['gln_type'] == 'growing':
        config['d'] = main_args.sigma
        assert 0.0 <= main_args.dropout_rate <= 1.0
        config['dropout_rate'] = main_args.dropout_rate
        if main_args.removal is not None:
            config['removal'] = True
        else:
            config['removal'] = False

    return config


def sample(train_data, shapes, norm='none', proj_matrix=None):
    loader = DataLoader(train_data, batch_size=train_data.data.shape[0], shuffle=False)
    data, labels = next(iter(loader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    if norm == 'online':
        estimator = MeanStdEstimator()
        tasks_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        for tasks in tasks_list:
            for i in range(data.shape[0]):
                if labels[i] in tasks:
                    mean, std = estimator(data[i])
                    data[i] = torch.div(data[i] - mean, std + 1.)
                if (i+1) % 10000 == 0:
                    print('processed', i+1, 'examples')
    elif norm == 'feature':
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        data = torch.div(data - mean, 1. + std)
    elif norm == 'global':
        mean = torch.mean(data)
        std = torch.std(data)
        data = torch.div(data - mean, 1. + std)
    elif norm == 'rndproj':
        data = torch.matmul(data, proj_matrix)

    samples = []
    for shape in shapes:
        print(shape)
        idx = torch.randint(0, data.shape[0], size=(shape[0] * shape[1] * shape[2],))
        sample = data[idx].view(shape)
        samples.append(sample)
    return samples


if __name__ == '__main__':
    # Parse main args
    # experiment setup
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--exp-seed', type=int, default=42)
    base_parser.add_argument('--log-dir', type=str, required=True)
    # model common parameters
    base_parser.add_argument('--model-arch', metavar='layer_neurons', type=int, nargs='+', required=True)
    base_parser.add_argument('--lr', type=float, default=0.01)
    base_parser.add_argument('--norm', type=str, default='online', choices=['online', 'global', 'feature', 'none'])

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='gln_type')
    # hyperplanes specific parameters
    hyp_parser = subparsers.add_parser('hyperplanes', parents=[base_parser])
    hyp_parser.add_argument('--n-hyperplanes', type=int, required=True)
    # prototypes specific parameters
    prot_parser = subparsers.add_parser('prototypes', parents=[base_parser])
    prot_parser.add_argument('--n-prototypes', type=int)
    prot_parser.add_argument('--init', type=str, default='random', choices=['random', 'dataset'])
    prot_parser.add_argument('--prot-bias-std', type=float, default=0.05)
    # inc growing specific parameters
    grow_parser = subparsers.add_parser('growing', parents=[base_parser])
    grow_parser.add_argument('--d', type=float, required=True)
    grow_parser.add_argument('--dropout-rate', type=float, required=True)
    grow_parser.add_argument('--removal', action='store_true')

    args = parser.parse_args()

    config = build_config(args)

    # Set cpu/gpu as device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available()
                                  else torch.FloatTensor)

    trans_seq = transforms.Compose([
        transforms.ToTensor(),
        Flatten()
    ])
    n_classes = 2

    data_dir = join('..', 'data')
    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=trans_seq)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=trans_seq)
    n_features = train_data.data.shape[1] * train_data.data.shape[2]

    mean, std = None, None
    if config['norm'] == 'global':
        mean = torch.mean(torch.Tensor.float(train_data.data / 255)).to(device)
        std = torch.std(torch.Tensor.float(train_data.data / 255)).to(device)
    elif config['norm'] == 'feature':
        mean = torch.mean(torch.Tensor.float(train_data.data / 255), dim=0).reshape(-1).to(device)
        std = torch.std(torch.Tensor.float(train_data.data / 255), dim=0).reshape(-1).to(device)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    split = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    if not os.path.exists(config['log_dir']):
        os.mkdir(config['log_dir'])

    # Model definition
    in_features = n_features  # 784
    outputs_list = config['model_arch']
    epochs_per_task = 1

    # Set experiment seed
    torch.manual_seed(config['exp_seed'])
    if config['gln_type'] == 'prototypes':
        gln = ProtoGLN(in_features, outputs_list, n_classes, config['n_prototypes'], config['prot_bias_std'])

        if config['init'] == 'dataset':
            shapes = []
            for out_dim in outputs_list:
                shapes.append((n_classes, out_dim, config['n_prototypes'], in_features))

            prot_list = sample(train_data, shapes, config['norm'])
            for i, prot in enumerate(prot_list):
                gln.layers[i].prototypes = prot.to(device)
    elif config['gln_type'] == 'hyperplanes':
        gln = HypGLN(in_features, outputs_list, n_classes, config['n_hyperplanes'])
    else:
        gln = GrowPGLN(in_features,
                       outputs_list,
                       n_classes,
                       config['d'],
                       dropout=config['dropout_rate'])

    gln.to(device)

    optim = torch.optim.SGD(gln.parameters(), config['lr'])
    loss_fn = nn.BCELoss(reduction='none')

    # Logging
    accs = torch.zeros(len(split), len(split))
    log = Logger(config)

    # Temp
    mean_std_estimator = MeanStdEstimator()

    # Training
    print('Start training')

    for idx, labels in enumerate(split):
        gln.train()
        for epoch in range(epochs_per_task):
            for i, (train_x, train_y) in enumerate(iter(train_loader)):
                if train_y in labels:
                    x = train_x.to(device)

                    if config['norm'] == 'online':
                        mean, std = mean_std_estimator(x)

                    if config['norm'] == 'none':
                        norm_x = x
                    else:
                        norm_x = (x - mean) / (std + 1.)

                    output = gln(norm_x)

                    optim.zero_grad()

                    y = one_hot(train_y % 2, 2).to(torch.float32).to(device)
                    gln.backward(loss_fn, y)

                    optim.step()
                    gln.propagate_updates()

                # print progress
                if (i + 1) % 1000 == 0:
                    print(f'Task {idx}, epoch {epoch}, step {i + 1}')

        if config['removal']:
            gln.remove_unused_prototypes(1)

        gln.eval()
        for i, lab_couple in enumerate(split[:idx+1]):

            acc = 0.0
            tot = 0
            for test_x, test_y in iter(test_loader):
                if test_y in lab_couple:
                    x = test_x.to(device)
                    if config['norm'] == 'online':
                        mean, std = mean_std_estimator(x, test=True)

                    if config['norm'] == 'none':
                        norm_x = x
                    else:
                        norm_x = (x - mean) / (std + 1.)

                    pred = gln(norm_x)

                    y = one_hot(test_y % 2, 2).to(torch.float32).to(device)

                    if torch.argmax(pred) == torch.argmax(y):
                        acc += 1
                    tot += 1
            final_acc = acc / tot
            print(f'Accuracy on task {i}: {final_acc * 100:.2f}%')
            log.append(f'Accuracy on task {i}: {final_acc}')
            accs[idx, i] = final_acc

    print(accs)
    log.save_to_file()
    torch.save(accs, join(config['log_dir'], f'accuracy_seed_{config["exp_seed"]}.pt'))
