import argparse
import os
from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torch import nn

from models.grow_pgln import GrowPGLN
from models.hyperplanes_gln import HypGLN
from models.prototypes_gln import ProtoGLN
from utils.loggers import Logger

from utils.mean_std_estimators import MeanStdEstimator
from utils.transforms import Flatten, OneHot


def build_config(main_args):
    config = dict()
    config['exp_seed'] = main_args.exp_seed
    config['n_permutations'] = main_args.n_permutations
    config['perm_seed'] = main_args.perm_seed
    config['gln_type'] = main_args.gln_type
    config['model_arch'] = main_args.model_arch
    config['lr'] = main_args.lr
    config['norm'] = main_args.norm
    config['log_dir'] = main_args.log_dir

    if config['gln_type'] == 'hyperplanes':
        config['n_hyperplanes'] = main_args.n_hyperplanes
        config['hyp_bias_std'] = main_args.hyp_bias_std
    if config['gln_type'] == 'prototypes':
        config['n_prototypes'] = main_args.n_prototypes
        config['init'] = main_args.init
        config['prot_bias_std'] = main_args.prot_bias_std
    if config['gln_type'] == 'growing':
        config['d'] = main_args.d
        assert 0.0 <= main_args.dropout_rate <= 1.0
        config['dropout_rate'] = main_args.dropout_rate
        if main_args.removal is not None:
            config['removal'] = True
        else:
            config['removal'] = False

    return config


def sample(train_data, permutations, shapes, norm='none'):
    loader = DataLoader(train_data, batch_size=train_data.data.shape[0], shuffle=False)
    data, _ = next(iter(loader))
    concat = None

    for perm in permutations:
        if concat is None:
            concat = data[:, perm]
        else:
            concat = torch.cat((concat, data[:, perm]), dim=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    concat = concat.to(device)
    if norm == 'online':
        estimator = MeanStdEstimator()
        for i in range(concat.shape[0]):
            mean, std = estimator(concat[i])
            concat[i] = torch.div(concat[i] - mean, std + 1.)
            if (i+1) % 10000 == 0:
                print('processed', i+1, 'examples')
    elif norm == 'feature':
        mean = torch.mean(concat, dim=0)
        std = torch.std(concat, dim=0)
        concat = torch.div(concat - mean, 1. + std)
    elif norm == 'global':
        mean = torch.mean(concat)
        std = torch.std(concat)
        concat = torch.div(concat - mean, 1. + std)

    samples = []
    for shape in shapes:
        print(shape)
        idx = torch.randint(0, concat.shape[0], size=(shape[0] * shape[1] * shape[2],))
        sample = concat[idx].view(shape)
        samples.append(sample)
    return samples


def compute_mean_std(train_data, permutations):
    loader = DataLoader(train_data, batch_size=train_data.data.shape[0], shuffle=False)
    data, _ = next(iter(loader))
    concat = None
    for perm in permutations:
        if concat is None:
            concat = data[:, perm]
        else:
            concat = torch.cat((concat, data[:, perm]), dim=0)

    mean = torch.mean(concat, dim=0)
    std = torch.std(concat, dim=0)
    return mean, std


if __name__ == '__main__':
    # Parse main args
    # experiment setup
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--exp-seed', type=int, default=42)
    base_parser.add_argument('--perm-seed', type=int, default=42)
    base_parser.add_argument('--n-permutations', type=int, default=8)
    base_parser.add_argument('--log-dir', type=str, required=True)
    # model common parameters
    base_parser.add_argument('--model-arch', metavar='layer_neurons', type=int, nargs='+', required=True)
    base_parser.add_argument('--lr', type=float, default=0.01)
    base_parser.add_argument('--norm', type=str, default='none', choices=['online', 'global', 'feature', 'none'])

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='gln_type')
    # hyperplanes specific parameters
    hyp_parser = subparsers.add_parser('hyperplanes', parents=[base_parser])
    hyp_parser.add_argument('--n-hyperplanes', type=int, required=True)
    hyp_parser.add_argument('--hyp-bias-std', type=float, default=0.05)
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

    # Dataset preparation
    data_dir = join('..', 'data')
    mnist_train = datasets.MNIST(root=data_dir, train=True, download=True)
    n_classes = 10

    # Fix seed for reproducibility
    torch.manual_seed(config['perm_seed'])
    train_data = torch.Tensor.float(mnist_train.data / 255)
    n_features = train_data.shape[1] * train_data.shape[2]  # 28*28 = 784

    # Load or create permutations
    if os.path.exists(join(config['log_dir'], 'permutations.pt')):
        permutations = torch.load(join(config['log_dir'], 'permutations.pt'), map_location=device)
    else:
        permutations = [torch.randperm(n_features) for _ in range(config['n_permutations'])]
        if not os.path.exists(config['log_dir']):
            os.mkdir(config['log_dir'])
        torch.save(permutations, join(config['log_dir'], 'permutations.pt'))

    trans_seq = transforms.Compose([
        transforms.ToTensor(),
        Flatten()
    ])
    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=trans_seq, target_transform=OneHot(n_classes))
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=trans_seq, target_transform=OneHot(n_classes))

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Compute mean and std for normalization
    mean, std = None, None
    if config['norm'] == 'global':
        mean = torch.mean(train_data.data / 255.).to(device)
        std = torch.std(train_data.data / 255.).to(device)
    elif config['norm'] == 'feature':
        mean, std = compute_mean_std(train_data, permutations)
        mean = mean.to(device)
        std = std.to(device)

    mean_std_estimator = MeanStdEstimator()

    # Model definition
    in_features = n_features
    outputs_list = config['model_arch']
    epochs_per_task = 1

    # Set experiment seed
    torch.manual_seed(config['exp_seed'])

    # Build model
    if config['gln_type'] == 'prototypes':
        gln = ProtoGLN(in_features, outputs_list, n_classes, config['n_prototypes'], config['prot_bias_std'])

        if config['init'] != 'random':
            shapes = []
            for out_dim in outputs_list:
                shapes.append((n_classes, out_dim, config['n_prototypes'], in_features))
            prot_list = []
            if config['init'] == 'dataset':
                prot_list = sample(train_data, permutations, shapes, config['norm'])

            for i, prot in enumerate(prot_list):
                gln.layers[i].prototypes = prot.to(device)

    elif config['gln_type'] == 'hyperplanes':
        gln = HypGLN(in_features, outputs_list, n_classes, config['n_hyperplanes'], config['hyp_bias_std'])
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
    accs = torch.zeros(len(permutations), len(permutations))
    log = Logger(config)

    # Training
    print('Start training')
    for idx, permutation in enumerate(permutations):
        log.append(f'Train on task {idx}')
        # Switch to train mode
        gln.train()
        for epoch in range(epochs_per_task):
            for i, (train_x, train_y) in enumerate(iter(train_loader)):
                x = train_x[:, permutation].to(device)

                if config['norm'] == 'online':
                    mean, std = mean_std_estimator(x)

                if config['norm'] == 'none':
                    x = x
                else:
                    x = (x - mean) / (std + 1.)

                output = gln(x)

                optim.zero_grad()
                gln.backward(loss_fn, train_y.to(device))

                optim.step()

                gln.propagate_updates()

                # print progress every 1000 samples processed
                if (i+1) % 1000 == 0:
                    print(f'Task {idx}, epoch {epoch}, step {i+1}')

        if config['removal']:
            gln.remove_unused_prototypes(1)

        # Task evaluation, switch to eval mode
        gln.eval()
        for i in range(idx+1):
            acc = 0.0
            for test_x, test_y in iter(test_loader):
                x = test_x[:, permutations[i]].to(device)

                if config['norm'] == 'online':
                    mean, std = mean_std_estimator(x)

                if config['norm'] == 'none':
                    x = x
                else:
                    x = (x - mean) / (std + 1.)

                pred = gln(x)

                test_y = test_y.to(device)
                if torch.argmax(pred) == torch.argmax(test_y):
                    acc += 1

            final_acc = acc / len(test_loader)
            print(f'Accuracy on task {i}: {final_acc * 100:.2f}%')
            log.append(f'Accuracy on task {i}: {final_acc}')
            accs[idx, i] = final_acc

    print(accs)

    log.save_to_file()
    torch.save(accs, join(config['log_dir'], f'accuracy_seed_{config["exp_seed"]}.pt'))
