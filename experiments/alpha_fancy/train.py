from utils.utils_profiling import * # load before other local modules

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import dgl
import math
import numpy as np
import torch
import wandb

from torch import nn, optim
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from alpha import AlphaDataset

from experiments.alpha import models 

def init_process_group(world_size, rank):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12346',
        world_size=world_size,
        rank=rank)

def init_model(device, dataset, FLAGS):
    # Fix seed for random numbers
    if not FLAGS.seed: FLAGS.seed = 1992 #np.random.randint(100000)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Choose model
    model = models.__dict__.get(FLAGS.model)(FLAGS.num_layers, 
                                             dataset.atom_feature_size, 
                                             FLAGS.num_channels,
                                             num_nlayers=FLAGS.num_nlayers,
                                             num_degrees=FLAGS.num_degrees,
                                             edge_dim=dataset.num_bonds,
                                             div=FLAGS.div,
                                             pooling=FLAGS.pooling,
                                             n_heads=FLAGS.head)

    model = model.to(device)
    if device.type == 'cpu':
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    return model

def to_np(x):
    return x.cpu().detach().numpy()

def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS, device):
    model.train()

    num_iters = len(dataloader)
    for i, (g, y) in enumerate(dataloader):
        g = g.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # run model forward and compute loss
        # pred = model(g)
        pred, embedding = model(g)
        l1_loss, __, rescale_loss = loss_fnc(pred, y)

        # backprop
        l1_loss.backward()
        optimizer.step()

        if i % FLAGS.print_interval == 0 and device=='cuda:0':
            print(f"[{epoch}|{i}] l1 loss: {l1_loss:.5f} rescale loss: {rescale_loss:.5f} [units]")
        if i % FLAGS.log_interval == 0 and device=='cuda:0':
            wandb.log({"Train L1 loss": to_np(l1_loss), 
                       "Rescale loss": to_np(rescale_loss)})

        if FLAGS.profile and i == 10:
            sys.exit()
    
        scheduler.step(epoch + i / num_iters)

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)

def main(rank, world_size, dataset, FLAGS, UNPARSED_ARGV):
    if rank==0:
        # Log all args to wandb
        if FLAGS.name:
                wandb.init(project=f'{FLAGS.wandb}', name=f'{FLAGS.name}')
        else:
                wandb.init(project=f'{FLAGS.wandb}')

    init_process_group(world_size, rank)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # Prepare data
    train_loader = GraphDataLoader(dataset, use_ddp=True, 
                                        batch_size= FLAGS.batch_size,
                                        shuffle= True)

#     train_loader = DataLoader(dataset, 
#                               batch_size=FLAGS.batch_size, 
# 			      shuffle=True, 
#                               collate_fn=collate, 
#                               num_workers=FLAGS.num_workers)

    FLAGS.train_size = len(dataset)

    model = init_model(device, dataset, FLAGS)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                               FLAGS.num_epochs, 
                                                               eta_min=1e-4)

    # Loss function
    def task_loss(pred, target, use_mean=True):
        l1_loss = torch.sum(torch.abs(pred - target))
        l2_loss = torch.sum((pred - target)**2)
        if use_mean:
            l1_loss /= pred.shape[0]
            l2_loss /= pred.shape[0]

        rescale_loss = dataset.norm2units(l1_loss)
        return l1_loss, l2_loss, rescale_loss

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')

    # Run training
    print('Begin training')
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS, device)
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', type=str, default='SE3Transformer', 
            help="String name of model")
    parser.add_argument('--num_layers', type=int, default=4,
            help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=4,
            help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=16,
            help="Number of channels in middle layers")
    parser.add_argument('--num_nlayers', type=int, default=0,
            help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true',
            help="Include global node in graph")
    parser.add_argument('--div', type=float, default=4,
            help="Low dimensional embedding fraction")
    parser.add_argument('--pooling', type=str, default='avg',
            help="Choose from avg or max")
    parser.add_argument('--head', type=int, default=1,
            help="Number of attention heads")

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=32, 
            help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, 
            help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50, 
            help="Number of epochs")

    # Logging
    parser.add_argument('--name', type=str, default=None,
            help="Run name")
    parser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
            help="Directory name to save models")
    parser.add_argument('--wandb', type=str, default='equivariant-attention', 
            help="wandb project name")

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=4, 
            help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
            help="Exit after 10 steps for profiling")
    parser.add_argument('--gpus', type=int, default=1, 
            help="Number of gpus")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=None)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Fix name
    if not FLAGS.name:
        FLAGS.name = f'E-d{FLAGS.num_degrees}-l{FLAGS.num_layers}-{FLAGS.num_channels}-{FLAGS.num_nlayers}'

    # Create model directory
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    dataset = AlphaDataset(mode='train', 
                               transform=RandomRotation())

    # Where the magic is
    num_gpus = FLAGS.gpus
    procs = []
    mp.spawn(main, args=(num_gpus, dataset, FLAGS, UNPARSED_ARGV), nprocs=num_gpus)
