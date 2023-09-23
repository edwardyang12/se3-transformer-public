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
        init_method='tcp://127.0.0.1:12342', # change this for each run
        world_size=world_size,
        rank=rank)

def init_model(device, dataset, FLAGS):
    # Fix seed for random numbers
    if not FLAGS.seed: FLAGS.seed = 1992 #np.random.randint(100000)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    epoch = 0

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

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    if FLAGS.restore is not None and wandb.run.resumed:
        checkpoint = torch.load(wandb.restore(FLAGS.restore))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']          

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                               FLAGS.num_epochs, 
                                                               eta_min=1e-4)

    return model, optimizer, scheduler, epoch

def to_np(x):
    return x.cpu().detach().numpy()

def train_epoch(epoch, model, dataloader, optimizer, scheduler, FLAGS, device, train_dataset):
    model.train()

    num_iters = len(dataloader)
    dataloader.set_epoch(epoch)
    
    for i, (g, y) in enumerate(dataloader):
        g = g.to(device)
        y = y.to(device)

        # run model forward and compute loss
        # pred = model(g)
        pred, embedding = model(g)
        l1_loss, __, rescale_loss = task_loss(pred, y, train_dataset)

        # backprop
        optimizer.zero_grad()
        l1_loss.backward()
        optimizer.step()

        if i % FLAGS.print_interval == 0 and str(device) == 'cuda:0':
            print(f"[{epoch}|{i}] l1 loss: {l1_loss:.5f}")
        if i % FLAGS.log_interval == 0 and str(device) == 'cuda:0':
            wandb.log({"Train L1 loss": to_np(l1_loss)})

    if str(device) =='cuda:0':
        wandb.log({"Train Epoch L1 loss": to_np(l1_loss)}, step=epoch)

    scheduler.step(epoch)
    

def val_epoch(epoch, model, dataloader, FLAGS, device, val_dataset):
    model.eval()

    rloss = 0
    l1loss = 0
    for i, (g, y, seq) in enumerate(dataloader):
        g = g.to(device)
        y = y.to(device)

        # run model forward and compute loss
        # pred = model(g)
        pred, embedding = model(g)
        l1_loss, __, rescale_loss = task_loss(pred.detach(), y, val_dataset, use_mean=False)
        rloss += rescale_loss
        l1loss += l1_loss
    rloss /= FLAGS.val_size
    l1loss /= FLAGS.val_size

    print(f"...[{epoch}|val] l1 loss: {l1_loss:.5f}")
    if str(device) =='cuda:0':
        wandb.log({"Val L1 loss": to_np(l1loss)}, step=epoch)

    return l1loss

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

# Loss function
def task_loss(pred, target, dataset, use_mean=True):
    l1_loss = torch.sum(torch.abs(pred - target))
    l2_loss = torch.sum((pred - target)**2)
    if use_mean:
        l1_loss /= pred.shape[0]
        l2_loss /= pred.shape[0]

    rescale_loss = dataset.norm2units(l1_loss)
    return l1_loss, l2_loss, rescale_loss

def main(rank, world_size, train_dataset, val_dataset, FLAGS, UNPARSED_ARGV):
    resume = False
    if FLAGS.restore is not None:
        resume = True
    if rank==0:
        # Log all args to wandb
        if FLAGS.name:
                wandb.init(project=f'{FLAGS.wandb}', name=f'{FLAGS.name}', resume=resume)
        else:
                wandb.init(project=f'{FLAGS.wandb}')

    init_process_group(world_size, rank)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # Prepare data
    train_loader = GraphDataLoader(train_dataset, use_ddp=True, 
                                        batch_size= FLAGS.batch_size,
                                        shuffle= True)
    val_loader = GraphDataLoader(val_dataset,
                                        batch_size= 1,
                                        shuffle= False)


    FLAGS.train_size = len(train_dataset)
    FLAGS.val_size = len(val_dataset)

    model, optimizer, scheduler, epoch = init_model(device, train_dataset, FLAGS)

    # Run training
    print('Begin training')
    lowest_l1 = float('inf')
    while epoch < FLAGS.num_epochs:
        
        train_epoch(epoch, model, train_loader, optimizer, scheduler, FLAGS, device, train_dataset)
        l1loss = val_epoch(epoch, model, val_loader, FLAGS, device, val_dataset)

        if epoch%20==0 and epoch>0 and rank==0:
            save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '_' + str(epoch) + '.pt')
            print(f"Saved: {save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path) 
            wandb.save(save_path)
        if l1loss< lowest_l1 and rank==0:
            best_path = os.path.join(FLAGS.save_dir, FLAGS.name + '_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, best_path) 
            wandb.save(best_path)
        epoch += 1
            
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
    parser.add_argument('--atoms', type=int, default=36, 
            help="Number of atom features")
    parser.add_argument('--bonds', type=int, default=1, 
            help="Number of bonds")

    # Logging
    parser.add_argument('--name', type=str, default=None,
            help="Run name")
    parser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
            help="Directory name to save models")
    parser.add_argument('--restore', type=str, default=None,
            help="Path to model to restore")
    parser.add_argument('--wandb', type=str, default='equivariant-attention', 
            help="wandb project name")

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=4, 
            help="Number of data loader workers")
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

    train_dataset = AlphaDataset(mode='train', 
                               immuno_path='/home/ey229/project/immunoai/data/immuno_data_train_IEDB_A0201_HLAseq_2_csv.csv',
                               structures_path = '/home/ey229/project/data/alpha_single/alpha_structure',
                               transform=RandomRotation(),
                               graph_path = '/home/ey229/project/data/alpha_single/alpha_dgl_l4',
                               atom_feature_size = FLAGS.atoms,
                               num_bonds= FLAGS.bonds, 
                               )

    val_dataset = AlphaDataset(mode='val', 
                               immuno_path = '/home/ey229/project/immunoai/data/immuno_data_test_IEDB_A0201_HLAseq_2_csv.csv', 
                               structures_path = '/home/ey229/project/data/alpha_single/alpha_structure_test',
                               transform=RandomRotation(),
                               graph_path = '/home/ey229/project/data/alpha_single/alpha_dgl_l4_test',
                               atom_feature_size = FLAGS.atoms,
                               num_bonds= FLAGS.bonds, 
                               )

    # Where the magic is
    num_gpus = FLAGS.gpus
    procs = []
    mp.spawn(main, args=(num_gpus, train_dataset, val_dataset, FLAGS, UNPARSED_ARGV), nprocs=num_gpus)
