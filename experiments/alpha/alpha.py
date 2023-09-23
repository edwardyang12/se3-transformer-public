import os
import sys

import dgl
from dgl.data.utils import load_graphs
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import torch
import pickle
import csv

from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset, DataLoader

class AlphaDataset(Dataset):

    def __init__(self, 
            immuno_path = '/home/ey229/project/immunoai/data/immuno_data_train_IEDB_A0201_HLAseq_2_csv.csv', 
            structures_path = '/home/ey229/project/data/alpha_single/alpha_structure',
            graph_path = '/home/ey229/project/data/alpha_single/alpha_dgl_l4',
            split_path = None,
            atom_feature_size = 22,
            num_bonds = 1,
            mode= 'train', # train, val, test
            transform=None): 
        self.immuno_path = immuno_path
        self.structures_path = structures_path
        self.graph_path = graph_path

        if split_path:
            with open(split_path, 'rb') as p:
                self.split = pickle.load(p)[mode]
        else:
            self.split = None

        self.mode = mode
        self.transform = transform
        self.load_data()
        self.len = len(self.targets)
        self.qt = None # quantile transform for normal distribution of data


        self.atom_feature_size = atom_feature_size # NOTE change this depenending on number of node features
        # 36 is number of atoms 
        # 20 is number of amino acids 
        # 2 is hydrogen acceptor and donor
        # 61 is number of additional properties
        self.num_bonds = num_bonds

        print(f"Loaded {mode}-set, source: {self.immuno_path}, length: {len(self)}")
    
    def __len__(self):
        return self.len
    
    def load_data(self):

        self.inputs = []
        self.targets = []
        self.immuno_list = []
        self.sequence_list = []
        with open(self.structures_path + '/mapping.pickle', 'rb') as p:
            mapping = pickle.load(p)
            with open(self.immuno_path, "r") as f:
                reader = csv.reader(f)
                for count, line in enumerate(reader):
                    count -=1
                    if count<0 or (self.split and count not in self.split):
                        continue

                    peptide = line[0].replace("J", "")
                    sequence = line[1]
                    sequence = sequence + peptide
                    self.sequence_list.append(sequence)

                    enrichment = float(line[2])
                    immuno = int(line[3])

                    x = self.graph_path + "/rank_1_" + mapping[sequence] + ".bin"
                    if not os.path.isfile(x) or os.stat(x).st_size == 0:
                        print(x)
                        continue

                    self.inputs.append(x)
                    self.targets.append(enrichment)
                    self.immuno_list.append(immuno)

        self.targets = np.array(self.targets).reshape(-1,1)
        self.qt = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')
        self.targets = self.qt.fit_transform(self.targets).reshape(-1)

        self.mean = np.mean(self.targets)
        self.std = np.std(self.targets)


    def get_target(self, idx, normalize=True):
        target = self.targets[idx]
        if normalize:
            target = (target - self.mean) / self.std
        if self.mode =='train':
            return target
        else:
            return target, self.immuno_list[idx]

    def norm2units(self, x, denormalize=True, center=True):
        # Convert from normalized to original representation
        if denormalize:
            x = x * self.std
            # Add the mean: not necessary for error computations
            if not center:
                x += self.mean
        return x

    def get(self, idx):
        return self.inputs[idx]

    def __getitem__(self, idx):
        # Load node features

        x = self.get(idx)

        # if 'rank_1_prediction_Immunogenicity_dd94e' not in x:
        #     src_ids = torch.tensor([2, 3, 4])
        #     # Destination nodes for edges (2, 1), (3, 2), (4, 3)
        #     dst_ids = torch.tensor([1, 2, 3])
        #     g = dgl.graph((src_ids, dst_ids))
        #     return g , 0

        glist, label_dict = load_graphs(x)
        g = glist[0]

        # Augmentation on the coordinates
        if self.transform:
            g.ndata['x'] = self.transform(g.ndata['x'])

        # Load target
        y = self.get_target(idx, normalize=True)
        y = np.array([y])

        # print("================================")
        # print(g)
        if self.mode =='train':
            return g, y
        else:
            return g, y, self.sequence_list[idx]
        

if __name__ == "__main__":
    mode = 'train'
    if mode =='train':
        def collate(samples):
            graphs, y = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            return batched_graph, torch.tensor(y)
    else:
        def collate(samples):
            graphs, y, seq = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            return batched_graph, torch.tensor(y), seq

    dataset = AlphaDataset(mode = mode, atom_feature_size = 22) 
    print(len(dataset))
    dataloader = GraphDataLoader(dataset, use_ddp=False, 
                                        batch_size= 32,
                                        shuffle= True)

    for i in range(5):
        for data in dataloader:
            print("MINIBATCH")
            print(data)
            break
