import os
import sys

import dgl
import numpy as np
import torch
import pickle
import graphein.protein as gp
import torch_geometric
import csv
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import Dataset, DataLoader
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_peptide_bonds
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.features.nodes.amino_acid import hydrogen_bond_acceptor
from graphein.protein.features.nodes.amino_acid import hydrogen_bond_donor
from graphein.protein.edges.intramolecular import hydrogen_bond, hydrophobic, peptide_bonds, van_der_waals
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.edges.distance import add_aromatic_interactions, add_cation_pi_interactions, add_hydrophobic_interactions, add_ionic_interactions
from graphein.protein.edges.atomic import add_atomic_edges, add_bond_order
from graphein.protein.features.nodes.dssp import asa 
from graphein.protein.features.nodes.amino_acid import expasy_protein_scale
from graphein.protein.features.nodes.geometry import add_sidechain_vector
from graphein.protein.subgraphs import extract_subgraph_by_sequence_position
import graphein.protein as gp

class AlphaDataset(Dataset):

    atom_feature_size = 36 # NOTE change this depenending on number of node features
    # numbers may be wrong
    # 36 is number of atoms ?
    # 20 is number of amino acids 
    # 2 is hydrogen acceptor and donor ?
    # 42 is number of additional properties ?
    num_bonds = 1

    def __init__(self, 
            immuno_path= '/edward-slow-vol/CPSC_552/immunoai/data/immuno_data_train_IEDB_A0201_HLAseq_2_csv.csv', 
            structures_path= '/edward-slow-vol/CPSC_552/alpha_structure', 
            mode: str='train', 
            transform=None): 
        self.immuno_path = immuno_path
        self.structures_path = structures_path
        self.mode = mode
        self.transform = transform
        self.new_edge_funcs = {"edge_construction_functions": [add_atomic_edges]
                  ,"node_metadata_functions": [amino_acid_one_hot,hydrogen_bond_acceptor,hydrogen_bond_donor, expasy_protein_scale]
                  ,"granularity": "atom"
                  ,"exclude_waters": False}
        self.config = ProteinGraphConfig(**self.new_edge_funcs)

        self.load_data()
        self.len = len(self.targets)
        self.convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg")
        self.sequence_positions = range(1, 60)
        
        atom_labels = ['NE', 'CG1', 'CE2', 'OG1', 'CE1', 'OG', 'OE2', 'CZ3', 'OD2', 'OD1', 'NE2', 'CD', 'NZ', 'CZ2', 'SG', 'OE1', 'O', 'CE', 'CZ', 'CA', 'ND2', 'NH1', 'ND1', 'OH', 'CD2', 'NH2', 'CH2', 'CD1', 'CG2', 'C', 'CB', 'CG', 'NE1', 'SD', 'CE3', 'N']
        atom_labels = sorted(atom_labels)
        self.atom_labels = {string: [int(i == idx) for idx in range(len(atom_labels))] for i, string in enumerate(atom_labels)}


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
                    if count==0:
                        continue
                    count = count - 1

                    peptide = line[0].replace("J", "")
                    sequence = line[1]
                    sequence = sequence + peptide
                    self.sequence_list.append(line[0])

                    enrichment = float(line[2])
                    immuno = float(line[3])

                    x = self.structures_path + "/rank_1_" + mapping[sequence] + ".pdb"
                    if not os.path.isfile(x) or os.stat(x).st_size == 0:
                        continue

                    self.inputs.append(x)
                    self.targets.append(enrichment)
                    self.immuno_list.append(immuno)
    
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

        g = construct_graph(config=self.config, path= x)
        # s_g = extract_subgraph_by_sequence_position(g, self.sequence_positions)
        # g = gp.extract_subgraph_from_chains(s_g, ["A","B"])
        g = gp.extract_subgraph_from_chains(g, ["B"])

        data = self.convertor(g)

        data.edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        start = data['edge_index'][0]
        end = data['edge_index'][1]

        # one hot
        one_hot = [d['amino_acid_one_hot'] for n, d in g.nodes(data=True)]
        one_hot = torch.tensor(one_hot[-data.num_nodes:])

        # h donors
        h_donors = [d['hbond_donors'] for n, d in g.nodes(data=True)]
        h_donors = torch.tensor(h_donors[-data.num_nodes:])

        # h acceptors
        h_acceptors = [d['hbond_acceptors'] for n, d in g.nodes(data=True)]
        h_acceptors = torch.tensor(h_acceptors[-data.num_nodes:])

        # physio chem 
        physio_chem = [d['expasy'].tolist() for n, d in g.nodes(data=True)]
        physio_chem = torch.tensor(physio_chem[-data.num_nodes:])

        # atom type
        atoms = [d['atom_type'] for n, d in g.nodes(data=True)]
        atom_encoded = [self.atom_labels[x] for x in atoms]
        atom_encoded = torch.tensor(atom_encoded[-data.num_nodes:])

        data.x = torch.cat([atom_encoded], dim =1)

        row, col = data.edge_index

        g = dgl.graph((row, col))

        # Augmentation on the coordinates
        if self.transform:
            data['coords'] = self.transform(data['coords']).float()

        g.ndata['x'] = data['coords']
        g.ndata['f'] = torch.unsqueeze(data['x'],2).float()
        g.edata['d'] = (data['coords'][start] - data['coords'][end]).float()
        g.edata['w'] = torch.ones(g.edata['d'].shape[0],1).float() # all edges are of the same type

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

    # dataset = AlphaDataset()
    dataset = AlphaDataset(mode=mode,
                            immuno_path='/edward-slow-vol/CPSC_552/immunoai/data/immuno_data_train_IEDB_A0201_HLAseq_2_csv.csv',
                            structures_path='/edward-slow-vol/CPSC_552/alpha_structure') 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate)

    for data in dataloader:
        print("MINIBATCH")
        # print(data)
        # break
