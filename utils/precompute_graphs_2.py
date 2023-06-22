import os
import sys

import dgl
import multiprocessing
from dgl.data.utils import save_graphs
import numpy as np
import torch
import pickle
import torch_geometric
import csv
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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

# need to change these two for testing
HLA = "/edward-slow-vol/CPSC_552/immunoai/data/HLA_27_seqs.txt"
pep = "/edward-slow-vol/CPSC_552/immunoai/data/immuno_data_multi_allele_for_Edward.txt"
structures_path = '/edward-slow-vol/CPSC_552/alpha_multi/alpha_structure'

new_edge_funcs = {"edge_construction_functions": [add_peptide_bonds]
                  ,"node_metadata_functions": [amino_acid_one_hot,hydrogen_bond_acceptor,hydrogen_bond_donor, expasy_protein_scale]
                  ,"granularity": "CA"
                  ,"exclude_waters": False}

config = ProteinGraphConfig(**new_edge_funcs)

convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg")

save_path = '/edward-slow-vol/CPSC_552/alpha_multi/alpha_dgl_l11'

def load_data(structures_path, peptides):
    inputs = []
    with open(structures_path + '/mapping.pickle', 'rb') as p:
        mapping = pickle.load(p)
        for pep in peptides:
            x = structures_path + "/rank_1_" + mapping[pep] + ".pdb"
            if not os.path.isfile(x) or os.stat(x).st_size == 0:
                print(x)
                continue
            inputs.append(x)
    return inputs

def generate(x):
    data_label = x.split("/")[-1][:-4]

    g = construct_graph(config=config, path= x)
    g = gp.extract_subgraph_from_chains(g, ["A", "B"])

    data = convertor(g)

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


    data.x = torch.cat([one_hot, h_donors, h_acceptors], dim =1)

    data['coords'] = data['coords'].float()

    row, col = data.edge_index

    g = dgl.graph((row, col))

    g.ndata['x'] = data['coords']
    g.ndata['f'] = torch.unsqueeze(data['x'],2).float()
    g.edata['d'] = (data['coords'][start] - data['coords'][end]).float()
    g.edata['w'] = torch.ones(g.edata['d'].shape[0],1).float() # all edges are of the same type

    save_graphs(save_path +"/"+ data_label + ".bin", [g])

if __name__ == "__main__":
    HLA_processed = {}
    with open(HLA, 'r') as f:
        for count, line in enumerate(f):
            if count == 0:
                continue 
            allele, seq = line.strip().split("\t")
            HLA_processed[allele] = seq

    peptides = set()
    with open(pep, 'r') as f:
        for count, line in enumerate(f):
            if count == 0:
                continue 
            pep, allele, _ = line.strip().split("\t")
            peptides.add(HLA_processed[allele]+pep)

    inputs = load_data(structures_path, peptides)

    CPUS = 4
    pool = multiprocessing.Pool(CPUS)
    result = pool.map(generate, inputs)
