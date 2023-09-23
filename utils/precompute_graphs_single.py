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
from graphein.protein.graphs import read_pdb_to_dataframe
import graphein.protein as gp

# need to change these two for testing
immuno_path = '/home/ey229/project/immunoai/data/immuno_data_test_IEDB_A0201_HLAseq_2_csv.csv'
structures_path = '/home/ey229/project/data/alpha_single/alpha_structure_test'

new_edge_funcs = {"edge_construction_functions": [add_peptide_bonds]
                  ,"node_metadata_functions": [amino_acid_one_hot,hydrogen_bond_acceptor,hydrogen_bond_donor]
                  ,"granularity": "CA"
                  ,"exclude_waters": False}

config = ProteinGraphConfig(**new_edge_funcs)
LENGTH = 180

sequence_positions = range(1, LENGTH)
convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg")

enc_dict = {'GLY': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 'HIS': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'ARG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 'PHE': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 'GLU': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'ILE': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'ASP': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'GLN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'ASN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 'CYS': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'MASK': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

save_path = '/home/ey229/project/data/alpha_single/alpha_dgl_l4_1_test'

def load_data(structures_path, immuno_path):
    inputs = []
    with open(structures_path + '/mapping.pickle', 'rb') as p:
        mapping = pickle.load(p)
        with open(immuno_path, "r") as f:
            reader = csv.reader(f)
            for count, line in enumerate(reader):
                if count==0:
                    continue
                count = count - 1

                peptide = line[0].replace("J", "")
                sequence = line[1]
                sequence = sequence + peptide

                x = structures_path + "/rank_1_" + mapping[sequence] + ".pdb"
                if not os.path.isfile(x) or os.stat(x).st_size == 0:
                    continue

                inputs.append(x)
    return inputs

def generate(x):
    data_label = x.split("/")[-1][:-4]

    g = construct_graph(config=config, path= x)
    s_g = extract_subgraph_by_sequence_position(g, sequence_positions)
    g = gp.extract_subgraph_from_chains(s_g, ["A","B"])
    # g = gp.extract_subgraph_from_chains(g, ["B"])

    data = convertor(g)
    tdf = read_pdb_to_dataframe(x)

    tdfa = tdf[tdf['chain_id'] == 'A']
    tdfa = tdfa.drop_duplicates('residue_number')
    tdfa = tdfa[:LENGTH-1]
    sequence_a = tdfa['residue_name'].tolist()

    tdfb = tdf[tdf['chain_id'] == 'B']
    tdfb = tdfb.drop_duplicates('residue_number')
    sequence_b = tdfb['residue_name'].tolist()

    #amino acid one-hot encoding
    aa_one_hot_a = torch.tensor([enc_dict[x] for x in sequence_a])
    aa_one_hot_b = torch.tensor([enc_dict[x] for x in sequence_b])
    aa_one_hot = torch.cat([aa_one_hot_a, aa_one_hot_b], dim=0)

    # both ways give the same output encoding but the amino acids it corresponds with are wrong! 
    # one_hot = torch.tensor([d['amino_acid_one_hot'] for n, d in g.nodes(data=True)])

    # h donors
    h_donors = torch.tensor([d['hbond_donors'] for n, d in g.nodes(data=True)])
    # h acceptors
    h_acceptors = torch.tensor([d['hbond_acceptors'] for n, d in g.nodes(data=True)])

    # physio chem 
    # physio_chem = [d['expasy'].tolist() for n, d in g.nodes(data=True)]
    # physio_chem = torch.tensor(physio_chem[-data.num_nodes:])

    # atom type
    # atoms = [d['atom_type'] for n, d in g.nodes(data=True)]
    # atom_encoded = [atom_labels[x] for x in atoms]
    # atom_encoded = torch.tensor(atom_encoded[-data.num_nodes:])

    data.x = torch.cat([aa_one_hot, h_donors, h_acceptors], dim =1)

    data.edge_index = torch_geometric.utils.to_undirected(data.edge_index)
    start = data['edge_index'][0]
    end = data['edge_index'][1]

    data['coords'] = data['coords'].float()

    row, col = data.edge_index

    g = dgl.graph((row, col))

    g.ndata['x'] = data['coords']
    g.ndata['f'] = torch.unsqueeze(data['x'],2).float()
    g.edata['d'] = (data['coords'][start] - data['coords'][end]).float()
    g.edata['w'] = torch.ones(g.edata['d'].shape[0],1).float() # all edges are of the same type

    save_graphs(save_path +"/"+ data_label + ".bin", [g])

if __name__ == "__main__":
    inputs = load_data(structures_path, immuno_path)

    CPUS = 8
    pool = multiprocessing.Pool(CPUS)
    result = pool.map(generate, inputs)
