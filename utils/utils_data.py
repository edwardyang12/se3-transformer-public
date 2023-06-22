import warnings

import dgl
import torch
import random
import csv
import pickle

def to_np(x):
    return x.cpu().detach().numpy()


class PickleGraph:
    """Lightweight graph object for easy pickling. Does not support batched graphs."""

    def __init__(self, G=None, desired_keys=None):
        self.ndata = dict()
        self.edata = dict()

        if G is None:
            self.src = []
            self.dst = []
        else:
            if G.batch_size > 1:
                warnings.warn("Copying a batched graph to a PickleGraph is not supported. "
                              "All node and edge data will be copied, but batching information will be lost.")

            self.src, self.dst = (to_np(idx) for idx in G.all_edges())

            for k in G.ndata:
                if desired_keys is None or k in desired_keys:
                    self.ndata[k] = to_np(G.ndata[k])

            for k in G.edata:
                if desired_keys is None or k in desired_keys:
                    self.edata[k] = to_np(G.edata[k])

    def all_edges(self):
        return self.src, self.dst


def copy_dgl_graph(G):
    if G.batch_size == 1:
        src, dst = G.all_edges()
        G2 = dgl.DGLGraph((src, dst))
        for edge_key in list(G.edata.keys()):
            G2.edata[edge_key] = torch.clone(G.edata[edge_key])
        for node_key in list(G.ndata.keys()):
            G2.ndata[node_key] = torch.clone(G.ndata[node_key])
        return G2
    else:
        list_of_graphs = dgl.unbatch(G)
        list_of_copies = []

        for batch_G in list_of_graphs:
            list_of_copies.append(copy_dgl_graph(batch_G))

        return dgl.batch(list_of_copies)


def update_relative_positions(G, *, relative_position_key='d', absolute_position_key='x'):
    """For each directed edge in the graph, calculate the relative position of the destination node with respect
    to the source node. Write the relative positions to the graph as edge data."""
    src, dst = G.all_edges()
    absolute_positions = G.ndata[absolute_position_key]
    G.edata[relative_position_key] = absolute_positions[dst] - absolute_positions[src]

def split_data(path, split= [.8,.1,.1],
                data= "/edward-slow-vol/CPSC_552/immunoai/data/immuno_data_multi_allele_for_Edward.csv", 
                HLA="/edward-slow-vol/CPSC_552/immunoai/data/HLA_27_seqs.txt"):
    HLA_processed = {}
    with open(HLA, 'r') as f:
        for count, line in enumerate(f):
            if count == 0:
                continue 
            allele, seq = line.strip().split("\t")
            HLA_processed[allele] = seq

    all_seqs = []
    with open(data, "r") as f:
        reader = csv.reader(f)
        for count, line in enumerate(reader):
            if count==0:
                continue

            peptide = line[1]
            allele = line[2]
            sequence = HLA_processed[allele]+peptide

            enrichment = float(line[4])
            immuno = int(line[3])

            all_seqs.append((sequence,immuno))
    print(count)

    # assign to train, val, test
    distribute = random.choices([0, 1, 2], weights=split, k=len(all_seqs)) 
    trainset = set()
    valset = set()
    testset = set()
    for x,y in enumerate(distribute):
        if y==0:
            trainset.add(x)
        elif y==1:
            valset.add(x)
        else:
            testset.add(x)

    distribution ={
        'train': trainset,
        'val': valset,
        'test': testset
    }

    with open(path + '/split.pickle', 'wb') as handle:
        pickle.dump(distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_y = 0
    train_n = 0
    val_y = 0
    val_n = 0
    test_y = 0
    test_n = 0
    for i, (location, seq) in enumerate(zip(distribute, all_seqs)): 
        if location==0: # train
            if seq[1]==0: # non immuno
                train_n +=1
            else:
                train_y +=1
        elif location==1: # val
            if seq[1]==0: # non immuno
                val_n +=1
            else:
                val_y +=1
        else: # test
            if seq[1]==0: # non immuno
                test_n +=1
            else:
                test_y +=1
    
    print(train_y, train_n, val_y, val_n, test_y, test_n)

if __name__ == "__main__":
    split_data('/edward-slow-vol/CPSC_552/alpha_multi')
