# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def loadG(data_folder, filename):
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names = ['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2.edges', names = ['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    return G1, G2

def loadG_link(data_folder, test_frac, filename):
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1_edges = pd.read_csv(data_folder + filename + '1.edges', names = ['0', '1'])
    G1.add_edges_from(np.array(G1_edges))
    G2_edges = pd.read_csv(data_folder + filename + '2_' + str(test_frac) + '.edges', names = ['0', '1'])
    G2.add_edges_from(np.array(G2_edges))
    test_edges = pd.read_csv(data_folder + filename + '2_' + str(test_frac) + '_test.edges', names = ['0', '1'])
    return G1, G2, test_edges

def load_attribute(attribute_folder, filename, G1, G2):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    attribute1 = pd.read_csv(attribute_folder + filename + 'attr1.csv', header = None, index_col = 0)
    attribute2 = pd.read_csv(attribute_folder + filename + 'attr2.csv', header = None, index_col = 0)
    attribute1 = np.array(attribute1.loc[G1_nodes, :])
    attribute2 = np.array(attribute2.loc[G2_nodes, :])
    attr_cos = cosine_similarity(attribute1, attribute2)
    #attr_cos = pd.DataFrame(attr_cos, index = attribute1.index, columns = attribute2.index)
    #attr_cos = attr_cos.loc[G1_nodes, G2_nodes]
    #attr_cos = np.array(attr_cos)
    return attr_cos, attribute1, attribute2

def greedy_match(X, G1, G2):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())
    m, n = X.shape
    x = np.array(X.flatten()).reshape(-1, )
    minSize = min(m, n)
    usedRows = np.zeros(n)
    usedCols = np.zeros(m)
    maxList = np.zeros(minSize)
    row = np.zeros(minSize)
    col = np.zeros(minSize)
    ix = np.argsort(-np.array(x))
    matched = 0
    index = 0
    while (matched < minSize):
        ipos = ix[index]
        jc = int(np.floor(ipos / n))
        ic = int(ipos - jc * n)
        if (usedRows[ic] != 1 and usedCols[jc] != 1):
            row[matched] = G1_nodes[ic]
            col[matched] = G2_nodes[jc]
            maxList[matched] = x[index]
            usedRows[ic] = 1
            usedCols[jc] = 1
            matched += 1
        index += 1;
    row = row.astype(int)
    col = col.astype(int)
    return zip(col, row)

def greedy_match_CENALP(X, G1_nodes, G2_nodes, minSize = 10):
    m, n = X.shape
    x = np.array(X.flatten()).reshape(-1, )
    usedRows = np.zeros(n)
    usedCols = np.zeros(m)
    maxList = np.zeros(minSize)
    row = np.zeros(minSize)
    col = np.zeros(minSize)
    ix = np.argsort(-np.array(x))
    matched = 0
    index = 0
    while (matched < minSize):
        ipos = ix[index]
        jc = int(np.floor(ipos / n))
        ic = int(ipos - jc * n)
        if (usedRows[ic] != 1 and usedCols[jc] != 1):
            row[matched] = G1_nodes[ic]
            col[matched] = G2_nodes[jc]
            maxList[matched] = x[index]
            usedRows[ic] = 1
            usedCols[jc] = 1
            matched += 1
        index += 1;
    row = row.astype(int)
    col = col.astype(int)
    return zip(col, row)


def one2one_accuracy_supervised(S, G1, G2, alignment, seed_list1, seed_list2):
    ss = list(greedy_match(S, G1, G2))
    ground_truth = list(zip(alignment.iloc[:, 1], alignment.iloc[:, 0]))
    train = list(zip(seed_list1, seed_list2))
    ss = [str(x) for x in ss]
    ground_truth = [str(x) for x in ground_truth]
    train = [str(x) for x in train]
    ss = list(set(ss).difference(set(train)))
    ground_truth = list(set(ground_truth).difference(set(train)))

    return 100 * len(np.intersect1d(ss, ground_truth))/len(ground_truth)

def one2one_accuracy(S, G1, G2, alignment):
    ss = list(greedy_match(S, G1, G2))
    ground_truth = list(zip(alignment.iloc[:, 1], alignment.iloc[:, 0]))
    ss = [str(x) for x in ss]
    ground_truth = [str(x) for x in ground_truth]
    return 100 * len(np.intersect1d(ss, ground_truth))/len(ground_truth)
def topk_accuracy(S, G1, G2, alignment_dict_reversed, k):
    G2_nodes = list(G2.nodes())
    argsort = np.argsort(-S, axis = 1)
    G1_dict = {}
    for key, value in enumerate(list(G1.nodes())):
        G1_dict[value] = key
    G2_dict = {}
    for key, value in enumerate(list(G2.nodes())):
        G2_dict[value] = key
    L = []
    for i in range(len(argsort)):
        index = alignment_dict_reversed.get(G2_nodes[i], None)
        if index == None:
            continue
        L.append(np.where(argsort[i, :] == G1_dict[index])[0][0] + 1)
    return np.sum(np.array(L) < k) / len(L) * 100
def topk_accuracy_supervised(S, G1, G2, alignment_dict_reversed, k, seed_list1, seed_list2):
    G2_nodes = list(set(list(alignment_dict_reversed.keys())) - set(seed_list2))
    G1_nodes = list(set(list(alignment_dict_reversed.values())) - set(seed_list1))
    S = pd.DataFrame(S, index = list(G2.nodes()), columns = list(G1.nodes()))
    S = np.array(S.loc[G2_nodes, G1_nodes])
    argsort = np.argsort(-S, axis = 1)
    G1_dict = {}
    for key, value in enumerate(G1_nodes):
        G1_dict[value] = key
    G2_dict = {}
    for key, value in enumerate(G2_nodes):
        G2_dict[value] = key
    L = []
    for i in range(len(argsort)):
        
        L.append(np.where(argsort[i, :] == G1_dict[alignment_dict_reversed[G2_nodes[i]]])[0][0] + 1)
    return np.sum(np.array(L) < k) / len(L) * 100

def prior_alignment(W1, W2):
    H = np.zeros([len(W2), len(W1)])
    d1 = np.sum(W1, 0)
    d2 = np.sum(W2, 0)
    for i in range(len(W2)):
        H[i, :] = np.abs(d2[i] - d1) / np.max(np.concatenate([np.array([d2[i]] * len(d1)), d1]).reshape(2, -1), axis = 0)
    H = 1 - H
    return H
def split_graph(edge_tuples, orig_num_cc, cut_size, G):
    g = G.copy()
    np.random.shuffle(edge_tuples)
    test_edges = []
    k = 0
    for edge in edge_tuples:
        print('\r{}/{}'.format(k, cut_size), end='')
        node1, node2 = edge[0], edge[1]
        
        g.remove_edge(node1, node2)
        if nx.number_connected_components(g) > orig_num_cc:
            g.add_edge(node1, node2)
            continue
        k += 1
        test_edges.append([node1, node2])
        if k > cut_size:
            break
    return g, test_edges
