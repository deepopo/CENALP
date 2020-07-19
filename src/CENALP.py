# -*- coding: utf-8 -*-
import pickle as pkl
import networkx as nx
import numpy as np
import pandas as pd
from utils import *
from walks import multi_simulate_walks, single_simulate_walks
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from sklearn.linear_model import LogisticRegression

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from model import structing
data_folder = '../graph/'
attribute_folder = '../attribute/'
alignment_folder = '../alignment/'

def CENALP(G1, G2, q, attr1, attr2, attribute, alignment_dict, alignment_dict_reversed, 
           layer, align_train_prop, alpha, c, multi_walk):
    iteration = 1
    anchor = 0
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    
    if len(attribute) != 0:
        attribute = attribute.T
        attribute = pd.DataFrame(attribute, index = list(G1.nodes()), 
                                 columns = list(G2.nodes()))
    if align_train_prop == 0:
        seed_list1 = []
        seed_list2 = []
    else:
        seed_list1 = list(np.random.choice(list(alignment_dict.keys()), int(align_train_prop * len(alignment_dict)), replace = False))
        seed_list2 = [alignment_dict[seed_list1[x]] for x in range(len(seed_list1))]
    seed_l1 = seed_list1.copy()
    seed_l2 = seed_list2.copy()
    G1_degree_dict = cal_degree_dict(list(G1.nodes()), G1, layer)
    G2_degree_dict = cal_degree_dict(list(G2.nodes()), G2, layer)
    seed_list_num = len(seed_list1)

    k = seed_link(seed_list1, seed_list2, G1, G2, anchor = anchor)
    print()
    k = np.inf
    test_edges_final1 = np.array(list(set(G1.edges()) - set([(alignment_dict_reversed[edge[0]], alignment_dict_reversed[edge[1]]) for edge in G2.edges()])))
    test_edges_final2 = np.array(list(set(G2.edges()) - set([(alignment_dict[edge[0]], alignment_dict[edge[1]]) for edge in G1.edges()])))
    pred_list1, pred_list2 = [], []

    while True:
        print('------ The current iteration : {} ------'.format(iteration))
        iteration += 1
        index = list(G1.nodes())
        columns = list(G2.nodes())



        index = list(set(index) - set(seed_list1))
        columns = list(set(columns) - set(seed_list2))
            
        columns = [x + mul + 1 for x in columns]

        if k != 0:
            print('structing...', end='')
            G1_degree_dict = cal_degree_dict(list(G1.nodes()), G1, layer)
            G2_degree_dict = cal_degree_dict(list(G2.nodes()), G2, layer)
            struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2 = \
                    structing(layer, G1, G2, G1_degree_dict, G2_degree_dict, attribute, alpha, c)
            print('finished!')
        print('walking...', end='')
        if multi_walk == True:
            multi_simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2, 
                                 struc_neighbor_sim1, struc_neighbor_sim2, 
                                 seed_list1, seed_list2,
                                 num_walks = 20, walk_length = 80, workers = 20)
        else:
            single_simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2, 
                                 struc_neighbor_sim1, struc_neighbor_sim2, 
                                 seed_list1, seed_list2,
                                 num_walks = 20, walk_length = 80, workers = 20)
        walks = LineSentence('random_walks.txt')
        print('finished!')
        print('embedding...', end='')
        model = Word2Vec(walks, size=64, window=5, min_count=0, hs=1, sg=1, workers=32, iter=5)
        print('finished!')
        if len(columns) == 0 or len(index) == 0:
            break
        if len(alignment_dict) == len(seed_list1):
            break
        columns = [x - mul - 1 for x in columns]

        embedding1 = np.array([model.wv[str(x)] for x in index])
        embedding2 = np.array([model.wv[str(x + mul + 1)] for x in columns])


        cos = cosine_similarity(embedding1, embedding2)
        adj_matrix = np.zeros((len(index) * len(columns), 3))
        for i in range(len(index)):
            for j in range(len(columns)):
                adj_matrix[i * len(columns) + j, 0] = index[i]
                adj_matrix[i * len(columns) + j, 1] = columns[j]
                adj_matrix[i * len(columns) + j, 2] = cos[i, j]
        adj_matrix[:, 2] = list(map(clip, adj_matrix[:, 2]))
        if len(seed_list1) != 0:
            adj_matrix2 = caculate_jaccard_coefficient(G1, G2, seed_list1, seed_list2, index, columns)
            adj_matrix[:, 2] *= adj_matrix2[:, 2]
            
        adj_matrix = adj_matrix[np.argsort(-adj_matrix[:, 2])]

        seed1 = []
        seed2 = []
        #np.mean(np.array(seed1)==np.array(seed2))
        len_adj_matrix = len(adj_matrix)
        if len_adj_matrix != 0:            
    
            len_adj_matrix = len(adj_matrix)
            #T = np.max([int(len(alignment_dict) / 100), len(seed_list1)/2])
            T = np.max([5, int(len(alignment_dict) / 100 * (1.5 ** (iteration - 1)))])
            #if len(index_neighbors) == 0:
            #    T = int(len(alignment_dict) / 100)
            #else:
            #    T = max([int(len(alignment_dict) / 100), int(len(index_neighbors)/2)])
            #T = len(alignment_dict) / 10
            while len(adj_matrix) > 0 and T > 0:
                T -= 1
                node1, node2 = int(adj_matrix[0, 0]), int(adj_matrix[0, 1])
                seed1.append(node1)
                seed2.append(node2)
                adj_matrix = adj_matrix[adj_matrix[:, 0] != node1, :]
                adj_matrix = adj_matrix[adj_matrix[:, 1] != node2, :]
            anchor = len(seed_list1)

        anchor = len(seed_list1)
        seed_list1 += seed1
        seed_list2 += seed2
        print('Add seed nodes : {}'.format(len(seed1)), end = '\t')
        
        count = 0
        for i in range(len(seed_list1)):
            try:
                if alignment_dict[seed_list1[i]] == seed_list2[i]:
                    count += 1
            except:
                continue
        print('All seed accuracy : %.2f%%'%(100 * count / len(seed_list1)))
        pred1, pred2 = seed_link_lr(model, G1, G2, seed_list1, seed_list2, 
                                mul, test_edges_final1, test_edges_final2, alignment_dict, alignment_dict_reversed)

        G1.add_edges_from(pred1)
        G2.add_edges_from(pred2)
        
        pred_list1 += list([[alignment_dict[edge[0]], alignment_dict[edge[1]]] for edge in pred1])
        pred_list2 += list([[alignment_dict_reversed[edge[0]], alignment_dict_reversed[edge[1]]] for edge in pred2])
        print('Add seed links: {}'.format(len(pred1) + len(pred2)))
        
        count -= seed_list_num
        precision = 100 * count / (len(seed_list1) - seed_list_num)
        recall = 100 * count / (len(alignment_dict) - seed_list_num)
        
        sub1 = np.sum([G1.has_edge(edge[0], edge[1]) for edge in pred_list2])
        sub2 = np.sum([G2.has_edge(edge[0], edge[1]) for edge in pred_list1])
        if (len(pred_list2) + len(pred_list1)) == 0:
            precision2 = 0
        else:
            precision2 = (sub1 + sub2) / (len(pred_list2) + len(pred_list1)) * 100
        recall2 = (sub1 + sub2) / (len(test_edges_final1) + len(test_edges_final2)) * 100
        print('Precision : %.2f%%\tRecall :  %.2f%%'%(precision, recall))
        print('Link Precision:: %.2f%%\tRecall :  %.2f%%'%(precision2, recall2))
        
    embedding1 = np.array([model.wv[str(x)] for x in list(G1.nodes())])
    embedding2 = np.array([model.wv[str(x + mul + 1)] for x in list(G2.nodes())])
    #adj = np.array(caculate_jaccard_coefficient(G1, G2, seed_list1, seed_list2, list(G1.nodes()), list(G2.nodes())))
    #L = [[adj[i * G1.number_of_nodes() + j, 2] for j in range(G2.number_of_nodes())] for i in range(G1.number_of_nodes())]
    S = cosine_similarity(embedding2, embedding1)
    return S, precision, seed_l1, seed_l2

def cal_degree_dict(G_list, G, layer):
    G_degree = G.degree()
    degree_dict = {}
    degree_dict[0] = {}
    for node in G_list:
        degree_dict[0][node] = {node}
    for i in range(1, layer + 1):
        degree_dict[i] = {}
        for node in G_list:
            neighbor_set = []
            for neighbor in degree_dict[i - 1][node]:
                neighbor_set += nx.neighbors(G, neighbor)
            neighbor_set = set(neighbor_set)
            for j in range(i - 1, -1, -1):
                neighbor_set -= degree_dict[j][node]
            degree_dict[i][node] = neighbor_set
    for i in range(layer + 1):
        for node in G_list:
            if len(degree_dict[i][node]) == 0:
                degree_dict[i][node] = [0]
            else:
                degree_dict[i][node] = node_to_degree(G_degree, degree_dict[i][node])
    return degree_dict

def seed_link(seed_list1, seed_list2, G1, G2, anchor):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([anchor + 1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end = '\t')
    return k
def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET
    
def caculate_jaccard_coefficient(G1, G2, seed_list1, seed_list2, index, columns, alignment_dict = None):
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    seed1_dict = {}
    seed1_dict_reversed = {}
    seed2_dict = {}
    seed2_dict_reversed = {}
    for i in range(len(seed_list1)):
        seed1_dict[i + 2 * (mul + 1)] = seed_list1[i]
        seed1_dict_reversed[seed_list1[i]] = i + 2 * (mul + 1)
        seed2_dict[i + 2 * (mul + 1)] = seed_list2[i] + mul + 1
        seed2_dict_reversed[seed_list2[i] + mul + 1] = i + 2 * (mul + 1)
    G1_edges = pd.DataFrame(G1.edges())
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 1] = G1_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G2_edges = pd.DataFrame(G2.edges())
    G2_edges += mul + 1
    G2_edges.iloc[:, 0] = G2_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed2_dict_reversed))
    G2_edges.iloc[:, 1] = G2_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed2_dict_reversed))
    adj = nx.Graph()
    adj.add_edges_from(np.array(G1_edges))
    adj.add_edges_from(np.array(G2_edges))
    jaccard_dict = {}
    for G1_node in index:
        for G2_node in columns:
            if (G1_node, G2_node) not in jaccard_dict.keys():
                jaccard_dict[G1_node, G2_node] = 0
            jaccard_dict[G1_node, G2_node] += calculate_adj(adj.neighbors(G1_node), adj.neighbors(G2_node + mul + 1))
    
    jaccard_dict = [[x[0][0], x[0][1], x[1]] for x in jaccard_dict.items()]
    adj_matrix = np.array(jaccard_dict)
    return adj_matrix
def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, i in enumerate(input_list):

            if type(i)== list:
                input_list = i + input_list[index+1:]
                break
            else:
                output_list.append(i)
                input_list.pop(index)
                break

    return output_list
def clip(x):
    if x <= 0:
        return 0
    else:
        return x
def calculate_adj(setA, setB):
    setA = set(setA)
    setB = set(setB)
    ep = 0.5
    inter = len(setA & setB) + ep
    union = len(setA | setB) + ep

    adj = inter / union
    return adj
def to_seed(x, dictionary):
    try:
        return dictionary[x]
    except:
        return x
def edge_sample(G):
    edges = list(G.edges())
    test_edges_false = []
    while len(test_edges_false) < G.number_of_edges():
        node1 = np.random.choice(G.nodes())
        node2 = np.random.choice(G.nodes())
        if node1 == node2:
            continue
        if G.has_edge(node1, node2):
            continue
        test_edges_false.append([min(node1, node2), max(node1, node2)])
    edges = edges + test_edges_false
    return edges
def seed_link_lr(model, G1, G2, seed_list1, seed_list2, mul, test_edges_final1, test_edges_final2, alignment_dict, alignment_dict_reversed):

    train_edges_G1 = edge_sample(G1)
    embedding1 = [np.concatenate([model.wv[str(edge[0])], model.wv[str(edge[1])], 
                model.wv[str(edge[0])] * model.wv[str(edge[1])]]) for edge in train_edges_G1]
    label1 = [1] * G1.number_of_edges() + [0] * (len(train_edges_G1) - G1.number_of_edges())
    
    train_edges_G2 = edge_sample(G2)
    embedding2 = [np.concatenate([model.wv[str(edge[0] + mul + 1)], model.wv[str(edge[1] + mul + 1)], 
                model.wv[str(edge[0] + mul + 1)] * model.wv[str(edge[1] + mul + 1)]]) for edge in train_edges_G2]
    label2 = [1] * G2.number_of_edges() + [0] * (len(train_edges_G2) - G2.number_of_edges())
    
    embedding = embedding1 + embedding2
    label = label1 + label2

    edge_classifier = LogisticRegression(solver = 'liblinear', random_state=0)
    edge_classifier.fit(np.array(embedding), label)

    test_edges1 = []
    test_edges2 = []
    for i in range(len(seed_list1)):
        for j in range(i+1, len(seed_list1)):
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                test_edges1.append([min(seed_list1[i], seed_list1[j]), max(seed_list1[i], seed_list1[j])])
            if not G2.has_edge(seed_list2[i], seed_list2[j]) and G1.has_edge(seed_list1[i], seed_list1[j]):
                test_edges2.append([min(seed_list2[i], seed_list2[j]), max(seed_list2[i], seed_list2[j])])
    test_edges1, test_edges2 = np.array(test_edges1), np.array(test_edges2)
    embedding1 = [np.concatenate([model.wv[str(edge[0])], model.wv[str(edge[1])], 
                model.wv[str(edge[0])] * model.wv[str(edge[1])]]) for edge in test_edges1]
    embedding2 = [np.concatenate([model.wv[str(edge[0] + mul + 1)], model.wv[str(edge[1] + mul + 1)], 
                model.wv[str(edge[0] + mul + 1)] * model.wv[str(edge[1] + mul + 1)]]) for edge in test_edges2]
    if len(embedding1) != 0:
        val_preds1 = edge_classifier.predict_proba(embedding1)[:, 1]
        
        pred1 = test_edges1[val_preds1>0.5]
    else:
        pred1 = []
    if len(embedding2) != 0:
        val_preds2 = edge_classifier.predict_proba(embedding2)[:, 1]
        pred2 = test_edges2[val_preds2>0.5]
    else:
        pred2 = []
    '''
    pred1 = [(alignment_dict[edge[0]], alignment_dict[edge[1]]) for edge in pred1]
    pred2 = [(alignment_dict_reversed[edge[0]], alignment_dict_reversed[edge[1]]) for edge in pred2]
    if len(test_edges_final1) != 0:
        test_edges_final1 = [(min([x[0], x[1]]), max([x[0], x[1]])) for x in test_edges_final1]
        #test_edges_final1 = [str(x) for x in test_edges_final1]
        precision1 = 100 * len(np.intersect1d(test_edges_final1, pred2))/len(test_edges_final1)
    else:
        precision1 = 0
    if len(test_edges_final2) != 0:
        test_edges_final2 = [(min([x[0], x[1]]), max([x[0], x[1]])) for x in test_edges_final2]
        #test_edges_final2 = [str(x) for x in test_edges_final2]
        precision2 = 100 * len(np.intersect1d(test_edges_final2, pred1))/len(test_edges_final2)
    else:
        precision2 = 0
    pred1 = [(alignment_dict_reversed[edge[0]], alignment_dict_reversed[edge[1]]) for edge in pred1]
    pred2 = [(alignment_dict[edge[0]], alignment_dict[edge[1]]) for edge in pred2]
    '''
    return pred1, pred2
    