# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
def structing(layers, G1, G2, G1_degree_dict, G2_degree_dict, attribute, alpha, c):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())

    k1 = k2 = 1
    pp_dist_matrix = {}
    pp_dist_df = pd.DataFrame(np.zeros((G1.number_of_nodes(), G2.number_of_nodes())), 
                                       index=G1_nodes, columns=G2_nodes)
    
    for layer in range(layers + 1):

        L1 = [np.log(k1 * np.max(G1_degree_dict[layer][x]) + np.e) for x in G1_nodes]
        L2 = [np.log(k2 * np.max(G2_degree_dict[layer][x]) + np.e) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())), 
                          index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()), 
                          index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])
    for layer in range(layers + 1):

        L1 = [np.log(k1 * np.min(G1_degree_dict[layer][x]) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.min(G2_degree_dict[layer][x]) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())), 
                          index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()), 
                          index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])
    pp_dist_df /= 2
    pp_dist_df = np.exp(-alpha * pp_dist_df)
    if len(attribute) != 0:
        pp_dist_df = c * pp_dist_df + np.array(attribute) * (1-c)
    struc_neighbor1 = {}
    struc_neighbor2 = {}
    struc_neighbor_sim1 = {}
    struc_neighbor_sim2 = {}
    for i in range(G1.number_of_nodes()):
        pp = pp_dist_df.iloc[i, np.argsort(-pp_dist_df.iloc[i, :])]
        struc_neighbor1[G1_nodes[i]] = list(pp.index[:10])
        struc_neighbor_sim1[G1_nodes[i]] = np.array(pp[:10])
        struc_neighbor_sim1[G1_nodes[i]] /= np.sum(struc_neighbor_sim1[G1_nodes[i]])
    pp_dist_df = pp_dist_df.transpose()
    for i in range(G2.number_of_nodes()):
        pp = pp_dist_df.iloc[i, np.argsort(-pp_dist_df.iloc[i, :])]
        struc_neighbor2[G2_nodes[i]] = list(pp.index[:10])
        struc_neighbor_sim2[G2_nodes[i]] = np.array(pp[:10])
        struc_neighbor_sim2[G2_nodes[i]] /= np.sum(struc_neighbor_sim2[G2_nodes[i]])
    return struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2
