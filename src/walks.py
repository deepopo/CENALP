# -*- coding: utf-8 -*-
from collections import deque
import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def matrix_walk(A, node, walk_length = 80):
    n = A.shape[1]
    walk = []
    walk.append(node)
    for i in range(walk_length - 1):
        node = random.choices(list(range(n)), A[node])[0]
        walk.append(node)
    return walk
    
def matrix_walks(A, walk_length = 80):
    walks = deque()
    for i in range(len(A)):
        walks.append(matrix_walk(A, i, walk_length))
    return walks

def multi_matrix_walks(A, num_walks = 20, walk_length = 80, workers = 20):
    walks = deque()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for walk_iter in range(num_walks):
            job = executor.submit(matrix_walks, A, walk_length)
            futures[job] = walk_iter
        
        for job in as_completed(futures):
            walk = job.result()
            walks.extend(walk)
        del futures
    with open('random_walks.txt', 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v)+' '
            line += '\n'
            file.write(line)
def simulate_walk(G1, G2, q, struc_neighbor1, struc_neighbor2, 
                         struc_neighbor_sim1, struc_neighbor_sim2, 
                         seed_list1 = [], seed_list2 = [],
                         walk_length = 80, node = 0, curr_graph = 1
                  ):
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    walk = deque()
    walk.append(node)
    for i in range(walk_length - 1):
        r = np.random.random()
        if r < q:
            if curr_graph == 1:
                list_Gneighbors = list(G1.neighbors(node))
                node = np.random.choice(list_Gneighbors)
            elif curr_graph == 2:
                list_Gneighbors = list(G2.neighbors(node - mul - 1))
                node = np.random.choice(list_Gneighbors) + mul + 1
        else:
            if curr_graph == 1:
                try:
                    node = seed_list2[seed_list1.index(node)] + mul + 1
                except:
                    node = random.choices(population=struc_neighbor1[node], weights = struc_neighbor_sim1[node])[0] + mul + 1
                curr_graph = 2
            else:
                try:
                    node = seed_list1[seed_list2.index(node - mul - 1)]
                except:
                    node = random.choices(population=struc_neighbor2[node - mul - 1], weights = struc_neighbor_sim2[node - mul - 1])[0]
                curr_graph = 1
        walk.append(node)
    return walk

def simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2, 
                         struc_neighbor_sim1, struc_neighbor_sim2, 
                         G1_list, G2_list, seed_list1 = [], seed_list2 = [],
                         walk_length = 80):
    walks = deque()
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    for node in G1_list:
        
        walks.append(simulate_walk(G1, G2, q, struc_neighbor1, struc_neighbor2, 
                         struc_neighbor_sim1, struc_neighbor_sim2, 
                         seed_list1 = [], seed_list2 = [], 
                         walk_length = 80, node = node, curr_graph = 1))
    for node in G2_list:
        node += mul + 1
        walks.append(simulate_walk(G1, G2, q, struc_neighbor1, struc_neighbor2, 
                         struc_neighbor_sim1, struc_neighbor_sim2, 
                         seed_list1 = [], seed_list2 = [], 
                         walk_length = 80, node = node, curr_graph = 2))
    return walks

def multi_simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2, 
                         struc_neighbor_sim1, struc_neighbor_sim2, 
                         seed_list1 = [], seed_list2 = [],
                         num_walks = 20, walk_length = 80, workers = 20):

    walks = deque()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        G1_list = list(G1.nodes())
        G2_list = list(G2.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(G1_list)
            random.shuffle(G2_list)
            job = executor.submit(simulate_walks, G1, G2, q, struc_neighbor1, struc_neighbor2, 
                         struc_neighbor_sim1, struc_neighbor_sim2, 
                         G1_list, G2_list, seed_list1 = [], seed_list2 = [],
                         walk_length = 80)
            futures[job] = walk_iter
            #part += 1
        
        for job in as_completed(futures):
            walk = job.result()
            walks.extend(walk)
        del futures
    with open('random_walks.txt', 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v)+' '
            line += '\n'
            file.write(line)
            
def single_simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2, 
                         struc_neighbor_sim1, struc_neighbor_sim2, 
                         seed_list1 = [], seed_list2 = [], 
                         num_walks = 20, walk_length = 80, workers = 20):
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    walks = []
    for i in tqdm(range(num_walks)):
        for node in list(G1.nodes()):

            walk = []
            walk.append(node)
            curr_graph = 1

            for j in range(walk_length - 1):
                r = np.random.random()
                if r < q:
                    if curr_graph == 1:
                        list_Gneighbors = list(G1.neighbors(node))
                        node = np.random.choice(list_Gneighbors)
                    elif curr_graph == 2:
                        list_Gneighbors = list(G2.neighbors(node - mul - 1))
                        node = np.random.choice(list_Gneighbors) + mul + 1
                else:
                    if curr_graph == 1:
                        try:
                            node = seed_list2[seed_list1.index(node)] + mul + 1
                        except:
                            node = random.choices(population=struc_neighbor1[node], weights = struc_neighbor_sim1[node])[0] + mul + 1
                        curr_graph = 2
                    else:
                        try:
                            node = seed_list1[seed_list2.index(node - mul - 1)]
                        except:
                            node = random.choices(population=struc_neighbor2[node - mul - 1], weights = struc_neighbor_sim2[node - mul - 1])[0]
                        curr_graph = 1
                walk.append(node)
            walks.append(walk)
        for node in list(G2.nodes()):
            node += mul + 1
            walk = []
            walk.append(node)
            curr_graph = 2

            for j in range(walk_length - 1):
                r = np.random.random()
                if r < q:
                    if curr_graph == 1:
                        list_Gneighbors = list(G1.neighbors(node))
                        node = np.random.choice(list_Gneighbors)
                    elif curr_graph == 2:
                        list_Gneighbors = list(G2.neighbors(node - mul - 1))
                        node = np.random.choice(list_Gneighbors) + mul + 1
                else:
                    if curr_graph == 1:
                        try:
                            node = seed_list2[seed_list1.index(node)] + mul + 1
                        except:
                            node = random.choices(population=struc_neighbor1[node], weights = struc_neighbor_sim1[node])[0] + mul + 1
                        curr_graph = 2
                    else:
                        try:
                            node = seed_list1[seed_list2.index(node - mul - 1)]
                        except:
                            node = random.choices(population=struc_neighbor2[node - mul - 1], weights = struc_neighbor_sim2[node - mul - 1])[0]
                        curr_graph = 1
                walk.append(node)
            walks.append(walk)
    with open('random_walks.txt', 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v)+' '
            line += '\n'
            file.write(line)
            