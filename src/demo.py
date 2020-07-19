# -*- coding: utf-8 -*-
from CENALP import CENALP
import pandas as pd
from utils import *
import warnings
import argparse
warnings.filterwarnings('ignore')
def read_alignment(alignment_folder, filename):
    alignment = pd.read_csv(alignment_folder + filename + '.csv', header = None)
    alignment_dict = {}
    alignment_dict_reversed = {}
    for i in range(len(alignment)):
        alignment_dict[alignment.iloc[i, 0]] = alignment.iloc[i, 1]
        alignment_dict_reversed[alignment.iloc[i, 1]] = alignment.iloc[i, 0]
    return alignment_dict, alignment_dict_reversed

def read_attribute(attribute_folder, filename, G1, G2):
    try:
        attribute, attr1, attr2 = load_attribute(attribute_folder, filename, G1, G2)
        attribute = attribute.transpose()
    except:
        attr1 = []
        attr2 = []
        attribute = []
        print('Attribute files not found.')
    return attribute, attr1, attr2
def parse_args():
    '''
    Parses the CLF arguments.
    '''
    parser = argparse.ArgumentParser(description="Run CENALP.")
    parser.add_argument('--attribute_folder', nargs='?', default='../attribute/', 
                        help='Input attribute path')
    
    parser.add_argument('--data_folder', nargs='?', default='../graph/', 
                        help='Input graph data path')
    parser.add_argument('--alignment_folder', nargs='?', default='../alignment/', 
                        help='Input ground truth alignment path')
    parser.add_argument('--filename', nargs='?', default='bigtoy',
                        help='Name of file')
    parser.add_argument('--alpha', type=int, default=5,
                        help="Hyperparameter controlling the distribution of vertices' similairties")
    parser.add_argument('--layer', type=int, default=3,
                        help="Depth of neighbors")
    parser.add_argument('--align_train_prop', type=float, default=0.0,
                        help="Proportion of training set. 0 represents unsupervised learning.")
    parser.add_argument('--q', type=float, default=0.5,
                        help="Probability of walking to the separate network during random walk")    
    parser.add_argument('--c', type=float, default=0.5,
                        help="Weight between sub-graph similarity and attribute similarity")    
    parser.add_argument('--multi_walk', type=bool, default=False,
                        help="Whether to use multi-processing")    
    return parser.parse_args()

def main(args):
    alignment_dict, alignment_dict_reversed = read_alignment(args.alignment_folder, args.filename)
    G1, G2 = loadG(args.data_folder, args.filename)
    attribute, attr1, attr2 = read_attribute(args.attribute_folder, args.filename, G1, G2)
    S, precision, seed_l1, seed_l2 = CENALP(G1, G2, args.q, attr1, attr2, attribute, alignment_dict, alignment_dict_reversed, 
       args.layer, args.align_train_prop, args.alpha, args.c, args.multi_walk)
if __name__ == '__main__':
    args = parse_args()
    main(args)
    

    
    
    