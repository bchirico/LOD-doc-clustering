__author__ = 'biagio'

import document_processor as dp
import pprint as pp
import numpy as np
import itertools
from scipy.cluster import hierarchy as hr
import argparse

def scipy_algo(dataset):
    doc_proc = dp.DocumentsProcessor(dataset)
    tfidf_matrix, f_score_dict = doc_proc.get_data()

    linkage_matrix = hr.average(tfidf_matrix.toarray())

    t = hr.to_tree(linkage_matrix, rd=True)

    clusters = {}

    for node in t[1]:
        if not node.is_leaf():
            l = []
            clusters[node.get_id()] = collect_leaf_nodes(node, l)

    f = f_score(clusters, f_score_dict)

    print_f_score_dict(f)

    print 'average f_score: %s' % average_f_score(f, tfidf_matrix.shape[0])

def collect_leaf_nodes(node, leaves):
    if node is not None:
        if node.is_leaf():
            leaves.append(node.get_id())
        for n in [node.get_left(), node.get_right()]:
            collect_leaf_nodes(n, leaves)
    return leaves

def print_f_score_dict(f):
    pp.pprint([f[d]['fscore'] for d in f])

def average_f_score(f_score_dict, n):
    f_score = [f_score_dict[d]['fscore'] for d in f_score_dict]
    average = 0
    n = np.float_(n)
    for d in f_score_dict:
        nr = np.float_(len(f_score_dict[d]['docs']))
        f = nr / n * f_score_dict[d]['fscore']
        #print n, nr, f, f_score_dict[d]['fscore']
        average += f
    return average

def f_score(clusters, f_score_dict):
    for cl in f_score_dict:
        for k,v in clusters.iteritems():
            docs = np.array(f_score_dict[cl]['docs'])
            cluster = np.array(v)
            nri = float(np.intersect1d(cluster, docs).shape[0])
            nr = float(docs.shape[0])
            ni = float(cluster.shape[0])

            try:
                recall = nri / nr
                precision = nri / ni
                f_score = (2 * precision * recall) / (precision + recall)
                f_score_dict[cl]['fscore'] = f_score if (f_score > f_score_dict[cl]['fscore'] or not f_score_dict[cl]['fscore']) else f_score_dict[cl]['fscore']
            except ZeroDivisionError, e:
                pass

    return f_score_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script that performs and evaluate hierachical clustering')

    parser.add_argument('-d',
                        dest='dataset',
                        help='Dataset name',
                        required=True,
                        choices=['re0', 're1'])
    args = parser.parse_args()

    dataset = args.dataset

    scipy_algo(dataset)
