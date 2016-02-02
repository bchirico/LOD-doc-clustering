import pprint as pp
import numpy as np
from scipy.cluster import hierarchy as hr
import argparse
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import document_processor as dp

__author__ = 'biagio'


# TODO: refactoring delle funzioni, c'e' troppa ripetizione e non ho voglia di
#       cambiare tutto se faccio delle modifiche

def scipy_algo(dataset, abstract=False):
    doc_proc = dp.DocumentsProcessor(dataset)
    tfidf_matrix, f_score_dict = doc_proc.get_data(abstract)

    svd = TruncatedSVD(tfidf_matrix.shape[0])
    lsa = make_pipeline(svd, Normalizer(copy=False))

    #tfidf_matrix = lsa.fit_transform(tfidf_matrix)

    print 'starting clustering after lsa: found %s document and %s features' \
          % (tfidf_matrix.shape[0], tfidf_matrix.shape[1])

    linkage_matrix = hr.average(tfidf_matrix.toarray())
    #linkage_matrix = hr.average(tfidf_matrix)

    t = hr.to_tree(linkage_matrix, rd=True)

    clusters = {}

    for node in t[1]:
        if not node.is_leaf():
            l = []
            clusters[node.get_id()] = collect_leaf_nodes(node, l)

    f = f_score(clusters, f_score_dict)

    print_f_score_dict(f)

    avg_f_score = average_f_score(f, tfidf_matrix.shape[0])
    print 'average f_score: %s' % avg_f_score
    return avg_f_score


def scipy_algo_cosine(dataset, abstract=False):
    '''
    al momento non riesco a clusterizzare utilizzando la cosine_similarity
    perche' sembra esserci un bug di scipy che ritorna una distanza negativa

    https://github.com/scipy/scipy/issues/5208

    :param dataset:
    :param abstract:
    :return:
    '''
    doc_proc = dp.DocumentsProcessor(dataset)
    tfidf_matrix, f_score_dict = doc_proc.get_data(abstract)

    linkage_matrix = hr.linkage(tfidf_matrix.todense(),
                                method='average',
                                metric='cosine')

    print linkage_matrix.shape
    print linkage_matrix[linkage_matrix < 0].shape
    return

    t = hr.to_tree(linkage_matrix, rd=True)

    clusters = {}

    for node in t[1]:
        if not node.is_leaf():
            l = []
            clusters[node.get_id()] = collect_leaf_nodes(node, l)

    f = f_score(clusters, f_score_dict)

    print_f_score_dict(f)

    print 'average f_score: %s' % average_f_score(f, tfidf_matrix.shape[0])


def cluster_alchemy(dataset, gamma=None, filter=False):
    doc_proc = dp.DocumentsProcessor(dataset)
    if gamma:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_with_alchemy(gamma=gamma, filter=filter)
    else:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_with_alchemy()

    print 'starting clustering: found %s document and %s features' \
          % (tfidf_matrix.shape[0], tfidf_matrix.shape[1])

    linkage_matrix = hr.average(tfidf_matrix.toarray())

    t = hr.to_tree(linkage_matrix, rd=True)

    clusters = {}

    for node in t[1]:
        if not node.is_leaf():
            l = []
            clusters[node.get_id()] = collect_leaf_nodes(node, l)

    f = f_score(clusters, f_score_dict)

    l = print_f_score_dict(f)

    params['avg_f_score'] = average_f_score(f, tfidf_matrix.shape[0])
    params['all_fscore'] = l

    print 'average f_score: %s' % params['avg_f_score']
    return params


def cluster_dandelion(dataset, gamma=0.89, filter=False):
    doc_proc = dp.DocumentsProcessor(dataset)
    if gamma:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_with_dandelion(
            gamma=gamma, filter=filter)
    else:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_with_dandelion()

    print 'starting clustering: found %s document and %s features' \
          % (tfidf_matrix.shape[0], tfidf_matrix.shape[1])

    svd = TruncatedSVD(tfidf_matrix.shape[0])
    lsa = make_pipeline(svd, Normalizer(copy=False))

    tfidf_matrix = lsa.fit_transform(tfidf_matrix)

    print 'starting clustering after lsa: found %s document and %s features' \
          % (tfidf_matrix.shape[0], tfidf_matrix.shape[1])

    #linkage_matrix = hr.average(tfidf_matrix.toarray())
    linkage_matrix = hr.average(tfidf_matrix)

    t = hr.to_tree(linkage_matrix, rd=True)

    clusters = {}

    for node in t[1]:
        if not node.is_leaf():
            l = []
            clusters[node.get_id()] = collect_leaf_nodes(node, l)

    f = f_score(clusters, f_score_dict)

    l = print_f_score_dict(f)

    params['avg_f_score'] = average_f_score(f, tfidf_matrix.shape[0])
    params['all_fscore'] = l

    print 'average f_score: %s' % params['avg_f_score']
    return params


def cluster_dandelion_2(dataset, gamma=0.91, filter=False):
    #duplicato, mi serve solo per tornare la linkage_matrix
    doc_proc = dp.DocumentsProcessor(dataset)
    if gamma:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_with_dandelion(
            gamma=gamma, filter=filter)
    else:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_with_dandelion()

    svd = TruncatedSVD(tfidf_matrix.shape[0])
    lsa = make_pipeline(svd, Normalizer(copy=False))

    tfidf_matrix = lsa.fit_transform(tfidf_matrix)

    #linkage_matrix = hr.average(tfidf_matrix.toarray())
    linkage_matrix = hr.average(tfidf_matrix)

    t = hr.to_tree(linkage_matrix, rd=True)

    clusters = {}

    for node in t[1]:
        if not node.is_leaf():
            l = []
            clusters[node.get_id()] = collect_leaf_nodes(node, l)

    f = f_score(clusters, f_score_dict)

    l = print_f_score_dict(f)

    params['avg_f_score'] = average_f_score(f, tfidf_matrix.shape[0])
    params['all_fscore'] = l

    return linkage_matrix


def cluster_dandelion_abstract(dataset, gamma=None, filter=False):
    doc_proc = dp.DocumentsProcessor(dataset)
    if gamma:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_only_with_abstract(
            gamma=gamma, filter=filter)
    else:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_only_with_abstract(min_df=2, relevance_threshold=0.95)

    doc, features = tfidf_matrix.shape

    print 'starting clustering: found %s document and %s features' \
          % (doc, features)

    svd = TruncatedSVD(1300)
    lsa = make_pipeline(svd, Normalizer(copy=False))

    tfidf_matrix = lsa.fit_transform(tfidf_matrix)

    print 'starting clustering: found %s document and %s features after LSA' \
         % (tfidf_matrix.shape[0], tfidf_matrix.shape[1])

    #linkage_matrix = hr.average(tfidf_matrix.toarray())
    linkage_matrix = hr.average(tfidf_matrix)

    t = hr.to_tree(linkage_matrix, rd=True)

    clusters = {}

    for node in t[1]:
        if not node.is_leaf():
            l = []
            clusters[node.get_id()] = collect_leaf_nodes(node, l)

    f = f_score(clusters, f_score_dict)

    l = print_f_score_dict(f)

    params['avg_f_score'] = average_f_score(f, tfidf_matrix.shape[0])
    params['all_fscore'] = l

    print 'average f_score: %s' % params['avg_f_score']
    return params

def cluster_dandelion_entities(dataset, gamma=None, filter=False):
    doc_proc = dp.DocumentsProcessor(dataset)
    if gamma:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_only_with_entities(gamma=gamma, filter=filter)
    else:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_only_with_entities()

    doc, features = tfidf_matrix.shape

    print 'starting clustering: found %s document and %s features' \
          % (doc, features)

    svd = TruncatedSVD(tfidf_matrix.shape[0])
    lsa = make_pipeline(svd, Normalizer(copy=False))

    tfidf_matrix = lsa.fit_transform(tfidf_matrix)

    print 'starting clustering: found %s document and %s features after LSA' \
         % (tfidf_matrix.shape[0], tfidf_matrix.shape[1])

    #linkage_matrix = hr.average(tfidf_matrix.toarray())
    linkage_matrix = hr.average(tfidf_matrix)

    t = hr.to_tree(linkage_matrix, rd=True)

    clusters = {}

    for node in t[1]:
        if not node.is_leaf():
            l = []
            clusters[node.get_id()] = collect_leaf_nodes(node, l)

    f = f_score(clusters, f_score_dict)

    l = print_f_score_dict(f)

    params['avg_f_score'] = average_f_score(f, tfidf_matrix.shape[0])
    params['all_fscore'] = l

    print 'average f_score: %s' % params['avg_f_score']
    return params

def cluster_fabio(db, dataset, gamma=None, with_lsa=False, ranking_metric='r'):
    doc_proc = dp.DocumentsProcessor(dataset, db=db)
    if gamma:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_fabio(
            rank_metric=ranking_metric, gamma=gamma)
    else:
        tfidf_matrix, f_score_dict, params = doc_proc.get_data_fabio(
            rank_metric=ranking_metric)

    doc, features = tfidf_matrix.shape

    print 'starting clustering: found %s document and %s features' \
          % (doc, features)

    if with_lsa:
        svd = TruncatedSVD(tfidf_matrix.shape[0])
        lsa = make_pipeline(svd, Normalizer(copy=False))

        tfidf_matrix = lsa.fit_transform(tfidf_matrix)
        linkage_matrix = hr.average(tfidf_matrix)
    else:
        linkage_matrix = hr.average(tfidf_matrix.toarray())

    t = hr.to_tree(linkage_matrix, rd=True)

    clusters = {}

    for node in t[1]:
        if not node.is_leaf():
            l = []
            clusters[node.get_id()] = collect_leaf_nodes(node, l)

    f = f_score(clusters, f_score_dict)

    l = print_f_score_dict(f)

    params['avg_f_score'] = average_f_score(f, tfidf_matrix.shape[0])
    params['all_fscore'] = l

    print 'average f_score: %s' % params['avg_f_score']
    return params


def collect_leaf_nodes(node, leaves):
    if node is not None:
        if node.is_leaf():
            leaves.append(node.get_id())
        for n in [node.get_left(), node.get_right()]:
            collect_leaf_nodes(n, leaves)
    return leaves


def print_f_score_dict(f):
    l = [f[d]['fscore'] for d in f]
    pp.pprint(l)
    return l


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

    parser.add_argument('--abstract', '-a',
                        dest='abstract',
                        help='specify action to perform',
                        required=False,
                        action='store_true')

    parser.add_argument('--entities', '-e',
                        dest='entities',
                        help='specify action to perform',
                        required=False,
                        action='store_true')

    parser.add_argument('--alchemy',
                        dest='alchemy',
                        help='Cluster solutions is obtained with BOW method and'
                             ' entities\'relevance extracted with alchemyAPI',
                        required=False,
                        action='store_true')

    parser.add_argument('--dandelion',
                    dest='dandelion',
                    help='Cluster solutions is obtained with BOW method and'
                         ' entities\'relevance extracted with dataTXT',
                    required=False,
                    action='store_true')

    args = parser.parse_args()

    dataset = args.dataset

    if args.alchemy:
        cluster_alchemy(dataset)
    elif args.dandelion:
        cluster_dandelion(dataset)
    elif args.abstract:
        cluster_dandelion_abstract(dataset)
    elif args.entities:
        cluster_dandelion_entities(dataset)
