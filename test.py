__author__ = 'biagio'

from mongo_hc import MongoHC
import classifier as clf
import pprint as pp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from text_utils import TextUtils
import argparse

mongo = MongoHC('hc', 'test_new')


def first_test_re0():
    for i, g in enumerate(np.arange(0.1, 1, 0.01)):
        result = clf.cluster_alchemy('re1', gamma=g, filter=True)
        pp.pprint(result)
        result['n_attempt'] = i+1
        result['test'] = 'fourth'
        mongo.save_document(result)


def test_without_entity():
    result = clf.cluster_alchemy('re1', gamma=1)
    pp.pprint(result)
    result['test'] = 'baseline'
    mongo.save_document(result)


def test_bow():
    result = clf.scipy_algo('re0')


def test_dandelion(dataset):
    for i, g in enumerate(np.arange(0.7, 1, 0.01)):
        result = clf.cluster_dandelion(dataset, gamma=g)
        pp.pprint(result)
        result['n_attempt'] = i+1
        result['test'] = 'second_dandelion_re1'
        result['dataset'] = '{0}_abstract_lsa'.format(dataset)
        mongo.save_document(result)

def test_only_entities(dataset):
    for i, g in enumerate(np.arange(0.7, 1, 0.01)):
        result = clf.cluster_dandelion_entities(dataset, gamma=g)
        pp.pprint(result)
        result['n_attempt'] = i+1
        result['test'] = 'only_entities'
        result['dataset'] = '{0}_abstract_lsa'.format(dataset)
        mongo.save_document(result)


def test_text_vectorization():
    mongo_dataset = MongoHC('hc', 're0')
    data = [d for d in mongo_dataset.get_all(order_by='id_doc')]
    text = [d['text'] for d in data[1:2]]
    tfidf_vectorizer = TfidfVectorizer(max_df=1,
                                       max_features=200000,
                                       min_df=1,
                                       stop_words='english',
                                       strip_accents='unicode',
                                       use_idf=True,
                                       ngram_range=(1, 1),
                                       norm='l2')
    tfidf_matrix = tfidf_vectorizer.fit_transform(text)
    print tfidf_vectorizer.get_feature_names()
    print tfidf_matrix.data

    indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
    print indices
    features = tfidf_vectorizer.get_feature_names()
    top_n = 5
    top_features = [features[i] for i in indices[:top_n]]

    print len(features)
    print tfidf_matrix.shape
    print top_features


def test_fabio(db, dataset, gamma=0.5, ranking_metric='pr', lsa=False, save=False):
    mongo_result = MongoHC(db, 'test_fabio')
    result = clf.cluster_fabio(db, dataset, gamma=gamma,
                               ranking_metric=ranking_metric,
                               with_lsa=lsa)
    if save:
        mongo_result.save_document(result)
    pp.pprint(result)


def madness():
    result = clf.cluster_dandelion('re1', gamma=0.93)
    pp.pprint(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gamma',
                        dest='gamma',
                        help = 'weight parameter',
                        type=float,
                        required=False,
                        default=0.5)
    parser.add_argument('-d',
                        dest='db',
                        help='db name',
                        required=False,
                        default='hc_fabio')
    parser.add_argument('-c',
                        dest='collection',
                        help='collection name',
                        required=True,
                        choices=['re0', 're1'])
    parser.add_argument('-r', '--rank-param',
                        dest='rank_param',
                        help='ranking metric',
                        required=False,
                        choices=['r', 'pr'],
                        default='r')
    parser.add_argument('-l', '--lsa',
                        dest='lsa',
                        help='lsa technique',
                        required=False,
                        action='store_true')
    parser.add_argument('-s', '--save',
                        dest='store_mongo',
                        help='save result on MongoDB',
                        required=False,
                        action='store_true')

    args = parser.parse_args()

test_fabio(args.db,
           args.collection,
           gamma=args.gamma,
           ranking_metric=args.rank_param,
           lsa=args.lsa,
           save=args.store_mongo)


