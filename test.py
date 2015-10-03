__author__ = 'biagio'

from mongo_hc import MongoHC
import classifier as clf
import pprint as pp
import numpy as np

mongo = MongoHC('hc', 'test')


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
    result = clf.scipy_algo('re1')


def test_dandelion():
    for i, g in enumerate(np.arange(0.1, 1, 0.01)):
        result = clf.cluster_dandelion('re1', gamma=g)
        pp.pprint(result)
        result['n_attempt'] = i+1
        result['test'] = 'second_dandelion_re1'
        result['dataset'] = 're1'
        mongo.save_document(result)


def madness():
    result = clf.cluster_dandelion('re1', gamma=0.93)
    pp.pprint(result)

if __name__ == '__main__':
    test_without_entity()

