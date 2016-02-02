import argparse
from scipy.stats.mstats_basic import threshold
import config
import itertools
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn import decomposition
from sklearn.cluster import KMeans
import itertools

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

import pprint as pp
import json
import numpy as np
import text_utils as tu
from text_utils import TextUtils
import mongo_hc
import warnings

__author__ = 'biagio'

'''
--------------------------------------------------------------------------------
test1:  tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=200000,
                         min_df=2, stop_words='english',
                         use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem,
                         ngram_range=(1, 1))
        re0 average f_score = 0.638534830083
        re1 average f_score = 0.765673119777

test2:  tfidf_vectorizer = TfidfVectorizer(max_df=0.6, max_features=200000,
                         min_df=2, stop_words='english',
                         use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem,
                         ngram_range=(1, 1))
        re0 average f_score =  0.658152070681
        re1 average f_score = 0.772388663679

test3:  tfidf_vectorizer = TfidfVectorizer(max_df=0.65, max_features=200000,
                     min_df=2, stop_words='english',
                     use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem,
                     ngram_range=(1, 1))
        re0 average f_score =
        re1 average f_score = 0.772388663679

test4:  tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=200000,
                     min_df=2, stop_words='english',
                     use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem,
                     ngram_range=(1, 1))
        re0 average f_score =
        re1 average f_score = 0.772388663679

test5:  tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                     min_df=2, stop_words='english',
                     use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem,
                     ngram_range=(1, 1))
        re0 average f_score =
        re1 average f_score = 0.772388663679

test6:  tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                     min_df=2, stop_words='english',
                     use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem,
                     ngram_range=(1, 1))
        re0 average f_score = 0.657453792372
        re1 average f_score = 0.77630555331

test7:  tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                     min_df=0.005, stop_words='english',
                     use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem,
                     ngram_range=(1, 1))
        re0 average f_score = 0.650254341531
        re1 average f_score = 0.79978824548

test8:  tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                     min_df=0.006, stop_words='english',
                     use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem,
                     ngram_range=(1, 1))
        re0 average f_score = 0.626006569483
        re1 average f_score = 0.781790453423

test9:  tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                     min_df=0.004, stop_words='english',
                     use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem,
                     ngram_range=(1, 1))
        re0 average f_score = 0.628345214855
        re1 average f_score = 0.785153400931
--------------------------------------------------------------------------------
'''

#TODO: questo ha la priorita' assoluta, c'e' troppo casino e codice ripetuto

class DocumentsProcessor:
    def __init__(self, dataset_name, db='hc'):
        files_name = ['re0, re1']
        self.dataset_name = dataset_name
        self.mongo = mongo_hc.MongoHC(db, self.dataset_name)
        self.dbpedia = mongo_hc.MongoHC(db, 'dbpedia')
        self.tokenizer = TextUtils.tokenize_and_stem

    @property
    def data(self):
        '''
        Property that read data from JSON file

        :return: data
         dictionary that contatain two keys:
            data    -> all the documents' text
            label   -> all the documents' label in the same order that data
        '''
        data = None
        with open(config.PRE_PROCESSED_DATASETS + self.dataset_name + '.json',
                  'r') as f:
            data = json.load(f)
        f.close()
        return data

    def get_data_with_abstract(self, data):
        only_text = []
        for doc in data:
            text = doc['text']
            if 'abstracts' in doc:
                for abs in doc['abstracts']:
                    text += '\n'
                    text += abs['value']
            only_text.append(text)
        return only_text

    def get_data(self, abstract=False):
        data = self.mongo.get_all(order_by='id_doc')
        data = [doc for doc in data]

        if abstract:
            only_text = self.get_data_with_abstract(data)
        else:
            only_text = [doc['text'] for doc in data]

        only_labels = [doc['label'] for doc in data]
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
                                           max_features=200000,
                                           min_df=2,
                                           stop_words='english',
                                           strip_accents='unicode',
                                           use_idf=True,
                                           ngram_range=(1, 1),
                                           norm='l2',
                                           tokenizer=TextUtils.tokenize_and_stem)
        tfidf_matrix = tfidf_vectorizer.fit_transform(only_text)
        print 'After tfidf vectorizer: found %s documents and %s terms' \
              % (tfidf_matrix.shape[0], tfidf_matrix.shape[1])
        dict_out = {}
        for l in sorted(set(only_labels)):
            dict_out[l] = {
                'docs': [],
                'fscore': ''
            }
        for doc in data:
            dict_out[doc['label']]['docs'].append(doc['id_doc'])

        return tfidf_matrix, dict_out

    def entities_distribution(self, d):
        data = [doc for doc in d]

        entities = set()

        for d in data:
            for e in d['alchemy_response']['entities']:
                entities.add(e['text'])

        entities_dict = {e: 0 for i, e in enumerate(entities)}

        for d in data:
            for e in d['alchemy_response']['entities']:
                entities_dict[e['text']] += 1

        return entities_dict, entities

    def get_data_with_alchemy(self, relevance_threshold=0.8, min_df=0.003,
                              gamma=0.89, filter=False):
        print gamma
        data = self.mongo.get_all(order_by='id_doc')

        data = [doc for doc in data]
        only_text = [doc['text'] for doc in data]

        ent_dict, ent_set = self.entities_distribution(data)

        if filter:
            entities_set = set([k for k, v in ent_dict.iteritems() if (v > 2 and v < 300)])
        else:
            entities_set = ent_set

        '''entities_name = []
        for doc in data:
            if 'alchemy_response' in doc:
                for e in doc['alchemy_response']['entities']:
                    entities_name.append(e['text'])
        entities_set = set(entities_name)'''
        entities = {e: i for i, e in enumerate(entities_set)}
        alchemy_entities = np.zeros((len(data), len(entities_set)))
        for doc in data:
            if 'alchemy_response' in doc:
                for e in doc['alchemy_response']['entities']:
                    rel = np.float64(e['relevance'])
                    name = e['text']
                    if rel > relevance_threshold and name in entities:
                        alchemy_entities[doc['id_doc']][entities[name]] = rel

        entities_sparse = sparse.csr_matrix(alchemy_entities)

        tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
                                           max_features=200000,
                                           min_df=min_df,
                                           stop_words='english',
                                           strip_accents='unicode',
                                           use_idf=True,
                                           ngram_range=(1, 1),
                                           norm='l2',
                                           tokenizer=TextUtils.tokenize_and_stem)
        tfidf_matrix = tfidf_vectorizer.fit_transform(only_text)

        print 'tfifd matrix dimension: %s x %s' %(tfidf_matrix.shape[0],
                                                  tfidf_matrix.shape[1])
        print 'alchemy matrix dimension: %s x %s ' %(entities_sparse.shape[0],
                                                     entities_sparse.shape[1])
        print 'non zero elements in alchemy matrix: %s' \
              % len(entities_sparse.data)

        '''print tfidf_matrix[tfidf_matrix > 0].mean()
        print tfidf_matrix[tfidf_matrix > 0].max()

        print entities_sparse[entities_sparse > 0].mean()
        print entities_sparse[entities_sparse > 0].max()
        print '#' * 80'''
        #print 'after balancing'

        tfidf_matrix = tfidf_matrix * gamma
        entities_sparse = entities_sparse * (1 - gamma)

        #print tfidf_matrix[tfidf_matrix > 0].mean()
        #print tfidf_matrix[tfidf_matrix > 0].max()

        #print entities_sparse[entities_sparse > 0].mean()
        #print entities_sparse[entities_sparse > 0].max()

        f_score_dict = self.labels_dict(data)
        params = tfidf_vectorizer.get_params()
        params['alchemy_entities'] = entities_sparse.shape[1]
        params['original_terms'] = tfidf_matrix.shape[0]
        params['gamma'] = gamma
        params['relevance_threshold'] = relevance_threshold
        params['classes'] = len(f_score_dict)
        params['tokenizer'] = 'TextUtils.tokenize_and_stem'
        del params['dtype']

        params['avg_nnz_row'] = (entities_sparse > 0).sum(1).mean()

        return sparse.hstack([tfidf_matrix, entities_sparse]), f_score_dict,\
               params

    def get_dandelion_entities(self, data):
        entities = set()

        for d in data:
            for e in d['dandelion']['annotations']:
                entities.add(e['title'])

        entities_dict = {e: 0 for e in entities}

        for d in data:
            for e in d['dandelion']['annotations']:
                entities_dict[e['title']] += 1

        return entities_dict, entities

    def get_data_with_abstract_2(self, relevance_threshold=0.65):
        data = self.mongo.get_all(order_by='id_doc')

        data = [doc for doc in data]
        only_text = []
        ent_dict, ent_set = self.get_dandelion_entities(data)

        if filter:
            entities_set = set([k for k, v in ent_dict.iteritems()])
        else:
            entities_set = ent_set
        entities = {e: i for i, e in enumerate(entities_set)}
        dandelion_entities = np.zeros((len(data), len(entities_set)))

        for doc in data[:]:
            text = doc['text']
            if 'dandelion' in doc:
                abstract_matrix = []
                abstract_matrix.append(text)
                for e in doc['dandelion']['annotations']:
                    text += ' '
                    abstract = self.dbpedia.get_element_by_mongo_id(e['lod']['dbpedia'])
                    if abstract:
                        abstract_matrix.append(abstract['abstract']['value'])

                    rel = np.float64(e['confidence'])
                    name = e['title']
                    if rel > relevance_threshold:

                        if abstract:
                            text += abstract['abstract']['value']
                        dandelion_entities[doc['id_doc']][entities[name]] = rel
                tfidf_vectorizer = TfidfVectorizer(max_df=0.8,
                                       max_features=200000,
                                       min_df=2,
                                       stop_words='english',
                                       strip_accents='unicode',
                                       use_idf=True,
                                       ngram_range=(1, 1),
                                       norm='l2',
                                       tokenizer=TextUtils.tokenize_and_stem)

                try:
                    mat = tfidf_vectorizer.fit_transform(abstract_matrix)

                    # calcolo la sim tra il testo originale e gli abstract
                    sim_matrix = cosine_similarity(mat[0:1], mat)[0]
                    text = abstract_matrix[0]
                    for i, sim in enumerate(sim_matrix):
                        if 0.5 < sim < 1.0 and i != 0:
                            #print 'doc %s sim %s' %(doc['id_doc'], sim)
                            text += ' '
                            text += abstract_matrix[i]
                except ValueError:
                    text = abstract_matrix[0]

            only_text.append(text)

        return only_text, dandelion_entities, data

    def get_data_only_with_abstract(self, relevance_threshold=0.75, min_df=0.01,
                              gamma=0.89, filter=False):
        only_text, ent, data = self.get_data_with_abstract_2(relevance_threshold)
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
                                           max_features=200000,
                                           min_df=min_df,
                                           stop_words='english',
                                           strip_accents='unicode',
                                           use_idf=True,
                                           ngram_range=(1, 1),
                                           norm='l2',
                                           tokenizer=TextUtils.tokenize_and_stem)

        tfidf_matrix = tfidf_vectorizer.fit_transform(only_text)
        f_score_dict = self.labels_dict(data)
        params = tfidf_vectorizer.get_params()
        params['original_terms'] = tfidf_matrix.shape[0]
        params['gamma'] = gamma
        params['relevance_threshold'] = relevance_threshold
        params['classes'] = len(f_score_dict)
        params['tokenizer'] = 'TextUtils.tokenize_and_stem'

        return tfidf_matrix, f_score_dict, params

    def get_data_with_dandelion(self, relevance_threshold=0.75, min_df=2,
                              gamma=0.89, filter=False):
        only_text, ent, data = self.get_data_with_abstract_2(relevance_threshold)
        entities_sparse = sparse.csr_matrix(ent)

        tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
                                           max_features=200000,
                                           min_df=min_df,
                                           stop_words='english',
                                           strip_accents='unicode',
                                           use_idf=True,
                                           ngram_range=(1, 1),
                                           norm='l2',
                                           tokenizer=TextUtils.tokenize_and_stem)

        tfidf_matrix = tfidf_vectorizer.fit_transform(only_text)

        print 'tfifd matrix dimension: %s x %s' %(tfidf_matrix.shape[0],
                                                  tfidf_matrix.shape[1])
        print 'entities matrix dimension: %s x %s ' %(entities_sparse.shape[0],
                                                     entities_sparse.shape[1])
        print 'non zero elements in entities matrix: %s' \
              % len(entities_sparse.data)

        '''print tfidf_matrix[tfidf_matrix > 0].mean()
        print tfidf_matrix[tfidf_matrix > 0].max()

        print entities_sparse[entities_sparse > 0].mean()
        print entities_sparse[entities_sparse > 0].max()
        print '#' * 80'''
        #print 'after balancing'

        tfidf_matrix = tfidf_matrix * 1
        entities_sparse = entities_sparse * (1 - gamma)

        #print tfidf_matrix[tfidf_matrix > 0].mean()
        #print tfidf_matrix[tfidf_matrix > 0].max()

        #print entities_sparse[entities_sparse > 0].mean()
        #print entities_sparse[entities_sparse > 0].max()

        f_score_dict = self.labels_dict(data)
        params = tfidf_vectorizer.get_params()
        params['dandelion_entities'] = entities_sparse.shape[1]
        params['original_terms'] = tfidf_matrix.shape[0]
        params['gamma'] = gamma
        params['relevance_threshold'] = relevance_threshold
        params['classes'] = len(f_score_dict)
        params['tokenizer'] = 'TextUtils.tokenize_and_stem'
        del params['dtype']

        params['avg_nnz_row'] = (entities_sparse > 0).sum(1).mean()

        return sparse.hstack([tfidf_matrix, entities_sparse]), f_score_dict, params
        #return tfidf_matrix, f_score_dict, params

    def get_data_fabio(self, gamma=0.89, rank_metric='r'):
        data = self.mongo.get_all(order_by='id_doc')

        data = [doc for doc in data]
        only_text = [doc['text'] for doc in data]

        entitySet = set()
        for d in data:
            if 'isa' in d:
                for e in d['isa']:
                    entitySet.add(e['entity'])

        current = np.zeros((len(data), len(entitySet)), dtype=np.float)
        count = 0
        invIndex = {}
        countFeatures = 0
        for i,d in enumerate(data):
            if 'isa' in d:
                for f in d['isa']:
                    if f['entity'] not in invIndex:
                       invIndex[f['entity']] = countFeatures
                       countFeatures += 1
                    current[count, invIndex[f['entity']]] = f[rank_metric]
            count += 1
        current = np.nan_to_num(current)
        current_sparse = sparse.csr_matrix(current)

        tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
                                           max_features=200000,
                                           min_df=2,
                                           stop_words='english',
                                           strip_accents='unicode',
                                           use_idf=True,
                                           ngram_range=(1, 1),
                                           norm='l2',
                                           tokenizer=TextUtils.tokenize_and_stem)

        tfidf_matrix = tfidf_vectorizer.fit_transform(only_text)

        tfidf_matrix = tfidf_vectorizer.fit_transform(only_text)

        print 'tfifd matrix dimension: %s x %s' %(tfidf_matrix.shape[0],
                                                  tfidf_matrix.shape[1])
        print 'entities matrix dimension: %s x %s ' %(current_sparse.shape[0],
                                                     current_sparse.shape[1])
        print 'non zero elements in entities matrix: %s' \
              % len(current_sparse.data)

        tfidf_matrix = tfidf_matrix * 1
        entities_sparse = current_sparse * (1 - gamma)

        f_score_dict = self.labels_dict(data)
        params = tfidf_vectorizer.get_params()
        params['dandelion_entities'] = entities_sparse.shape[1]
        params['original_terms'] = tfidf_matrix.shape[0]
        params['gamma'] = gamma
        params['rank_metric'] = rank_metric
        params['classes'] = len(f_score_dict)
        params['tokenizer'] = 'TextUtils.tokenize_and_stem'
        del params['dtype']

        params['avg_nnz_row'] = (entities_sparse > 0).sum(1).mean()

        return sparse.hstack([tfidf_matrix, entities_sparse]), f_score_dict,\
               params

    def get_data_only_with_entities(self, relevance_threshold=0.75, gamma=0.89, filter=False):
        data = self.mongo.get_all(order_by='id_doc')

        data = [doc for doc in data]
        only_text = [doc['text'] for doc in data]

        ent_dict, ent_set = self.get_dandelion_entities(data)

        if filter:
            entities_set = set([k for k, v in ent_dict.iteritems()])
        else:
            entities_set = ent_set
        entities = {e: i for i, e in enumerate(entities_set)}
        dandelion_entities = np.zeros((len(data), len(entities_set)))

        for doc in data[:]:
            text = doc['text']
            if 'dandelion' in doc:
                for e in doc['dandelion']['annotations']:
                    rel = np.float64(e['confidence'])
                    name = e['title']
                    if rel > relevance_threshold:
                        dandelion_entities[doc['id_doc']][entities[name]] = rel

        entities_sparse = sparse.csr_matrix(dandelion_entities)

        tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
                                           max_features=200000,
                                           min_df=2,
                                           stop_words='english',
                                           strip_accents='unicode',
                                           use_idf=True,
                                           ngram_range=(1, 1),
                                           norm='l2',
                                           tokenizer=TextUtils.tokenize_and_stem)

        tfidf_matrix = tfidf_vectorizer.fit_transform(only_text)

        print 'tfifd matrix dimension: %s x %s' %(tfidf_matrix.shape[0],
                                                  tfidf_matrix.shape[1])
        print 'entities matrix dimension: %s x %s ' %(entities_sparse.shape[0],
                                                     entities_sparse.shape[1])
        print 'non zero elements in entities matrix: %s' \
              % len(entities_sparse.data)

        '''print tfidf_matrix[tfidf_matrix > 0].mean()
        print tfidf_matrix[tfidf_matrix > 0].max()

        print entities_sparse[entities_sparse > 0].mean()
        print entities_sparse[entities_sparse > 0].max()
        print '#' * 80'''
        #print 'after balancing'

        tfidf_matrix = tfidf_matrix * 1
        entities_sparse = entities_sparse * (1 - gamma)

        #print tfidf_matrix[tfidf_matrix > 0].mean()
        #print tfidf_matrix[tfidf_matrix > 0].max()

        #print entities_sparse[entities_sparse > 0].mean()
        #print entities_sparse[entities_sparse > 0].max()

        f_score_dict = self.labels_dict(data)
        params = tfidf_vectorizer.get_params()
        params['dandelion_entities'] = entities_sparse.shape[1]
        params['original_terms'] = tfidf_matrix.shape[0]
        params['gamma'] = gamma
        params['relevance_threshold'] = relevance_threshold
        params['classes'] = len(f_score_dict)
        params['tokenizer'] = 'TextUtils.tokenize_and_stem'
        del params['dtype']

        params['avg_nnz_row'] = (entities_sparse > 0).sum(1).mean()

        return sparse.hstack([tfidf_matrix, entities_sparse]), f_score_dict,\
               params

    def dimensionality_reduction(self):
        # TODO -  uning PCA or other method
        pass

    def labels_dict(self, data):
        dict_out = {}
        only_labels = [doc['label'] for doc in data]
        for l in sorted(set(only_labels)):
            dict_out[l] = {
                'docs': [],
                'fscore': ''
            }
        for doc in data:
            dict_out[doc['label']]['docs'].append(doc['id_doc'])

        return dict_out

    @property
    def get_data_grouped(self):
        dict_out = {}
        labels = self.data['labels']
        docs = self.data['data']
        for l in sorted(set(labels)):
            dict_out[l] = {
                'docs': [],
                'fscore': ''
            }

        for i in range(len(docs)):
            # dict_out[labels[i]]['docs'].append(i)
            dict_out[labels[i]]['docs'].append({
                'id_doc': i,
                'text': self.data['data'][i],
                'label': labels[i]
            })
        # print dict_out
        return dict_out

    @property
    def tfidf_matrix(self):
        '''
        deprecated
        :return:
        '''
        print 'dataset %s: %s documents and %s classes' % (
            self.dataset_name, len(self.data['data']), len(self.data['labels']))
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=200000,
                                           min_df=2, stop_words='english',
                                           use_idf=True,
                                           tokenizer=tu.TextUtils.tokenize_and_stem,
                                           ngram_range=(1, 1))

        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['data'])

        print tfidf_matrix.shape
        return tfidf_matrix


def test_tokenize_abstract(dataset):
    dp = DocumentsProcessor(dataset)
    data = [d for d in dp.mongo.get_all(order_by='id_doc')]

    only_text = dp.get_data_with_abstract(data)

    for d in only_text[:1]:
        ret = tu.TextUtils.tokenize_and_stem(d)
        print len(ret)

        ret = tu.TextUtils.remove_stopwords(ret)
        print len(ret)

        # pp.pprint(ret)

def test_dandelion_abstract():
    docp = DocumentsProcessor('re0')
    docp.get_data_with_dandelion()

def main():
    parser = argparse.ArgumentParser(
        description='Script that performs action on db')

    parser.add_argument('-d',
                        dest='dataset',
                        help='Dataset name',
                        required=True,
                        choices=['re0', 're1'])

    parser.add_argument('--db',
                        dest='db',
                        help='DB name',
                        required=False,
                        choices=['hc'])

    parser.add_argument('--abstract', '-a',
                        dest='abstract',
                        help='specify action to perform',
                        required=False,
                        action='store_true')

    parser.add_argument('--test', '-t',
                        dest='test',
                        help='specify action to perform',
                        required=False,
                        action='store_true')

    args = parser.parse_args()

    dataset = args.dataset
    db = args.db

    if args.test:
        test_tokenize_abstract(dataset)
    else:
        rt = DocumentsProcessor(dataset)
        # rt.get_data_with_abstract()
        tfidf_matrix, dict_eval = rt.get_data(args.abstract)

        # tfidf_matrix, dict_eval = rt.get_data()
        # pp.pprint(tfidf_matrix)
        # pp.pprint(dict_eval)

        # pp.pprint(d[:5])


if __name__ == '__main__':
    rt = DocumentsProcessor('re0')
    rt.get_data_with_alchemy()
