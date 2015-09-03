import config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import itertools

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

import pprint as pp
import json
import text_utils as tu
import mongo_hc
import warnings

__author__ = 'biagio'

class DocumentsProcessor:

    def __init__(self, dataset_name):
        files_name = ['re0, re1']
        self.dataset_name = dataset_name
        self.mongo = mongo_hc.MongoHC('hc', self.dataset_name)

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
        with open(config.PRE_PROCESSED_DATASETS + self.dataset_name +'.json', 'r') as f:
            data = json.load(f)
        f.close()
        return data

    def get_data(self):
        data = self.mongo.get_all(order_by='id_doc')
        data = [doc for doc in data]

        only_text = [doc['text'] for doc in data]
        only_labels = [doc['label'] for doc in data]
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=200000,
                         min_df=2, stop_words='english',
                         use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem, ngram_range=(1, 1))
        tfidf_matrix = tfidf_vectorizer.fit_transform(only_text)

        dict_out = {}
        for l in sorted(set(only_labels)):
            dict_out[l] = {
                'docs': [],
                'fscore': ''
            }
        for doc in data:
            dict_out[doc['label']]['docs'].append(doc['id_doc'])

        return tfidf_matrix, dict_out

    @property
    def labels_dict(self):
        dict_out = {}
        labels = self.data['labels']
        docs = self.data['data']
        for l in sorted(set(labels)):
            dict_out[l] = {
                'docs': [],
                'fscore': ''
            }

        for i in range(len(docs)):
            dict_out[labels[i]]['docs'].append(i)
            '''dict_out[labels[i]]['docs'].append({
                'id_doc': i,
                'text': self.data['data'][i]
            })'''
        #print dict_out
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
            #dict_out[labels[i]]['docs'].append(i)
            dict_out[labels[i]]['docs'].append({
                'id_doc': i,
                'text': self.data['data'][i],
                'label': labels[i]
            })
        #print dict_out
        return dict_out

    @property
    def tfidf_matrix(self):
        '''
        deprecated
        :return:
        '''
        print 'dataset %s: %s documents and %s classes' %(self.dataset_name, len(self.data['data']), len(self.data['labels']))
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5, max_features=200000,
                         min_df=2, stop_words='english',
                         use_idf=True, tokenizer=tu.TextUtils.tokenize_and_stem, ngram_range=(1, 1))

        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['data'])

        print tfidf_matrix.shape
        return tfidf_matrix
if __name__ == '__main__':
    rt = DocumentsProcessor('re0')
    rt.tfidf_matrix

    #tfidf_matrix, dict_eval = rt.get_data()
    pp.pprint(tfidf_matrix)
    pp.pprint(dict_eval)

    #pp.pprint(d[:5])