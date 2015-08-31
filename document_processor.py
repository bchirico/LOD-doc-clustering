import config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import itertools

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

import pprint as pp
import json
import text_utils as tu

__author__ = 'biagio'


class DocumentsProcessor:

    def __init__(self, dataset_name):
        files_name = ['re0, re1']
        self.dataset_name = dataset_name

    @property
    def data(self):
        data = None
        with open(config.PRE_PROCESSED_DATASETS + self.dataset_name +'.json', 'r') as f:
            data = json.load(f)
        f.close()
        return data

    def getDocumentsText(self):
        for name in self.dataset_name[:1]:
            with open(config.PRE_PROCESSED_DATASETS + name +'.json', 'r') as f:
                data = json.load(f)
                print 'dataset %s: %s documents and %s classes' %(name, len(data['data']), len(data['labels']))
                tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                 min_df=2, stop_words='english',
                                 use_idf=True, tokenizer=tu.TextUtils.tokenize_only, ngram_range=(1, 1))

                tfidf_matrix = tfidf_vectorizer.fit_transform(data['data'])

                print tfidf_matrix.shape

                dist = 1 - cosine_similarity(tfidf_matrix)

                num_clusters = len(set(data['labels']))

    @property
    def tfidf_matrix(self):
        print 'dataset %s: %s documents and %s classes' %(self.dataset_name, len(self.data['data']), len(self.data['labels']))
        tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                         min_df=2, stop_words='english',
                         use_idf=True, tokenizer=tu.TextUtils.tokenize_only, ngram_range=(1, 1))

        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['data'])

        print tfidf_matrix.shape
        return tfidf_matrix

    @property
    def labels_dict(self):
        dict_out = {}
        labels = self.data['labels']
        docs = self.data['data']
        for l in sorted(set(labels)):
            dict_out[l] = {
                'docs': [],
                'fscore': []
            }

        print dict_out
        for i in range(len(docs)):
            dict_out[labels[i]]['docs'].append(i)

        return dict_out

    def apply_clustering(self):
        km = KMeans(n_clusters=2, max_iter=1000, n_init=30, verbose=10)

        km.fit(self.tfidf_matrix)

        clusters = km.labels_.tolist()
        pp.pprint(clusters[:10])
        print len(clusters)
        print clusters.count(0)
        print clusters.count(1)


if __name__ == '__main__':
    rt = DocumentsProcessor('re0')

    rt.apply_clustering()

    #pp.pprint(d[:5])