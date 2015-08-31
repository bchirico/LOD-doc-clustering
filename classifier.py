from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from document_processor import DocumentsProcessor
import pprint as pp
import numpy as np

from nltk.cluster import cosine_distance, euclidean_distance, KMeansClusterer

__author__ = 'biagio'

class PartitionalClustering:

    def __init__(self, tfidf_matrix):
        self.clf = KMeans(n_clusters=2,
                          max_iter=1000,
                          n_init=30,
                          verbose=0,
                          precompute_distances=True)

        self.tfidf_matrix = tfidf_matrix

    def cluster(self, tfidf_matrix=None):
        tfidf_matrix = tfidf_matrix if tfidf_matrix is not None else self.tfidf_matrix
        self.clf.fit(tfidf_matrix)

        clusters = self.clf.labels_.tolist()
        #pp.pprint(clusters[:10])
        '''print len(clusters)
        print clusters.count(0)
        print clusters.count(1)
        '''
        return clusters

    def f_score(self, cluster, f_score_dict):

        for cl in f_score_dict:
            docs = f_score_dict[cl]['docs']
            recall = float(len([d for d in cluster if d in docs])) / float(len(docs))
            precision = float(len([d for d in cluster if d in docs])) / float(len(cluster))

            try:
                f_score = (2 * precision * recall) / (precision + recall)
                f_score_dict[cl]['fscore'] = f_score if (f_score > f_score_dict[cl]['fscore'] or not f_score_dict[cl]['fscore']) else f_score_dict[cl]['fscore']
            except ZeroDivisionError, e:
                #print e
                pass

        return f_score_dict

    def evaluate(self, solution, f_score_dict, matrix):
        first_cluster = [i for i, d in enumerate(solution) if d == 0]
        second_cluster =[i for i, d in enumerate(solution) if d == 1]

        f_score_dict = self.f_score(first_cluster, f_score_dict)
        f_score_dict = self.f_score(second_cluster, f_score_dict)

        first_tfidf_matrix = matrix[first_cluster,:]
        second_tfidf_matrix = matrix[second_cluster,:]

        return first_tfidf_matrix, second_tfidf_matrix, f_score_dict

    def recursive(self, matrix, f_score_dict):
        s = self.cluster(matrix)
        a, b, f = self.evaluate(s, f_score_dict, matrix)
        '''print 'f_score list:'
        pp.pprint([f[d]['fscore'] for d in f])
        print '\n'
        '''
        if a.shape[0] > 10:
            #print 'solution a'
            self.recursive(a, f)
        if b.shape[0] > 10:
            #print 'solution b'
            self.recursive(b, f)
        return f

    def average_f_score(self, f_score_dict):
        f_score = [f_score_dict[d]['fscore'] for d in f_score_dict]
        average = 0
        n = float(1262)
        for d in f_score_dict:
            nr = float(len(f_score_dict[d]['docs']))
            f = nr / n * f_score_dict[d]['fscore']
            #print n, nr, f, f_score_dict[d]['fscore']
            average += f
        return average

class Agglomerative():

    def cluster(self):
        pass

class PartionalNltk():

    def __init__(self):
        self.clf = KMeansClusterer(2, euclidean_distance, repeats=5, avoid_empty_clusters=True)

    def cluster(self, data):
        clusters = self.clf.cluster(data.toarray(), True)
        return np.array(clusters)

    def f_score(self, cluster, f_score_dict):

        for cl in f_score_dict:
            docs = np.array(f_score_dict[cl]['docs'])
            nri = np.intersect1d(cluster, docs).shape[0]
            nr = docs.shape[0]
            ni = cluster.shape[0]
            #print nri, nr, ni

            try:
                recall = float(nri) / float(nr)
                precision = float(nri) / float(ni)
                f_score = (2 * precision * recall) / (precision + recall)
                f_score_dict[cl]['fscore'] = f_score if (f_score > f_score_dict[cl]['fscore'] or not f_score_dict[cl]['fscore']) else f_score_dict[cl]['fscore']
            except ZeroDivisionError, e:
                #print e
                pass

        return f_score_dict

    def evaluate(self, solution, f_score_dict, matrix):
        first_cluster = np.array([i for i, d in enumerate(solution) if d == 0])
        second_cluster = np.array([i for i, d in enumerate(solution) if d == 1])

        print 'dimension of first cluster: %s' %first_cluster.shape[0]
        print 'dimension of second cluster: %s' %second_cluster.shape[0]

        f_score_dict = self.f_score(first_cluster, f_score_dict)
        f_score_dict = self.f_score(second_cluster, f_score_dict)

        first_tfidf_matrix = matrix[first_cluster,:] if first_cluster.shape[0] > 0 else np.array([])
        second_tfidf_matrix = matrix[second_cluster,:] if second_cluster.shape[0] > 0 else np.array([])

        return first_tfidf_matrix, second_tfidf_matrix, f_score_dict

    def recursive(self, matrix, f_score_dict):
        #print matrix.shape
        s = self.cluster(matrix)
        a, b, f = self.evaluate(s, f_score_dict, matrix)
        '''print 'f_score list:'
        pp.pprint([f[d]['fscore'] for d in f])
        print '\n'
        '''

        #print 'dimension(a): %s' % a.shape[0]
        #print 'dimension(b): %s' % b.shape[0]
        if a.shape[0] > 10:
            #print 'solution a'
            self.recursive(a, f)
        if b.shape[0] > 10:
            #print 'solution b'
            self.recursive(b, f)
        return f

    def average_f_score(self, f_score_dict):
        f_score = [f_score_dict[d]['fscore'] for d in f_score_dict]
        average = 0
        n = float(1262)
        for d in f_score_dict:
            nr = float(len(f_score_dict[d]['docs']))
            f = nr / n * f_score_dict[d]['fscore']
            #print n, nr, f, f_score_dict[d]['fscore']
            average += f
        return average


def sklearn_algo():
    dp = DocumentsProcessor('re0')
    data = dp.tfidf_matrix
    f_score_dict = dp.labels_dict
    pc = PartitionalClustering(data)

    with open('results.csv', 'a') as f:
        for i in range(1):
            f = pc.recursive(data, f_score_dict)

            average_f_score = pc.average_f_score(f)
            print 'average f_score for clustering solution number %s: %s' % (i, average_f_score)
    f.close()

def nltk_algo(dataset):
    dp = DocumentsProcessor(dataset)
    data = dp.tfidf_matrix
    f_score_dict = dp.labels_dict

    pc_nltk = PartionalNltk()

    with open('results.csv', 'a') as f:
        for i in range(30):
            f = pc_nltk.recursive(data, f_score_dict)

            average_f_score = pc_nltk.average_f_score(f)
            row = '%s, %s, nltk, %s, euclidean\n' % (i, dataset, average_f_score)
            f.write(row)

            print 'average f_score for clustering solution number %s: %s' % (i, average_f_score)
    f.close()

def print_f_score_dict(f):
    pp.pprint([f[d]['fscore'] for d in f])

if __name__ == '__main__':
    #sklearn_algo()

    nltk_algo('re0')
