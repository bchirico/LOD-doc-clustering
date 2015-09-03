from sklearn.cluster import KMeans, AgglomerativeClustering, linkage_tree
from sklearn.metrics.pairwise import cosine_similarity
import document_processor as dp
import pprint as pp
import numpy as np
import itertools
from scipy.cluster import hierarchy as hr


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
        self.clf = KMeansClusterer(2, cosine_distance, repeats=30, avoid_empty_clusters=True)

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

        #print 'dimension of first cluster: %s' %first_cluster.shape[0]
        #print 'dimension of second cluster: %s' %second_cluster.shape[0]

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

    with open('results.csv', 'a') as res:
        for i in range(20):
            f = pc.recursive(data, f_score_dict)

            average_f_score = pc.average_f_score(f)
            print 'average f_score for clustering solution number %s: %s' % (i, average_f_score)
    res.close()

def nltk_algo(dataset):
    dp = DocumentsProcessor(dataset)
    data = dp.tfidf_matrix
    f_score_dict = dp.labels_dict

    pc_nltk = PartionalNltk()

    for i in range(500):
        with open('results.csv', 'a') as res:

            try:
                f = pc_nltk.recursive(data, f_score_dict)

                average_f_score = pc_nltk.average_f_score(f)
                row = '%s, %s, nltk, %s, cosine\n' % (i, dataset, average_f_score)
                res.write(row)
                print 'average f_score for clustering solution number %s: %s' % (i, average_f_score)
            except RuntimeError, e:
                print e
        res.close()

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

def agglomerative_algo(dataset):
    dp = DocumentsProcessor(dataset)
    data = dp.tfidf_matrix
    f_score_dict = dp.labels_dict

    clf = AgglomerativeClustering(n_clusters=1262, linkage='average', affinity='l2',compute_full_tree=True)

    res = clf.fit(data.toarray())

    ii = itertools.count(data.shape[0])
    a = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in res.children_]

    children = linkage_tree(data.toarray(), n_clusters=1262)

    pp.pprint(children[:10])



if __name__ == '__main__':
    #sklearn_algo()

    #nltk_algo('re0')

    #agglomerative_algo('re0')

    scipy_algo('re0')

    #xx = [(2,3,1), (7,8,3), (4,10,7), (6,5,8)]

    #a = [{'node_id': x[2], 'left': x[0], 'right':x[1]} for x in xx]




