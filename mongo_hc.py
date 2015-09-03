__author__ = 'biagio'

import pymongo as pm
import pprint as pp
import argparse

class MongoHCException(Exception):
    pass

class MongoHC:

    def __init__(self, db, collection):
        self.client = pm.MongoClient()
        self.db = getattr(self.client, db)
        self.collection = getattr(self.db, collection)

    def save_document(self, doc, collection=None):
        c = getattr(self.db, collection) if collection else self.collection
        c.save(doc)

    def get_element_by_id(self, id, collection=None):
        c = getattr(self.db, collection) if collection else self.collection
        return c.find_one({'id_doc': id})

    def get_all(self, collection=None, order_by=None):
        c = getattr(self.db, collection) if collection else self.collection
        if order_by:
            return c.find({}).sort([(order_by, pm.ASCENDING)])
        return c.find({})

    def is_empty(self, collection=None):
        c = getattr(self.db, collection) if collection else self.collection
        return c.count() == 0


def mongo_test():
    mongo = MongoHC('hc', 'test')
    for i in range(10):
        test = {
            'id_doc': i,
            'text': 'Fuck you!'
        }

        mongo.save_document(test)

    cursor = mongo.get_all()

    pp.pprint(cursor)

    for doc in cursor:
        pp.pprint(doc)



def init_reuters_db():
    from document_processor import DocumentsProcessor
    parser = argparse.ArgumentParser(
        description='Script to initialize db')

    parser.add_argument('-d',
                        dest='dataset',
                        help='Dataset name',
                        required=True,
                        choices=['re0', 're1'])

    parser.add_argument('--db',
                        dest='db',
                        help='DB name',
                        required=True,
                        choices=['hc'])

    args = parser.parse_args()

    dataset = args.dataset
    db = args.db

    dp = DocumentsProcessor(dataset)
    mongo_hc = MongoHC(db, dataset)

    if not mongo_hc.is_empty():
        raise MongoHCException('Operation denied: Database is not empty!')

    data = dp.get_data_grouped

    for k, v in data.iteritems():
        for doc in v['docs']:
            print 'Saving document %s' %doc['id_doc']
            mongo_hc.save_document(doc)

if __name__ == '__main__':
    init_reuters_db()
