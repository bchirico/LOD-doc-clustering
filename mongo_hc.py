__author__ = 'biagio'

import pymongo as pm
import pprint as pp
import argparse

class MongoHC:

    def __init__(self, db, collection):
        self.client = pm.MongoClient()
        self.db = getattr(self.client, db)
        self.collection = getattr(self.db, collection)

    def save_document(self, doc, collection=None):
        # TODO - replace save with insert_one

        '''

        :param doc:
        :param collection: Optional. Collection where document has to be saved.
        :return:
        '''
        c = getattr(self.db, collection) if collection else self.collection
        print 'Saving document'
        print '-' * 80
        c.save(doc)

    def update_document(self, doc):
        pass

    def get_element_by_mongo_id(self, id, collection=None):
        c = getattr(self.db, collection) if collection else self.collection
        return c.find_one({'_id': id})

    def get_element_by_id(self, id, collection=None):
        c = getattr(self.db, collection) if collection else self.collection
        return c.find_one({'id_doc': id})

    def get_all(self, collection=None, order_by=None):
        c = getattr(self.db, collection) if collection else self.collection
        if order_by:
            return c.find({}).sort([(order_by, pm.ASCENDING)])
        return c.find({})

    def get_by_key(self, key, value, collection=None, order_by=None):
        c = getattr(self.db, collection) if collection else self.collection
        query = {key: value}
        if order_by:
            return c.find(query).sort([(order_by, pm.ASCENDING)])
        return c.find(query)

    def get_empty_abstract(self, collection=None, order_by=None):
        c = getattr(self.db, collection) if collection else self.collection
        query = {'abstracts': {'$exists': True, '$size': 0}}
        if order_by:
            return c.find(query).sort([(order_by, pm.ASCENDING)])
        else:
            return c.find(query)

    def get_doc_with_no_key(self, key, collection=None, order_by=None):
        c = getattr(self.db, collection) if collection else self.collection
        query = {key: {'$exists': False}}
        if order_by:
            return c.find(query).sort([(order_by, pm.ASCENDING)])
        else:
            return c.find(query)

    def is_empty(self, collection=None):
        c = getattr(self.db, collection) if collection else self.collection
        return c.count() == 0

    def remove_document_by_id(self, id, collection=None):
        c = getattr(self.db, collection) if collection else self.collection
        return c.delete_one({'id_doc': id})

    def custom_query(self, query, collection=None):
        c = getattr(self.db, collection) if collection else self.collection
        return c.find(query)

    def safe_mode(self, collection=None):
        c = getattr(self.db, collection) if collection else self.collection
        if not self.is_empty(collection):
            raise Exception('Operation denied: Database is not empty!')
        else:
            return True

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

def init_db(dataset, db):
    from document_processor import DocumentsProcessor
    dp = DocumentsProcessor(dataset)
    mongo_hc = MongoHC(db, dataset)

    if mongo_hc.safe_mode():
        data = dp.get_data_grouped

        for k, v in data.iteritems():
            for doc in v['docs']:
                print 'Saving document %s' %doc['id_doc']
                mongo_hc.save_document(doc)

def duplicate_db(dataset, db):
    mongo_from = MongoHC(db, dataset)
    mongo_to = MongoHC(db, dataset + '_for_alchemy')

    if mongo_to.safe_mode():
        data = mongo_from.get_all(order_by='id_doc')
        for d in data:
            try:
                mongo_to.save_document(d)
            except Exception, e:
                print e

def clean_text(dataset, db):
    mongo = MongoHC(db, dataset)
    docs = [doc for doc in mongo.get_all(order_by='id_doc')]

    for doc in docs:
        doc['text'] = doc['text'].replace('\n', ' ')
        mongo.save_document(doc)


def main():
    parser = argparse.ArgumentParser(
        description='Script that performs action on db')

    parser.add_argument('-d',
                        dest='dataset',
                        help='Dataset name',
                        required=True,
                        choices=['re0', 're1', 're0_for_alchemy',
                                 're1_for_alchemy'])

    parser.add_argument('--db',
                        dest='db',
                        help='DB name',
                        required=True,
                        choices=['hc'])

    parser.add_argument('--action', '-a',
                        dest='action',
                        help='specify action to perform',
                        required=True,
                        choices=['create', 'duplicate', 'clean_text'])

    args = parser.parse_args()

    dataset = args.dataset
    db = args.db
    action = args.action

    if action == 'create':
        init_db(dataset, db)
    elif action == 'duplicate':
        duplicate_db(dataset, db)
    elif action == 'clean_text':
        clean_text(dataset, db)

def test_get_empty_abstract():
    mongo = MongoHC('hc', 're0')
    mongo.get_empty_abstract()

if __name__ == '__main__':
    main()
