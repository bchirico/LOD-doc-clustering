import argparse
import logging
import time
import random
import sys
import traceback

import spotlight
import queryGenerator

from mongo_hc import MongoHC
from alchemyapi_python.alchemyapi import AlchemyAPI
from dandelion import DataTXT
from SPARQLWrapper import SPARQLWrapper, XML, JSON

import pprint as pp
__author__ = 'biagio'

x = logging.getLogger("logfun")
x.setLevel(logging.DEBUG)
h = logging.StreamHandler()
f = logging.Formatter("%(levelname)s %(asctime)s %(funcName)s %(lineno)d %(message)s")
h.setFormatter(f)
x.addHandler(h)
logfun = logging.getLogger("logfun")

#TODO: refactoring, ma soprattutto documentazione

def extract_entity(db, dataset):
    mongo_from = MongoHC(db, dataset + '_for_alchemy')
    mongo_to = MongoHC(db, dataset)

    docs = mongo_from.get_all(order_by='id_doc')
    docs = [doc for doc in docs]

    for doc in docs[:]:
        logfun.info('#' * 80)
        logfun.info('Scanning documents: %(id_doc)s' % doc)
        logfun.info('#' * 80)
        try:
            entitySet,annotationsSorted,response = getAnnotation(doc['text'])
            doc['abstracts'] = []
            doc['alchemy_response'] = response
            for e in entitySet:
                logfun.info('Extracting abstract for entity %s' % e)

                abstract = get_abstract(e)
                if abstract:
                    doc['abstracts'].append(abstract)
                else:
                    logfun.warning('Abstract not found!')
                logfun.info('-' * 80)

            doc['entity_set'] = list(entitySet)
            mongo_to.save_document(doc)
            mongo_from.remove_document_by_id(doc['id_doc'])
        except Exception, e:
            logfun.error("Something awful happened!")
            logfun.error(e)
            logfun.error(sys.exc_info()[2])

def extract_abstract(db, dataset):
    mongo = MongoHC(db, dataset)

    docs = [doc for doc in mongo.get_empty_abstract()]

    for doc in docs:
        try:
            for e in doc['entity_set']:
                logfun.info('Extracting abstract for entity %s' % e)
                abstract = get_abstract(e)
                if abstract:
                    doc['abstracts'].append(abstract)
                else:
                    logfun.warning('Abstract not found!')
                logfun.info('-' * 80)

            mongo.save_document(doc)
        except Exception, e:
            logfun.error("Something awful happened!")
            logfun.error(e)
            logfun.error(sys.exc_info()[2])

def extract_abstract_dandelion(db, dataset):
    mongo = MongoHC(db, dataset)
    mongo_dbpedia = MongoHC(db, 'dbpedia')
    docs = [doc for doc in mongo.get_all(order_by='id_doc')]

    for doc in docs:
        try:
            entities = [e['lod']['dbpedia'] for e in doc['dandelion']['annotations']]
            for e in entities:
                if mongo_dbpedia.get_element_by_mongo_id(e):
                    logfun.info('Entities already in database')
                    continue
                dbpedia = {}
                logfun.info('Extracting abstract for entity %s' % e)
                abstract = get_abstract(e)
                if abstract:
                  dbpedia['_id'] = e
                  dbpedia['abstract'] = abstract
                  mongo_dbpedia.save_document(dbpedia)
                else:
                    logfun.warning('Abstract not found!')
                logfun.info('-' * 80)
        except Exception, e:
            logfun.error("Something awful happened!")
            logfun.error(e)
            logfun.error(sys.exc_info()[2])

def extract_alchemy(db, dataset):
    mongo = MongoHC(db, dataset)

    docs = [doc for doc in mongo.get_doc_with_no_key('alchemy_response')]

    for doc in docs:
        try:
            entitySet,annotationsSorted,response = getAnnotation(doc['text'])
            doc['alchemy_response'] = response
            mongo.save_document(doc)
        except Exception, e:
            logfun.error("Something awful happened!")
            logfun.error(e)
            logfun.error(sys.exc_info()[2])

def extract_dandelion(db, dataset):
    mongo = MongoHC(db, dataset)

    docs = [doc for doc in mongo.get_doc_with_no_key('dandelion',
                                                     order_by='id_doc')]

    for doc in docs:
        try:
            dan = get_entities_from_dandelion(doc['text'])
            logfun.info(dan['timestamp'])
            doc['dandelion'] = dan
            mongo.save_document(doc)
        except Exception, e:
            logfun.error(traceback.format_exc())


def get_entities_from_dandelion(text):
    # TODO: mettere le keys in un file di setting
    datatxt = DataTXT(app_id='7c418708', app_key='0043c60be84a1f471184a192fe06e540')
    result = datatxt.nex(text, include_lod=True, language='en')
    return result


def get_abstract(entity):
    url="http://dbpedia-live.openlinksw.com/sparql"
    sparql = SPARQLWrapper(url)
    sparql.setTimeout(300)
    sparql.setReturnFormat(JSON)
    q = queryGenerator.QueryGenerator()
    try:
        query = q.getAbstractFromEntity(entity)
        #logfun.info(query)
        sparql.setQuery(query)
        results = sparql.queryAndConvert()['results']['bindings']
        return results[0]['abstract'] if results else []
    except:
        logfun.exception("Something awful happened!")



def getAnnotation(text):

    annotations = spotlight.annotate('http://spotlight.dbpedia.org/rest/annotate',text,confidence=0.25, support=40)
    annotationsSorted = sorted(annotations, key=lambda k: k['similarityScore'])
    setSpotlight=set(map(lambda x:x['URI'],annotationsSorted))

    """
    { u'URI': u'http://dbpedia.org/resource/People',
      u'offset': 321,
      u'percentageOfSecondRank': -1.0,
      u'similarityScore': 0.08647863566875458,
      u'support': 426, #
      u'surfaceForm': u'people',
      u'types': u'DBpedia:TopicalConcept'}
    """

    alchemyapi = AlchemyAPI()
    response = alchemyapi.entities('text', text, {'sentiment': 1})
    resFilt=filter(lambda x: 'disambiguated' in x, response['entities'])
    key=['dbpedia', 'geonames', 'yago', 'opencyc']
    resFilt

    entitySet=set()

    for r in resFilt:
        for k in key:
            if k in r['disambiguated']:
                entitySet.add(r['disambiguated'][k])

    """
    {u'count': u'1',
      u'disambiguated': {u'dbpedia': u'http://dbpedia.org/resource/Kathmandu',
       u'freebase': u'http://rdf.freebase.com/ns/m.04cx5',
       u'geo': u'27.716666666666665 85.36666666666666',
       u'geonames': u'http://sws.geonames.org/1283240/',
       u'name': u'Kathmandu',
       u'subType': [u'TouristAttraction'],
       u'website': u'http://www.kathmandu.gov.np/',
       u'yago': u'http://yago-knowledge.org/resource/Kathmandu'},
      u'relevance': u'0.33',
      u'sentiment': {u'type': u'neutral'},
      u'text': u'Kathmandu',
      u'type': u'City'},
    """

    entitySet.update(setSpotlight)

    return entitySet,annotationsSorted,response

def test(db, dataset):
    mongo = MongoHC(db, dataset)
    docs = mongo.get_element_by_id(1114)
    docs = [docs]

    for doc in docs[:1]:
        logfun.info('#' * 80)
        logfun.info('Scanning documents: %(id_doc)s' % doc)
        logfun.info('#' * 80)
        #try:
        entitySet,annotationsSorted,response = getAnnotation(doc['text'])
        '''doc['abstracts'] = []
        for e in entitySet:
            logfun.info('Extracting abstract for entity %s' % e)

            abstract = get_abstract(e)
            if abstract:
                doc['abstracts'].append(abstract)
            else:
                logfun.warning('Abstract not found!')
            logfun.info('-' * 80)

        doc['entity_set'] = list(entitySet)'''

        pp.pprint(response)

        #except Exception, e:
        #    raise e

def entities_distribution(db, dataset):
    mongo = MongoHC(db, dataset)

    data = [doc for doc in mongo.get_all(order_by='id_doc')]

    entities = set()

    for d in data:
        for e in d['alchemy_response']['entities']:
            entities.add(e['text'])

    entities_dict = {e: 0 for i, e in enumerate(entities)}

    for d in data:
        for e in d['alchemy_response']['entities']:
            entities_dict[e['text']] += 1

    return entities_dict, entities

def test_alchemy():
    text_1 = "The decision by the independent MP Andrew Wilkie to withdraw his support for the minority Labor government sounded dramatic but it should not further threaten its stability. When, after the 2010 election, Wilkie, Rob Oakeshott, Tony Windsor and the Greens agreed to support Labor, they gave just two guarantees: confidence and supply"
    text_2 = "inflation plan, initially hailed at home and abroad as the saviour of the economy, is limping towards its first anniversary amid soaring prices, widespread shortages and a foreign payments crisis.     Announced last February 28 the plan  prices, fixed the value of the new Cruzado currency and ended widespread indexation of the economy in a bid to halt the country's 250 pct inflation rate.     But within a year the plan has all but collapsed.     \"The situation now is worse than it was. Although there was inflation, at least the economy worked,\" a leading bank economist said.     The crumbling of the plan has been accompanied by a dramatic reversal in the foreign trade account. In 1984 and 1985 Brazil's annual trade surpluses had been sufficient to cover the 12 billion dlrs needed to service its 109 billion dlr foreign debt.     For the first nine months of 1986 all seemed to be on target for a repeat, with monthly surpluses averaging one billion dlrs. But as exports were diverted and imports increased to avoid further domestic shortages the trade surplus plunged to 211 mln dlrs in October and since then has averaged under 150 mln.  Reuter "
    alchemyapi = AlchemyAPI()
    response = alchemyapi.entities('text', text_2, {'sentiment': 1})

    pp.pprint(response)
    print len(response['entities'])

def test_dandelion():
    text_1 = "The decision by the independent MP Andrew Wilkie to withdraw his support for the minority Labor government sounded dramatic but it should not further threaten its stability. When, after the 2010 election, Wilkie, Rob Oakeshott, Tony Windsor and the Greens agreed to support Labor, they gave just two guarantees: confidence and supply"
    text_2 = "inflation plan, initially hailed at home and abroad as the saviour of the economy, is limping towards its first anniversary amid soaring prices, widespread shortages and a foreign payments crisis.     Announced last February 28 the plan  prices, fixed the value of the new Cruzado currency and ended widespread indexation of the economy in a bid to halt the country's 250 pct inflation rate.     But within a year the plan has all but collapsed.     \"The situation now is worse than it was. Although there was inflation, at least the economy worked,\" a leading bank economist said.     The crumbling of the plan has been accompanied by a dramatic reversal in the foreign trade account. In 1984 and 1985 Brazil's annual trade surpluses had been sufficient to cover the 12 billion dlrs needed to service its 109 billion dlr foreign debt.     For the first nine months of 1986 all seemed to be on target for a repeat, with monthly surpluses averaging one billion dlrs. But as exports were diverted and imports increased to avoid further domestic shortages the trade surplus plunged to 211 mln dlrs in October and since then has averaged under 150 mln.  Reuter "
    result = get_entities_from_dandelion(text_2)

    pp.pprint(len(result['annotations']))

def main():
    parser = argparse.ArgumentParser(
        description='Script that extracts entities\' abstract from dbpedia')

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

    parser.add_argument('--action',
                        dest='action',
                        help='action to perform',
                        required=True,
                        choices=['all', 'abstract', 'alchemy', 'test', 'dandelion'])

    args = parser.parse_args()
    dataset = args.dataset
    db = args.db
    action = args.action

    if action == 'all':
        extract_entity(db, dataset)
    elif action == 'abstract':
        extract_abstract_dandelion(db, dataset)
    elif action == 'test':
        test(db, dataset)
    elif action == 'alchemy':
        extract_alchemy(db, dataset)
    elif action == 'dandelion':
        extract_dandelion(db, dataset)

if __name__ == '__main__':
    #main()
    test_alchemy()
    test_dandelion()