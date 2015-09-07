import argparse
import logging
import time
import random
import sys

import spotlight
import queryGenerator

from mongo_hc import MongoHC
from alchemyapi_python.alchemyapi import AlchemyAPI
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
      u'support': 426,
      u'surfaceForm': u'people',
      u'types': u'DBpedia:TopicalConcept'}
    """

    alchemyapi = AlchemyAPI()
    response = alchemyapi.entities('text', text, {'sentiment': 1})
    resFilt=filter(lambda x: 'disambiguated' in x, response['entities'])
    key=['dbpedia','geonames','yago','opencyc']
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

    args = parser.parse_args()
    dataset = args.dataset
    db = args.db

    extract_entity(db, dataset)

if __name__ == '__main__':
    main()

    #get_abstract()