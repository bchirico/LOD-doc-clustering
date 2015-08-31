import os
from bs4 import BeautifulSoup
import config
import json
import pprint as pp

__author__ = 'biagio'

class DocumentsExtractor:

    _allowed_datasets = ['reuters']

    def __init__(self, dataset_name):
        if dataset_name in self._allowed_datasets:
            self.dataset = dataset_name
        else:
            raise ValueError('Unknown dataset')

    def extract_documents(self):
        if self.dataset == 'reuters':
            rt_extractor = ReutersDocumentExtractor()
            rt_extractor.collect_documents()


class ReutersDocumentExtractor:

    reuters_dataset_dir = config.DATASETS_DIR + 'reuters/'
    classes = {
        're0': ['bop',
                    'cpi',
                    'gnp',
                    'housing',
                    'interest',
                    'ipi',
                    'jobs',
                    'lei',
                    'money',
                    'reserves',
                    'retail',
                    'trade',
                    'wpi'],
        're1': ['alum',
                    'carcass',
                    'cocoa',
                    'coffee',
                    'copper',
                    'cotton',
                    'crude',
                    'dlr',
                    'gas',
                    'gold',
                    'grain',
                    'iron',
                    'livestock',
                    'meal',
                    'nat',
                    'oilseed',
                    'orange',
                    'pet',
                    'rubber',
                    'ship',
                    'sugar',
                    'tin',
                    'veg',
                    'wheat',
                    'zinc']}

    def __init__(self):
        self.name = 'reuters'

    def get_files(self):
        '''
        Method that returns a list of filename alphabetically ordered

        :return: [] -> list of reuters files
        '''
        dir = self.reuters_dataset_dir
        sgm_files = [f for f in os.listdir(dir)
                     if os.path.isfile(os.path.join(dir, f))
                     and f.endswith('.sgm')]

        #pp.pprint(sorted(sgm_files))
        return sorted(sgm_files)

    def collect_documents(self, write=True):
        '''
        Method that transform the raw files. It selects only documents with single label
        and it filters documents according to the classes attribute (re0, re1).

        :param write: if True it stores results in file re0.sgm and re1.sgm
        :return:
        '''
        doc_files = self.get_files()
        for name in self.classes:
            out_content = {
                'data': [],
                'labels': []
            }
            out_file = open(config.PRE_PROCESSED_DATASETS + name +'.json', 'w')
            classes = self.classes.get(name)
            docs_found = 0
            for file in doc_files:
                print 'opening %s file' % file

                f = open(self.reuters_dataset_dir + file, 'r')
                data = f.read()
                soup = BeautifulSoup(data, 'lxml')
                documents = soup.findAll('reuters')

                for index, document in enumerate(documents):
                    if len(document.topics) == 1:
                        topic = document.topics.text
                        topic = topic.split('-')[0] if '-' in topic else topic
                        if topic in classes and write:
                            try:
                                text = document.content.text
                                docs_found += 1
                                out_content['data'].append(text)
                                out_content['labels'].append(classes.index(topic))
                                #out_file.write('\n')
                            except Exception, e:
                                print e
                                pass
                f.close()
            out_file.write(json.dumps(out_content))
            out_file.close()

        print '%s documents collected' % docs_found

if __name__ == '__main__':
    doc_proc = DocumentsExtractor('reuters')
    doc_proc.extract_documents()


