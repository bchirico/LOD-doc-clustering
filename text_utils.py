import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

__author__ = 'biagio'

class TextUtils:

    stemmer = PorterStemmer()

    @staticmethod
    def tokenize_and_stem(text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [TextUtils.stemmer.stem(t) for t in filtered_tokens]
        return stems

    @staticmethod
    def tokenize_only(text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens

    @staticmethod
    def remove_stopwords(text):
        stopwords = nltk.corpus.stopwords.words('english')
        d = [w for w in text if w not in stopwords]
        return d

    @staticmethod
    def tokenize_and_stem_remove_stopwords(text):
        res = TextUtils.tokenize_and_stem(text)
        return TextUtils.remove_stopwords(res)