from nltk.tokenize.nist import NISTTokenizer
from math import log
from collections import Counter, defaultdict

def computeWordIdf(documents, wordTokenizer=NISTTokenizer()):
    tot_document_count = 0
    total_word_count = Counter()
    for document in documents:
        words = wordTokenizer.tokenize(document, lowercase=True)
        doc_word_count = Counter(set(words))
        total_word_count += doc_word_count
        tot_document_count += 1
    word_idf = defaultdict(int)
    for (word, count) in total_word_count.items():
        word_idf[word] = log(tot_document_count/count)
        
    return word_idf

def tfIdf(wordIdf, words):
    word_tfidf = defaultdict(float)
    for word in words:
        word_tfidf[word] += wordIdf[word]
    return word_tfidf