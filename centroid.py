import nltk.data
from nltk.tokenize.nist import NISTTokenizer
from idf import computeWordIdf, tfIdf

punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


class CentroidScorer():
    def __init__(self, threshold=1.0):
        self.__threshold = threshold
    
    def fit(self, documents, wordTokenizer=NISTTokenizer()):
        self.__wordIdf = computeWordIdf(documents, wordTokenizer)
        
    def __computeTfIdf(self, document, wordTokenizer):
        words = wordTokenizer.tokenize(document, lowercase=True)
        return tfIdf(self.__wordIdf, words)
    
    def __computeCentroidSentence(self, wordTfIdf):
        word_centroid=dict()
        for (word, tfidf_score) in wordTfIdf.items():
            word_centroid[word] = tfidf_score if tfidf_score > self.__threshold else 0
        return word_centroid
    
    def __scoreSentencesAgainstCentroid(self, centroidSentence, sentences, wordTokenizer):
        sentence_dicts = []
        for i in range(len(sentences)):
            sentence_dict = dict()
            sentence_dict['index'] = i
            sentence = sentences[i]
            sentence_dict['text'] = sentence
            words = wordTokenizer.tokenize(sentence, lowercase=True)
            score = 0
            for word in words:
                score += centroidSentence[word]
            sentence_dict['score'] = score
            sentence_dicts.append(sentence_dict)
        return sentence_dicts
    
    def score(self, document, sentenceTokenizer=punkt_tokenizer, wordTokenizer=NISTTokenizer()):
        assert self.__wordIdf is not None, "Cannot score the model before fitting"
        word_tfidf = self.__computeTfIdf(document, wordTokenizer)
        centroid_sentence = self.__computeCentroidSentence(word_tfidf)
        sentences = sentenceTokenizer.tokenize(document)
        score_dicts = self.__scoreSentencesAgainstCentroid(centroid_sentence, sentences, wordTokenizer)
        
        return score_dicts
        
