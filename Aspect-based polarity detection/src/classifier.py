import pandas as pd
from preprocessing import *
from nltk.classify.util import accuracy
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.classify import MaxentClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm

class Classifier:
    """The Classifier"""


    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        train = pd.read_csv(trainfile, delimiter='\t', names=['polarity_label', 'aspect_category', 'term', 'char_term_offset', 'sentence'])
        train, feat_list = preprocessor(train)
        feat_set = nltk_compatible(train, feat_list)
        split = int(len(feat_set) * 0.75)
        feat_train = feat_set[:split]
        feat_test = feat_set[split:]
        #self.main_classifier = SklearnClassifier(RandomForestClassifier())
        #self.main_classifier = SklearnClassifier(MultinomialNB())
        #self.main_classifier = SklearnClassifier(BernoulliNB())
        #self.main_classifier = SklearnClassifier(LogisticRegression())
        self.main_classifier = SklearnClassifier(svm.LinearSVC())
        self.main_classifier.train(feat_train)




    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        devdata = pd.read_csv(datafile, delimiter='\t', names=['polarity_label', 'aspect_category', 'term', 'char_term_offset', 'sentence'])
        devdata, test_feat = preprocessor(devdata)
        test_set = nltk_compatible(devdata, test_feat)
        labels = []
        for (sentence, label) in test_set:
            predict = self.main_classifier.classify(sentence)
            labels.append(predict)
        return labels






