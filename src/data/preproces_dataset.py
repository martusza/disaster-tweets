import numpy as np
import pandas as pd
import re
import spacy

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class TextCleanTransformer(TransformerMixin, BaseEstimator):
    def __init__(self,
                 nlp_model=None):
        self.nlp_model = nlp_model

    @staticmethod
    def preprocess_text(text):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
        text = re.sub(r'#', ' ', text)
        return text

    def preprocess_sentence(self, text):
        text = self.preprocess_text(text)

        doc = self.nlp_model(text)
        output_text = []
        for token in doc:
            if not any([token.is_stop, token.like_url, token.is_punct,
                        not token.is_ascii, token.like_email, token.is_space]):
                output_text.append(token.lemma_)
        return ' '.join(output_text)

    def fit(self, X, y=None):
        if self.nlp_model is None:
            self.nlp_model = spacy.load("en_core_web_lg")
        return self

    def transform(self, X):
        return np.array([self.preprocess_sentence(xi) for xi in X])


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()



