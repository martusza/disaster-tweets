import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

import spacy


class TextCleanTransformer(TransformerMixin, BaseEstimator):
    """

    """

    def __init__(self,
                 nlp_model=None):
        self.nlp_model = nlp_model

    def preprocess_sentence(self, text):
        doc = self.nlp_model(text.replace('\n', ' '))
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


