import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

import spacy

nlp = spacy.load("en_core_web_lg")


def preprocess_sentence(text):
    doc = nlp(text)
    output_text = []
    for token in doc:
        if not any([token.is_stop, token.is_stop, token.like_url, token.is_punct]):
            output_text.append(token.lemma_)
    return ' '.join(output_text)