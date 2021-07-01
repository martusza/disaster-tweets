import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import io
import json

class TextCleaner(object):
    def __init__(self,
                 data: pd.DataFrame,
                 col: str
                 ):
        self.data = data
        self.text_col = self.data[col]
        self.target = self.data['target']

    @staticmethod
    def remove_links_tweet(text):
        """
        Method to remove links from the text
        :param text: input text
        :return: text without links
        """
        regex = re.compile(r'http\S+')
        clean_text = regex.sub('', text)
        return clean_text

    def remove_links_text(self):
        """

        :return:
        """
        self.text_col = self.text_col.apply(self.remove_links_tweet)

    def tf_tokenize(self,
                    dataset: 'str' = 'train',
                    num_words: int = 8000,
                    oov_token: str = '<OOV>',
                    maxlen: int = 157,
                    ):
        if dataset == 'train':
            tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
            tokenizer.fit_on_texts(self.text_col)

            # Dumping tokenizer to use it on test data
            tokenizer_json = tokenizer.to_json()
            with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        else:
            with open('tokenizer.json') as f:
                tokenizer = tokenizer_from_json(json.load(f))

        tokenized_text = tokenizer.texts_to_sequences(self.text_col)
        tokenized_text_pad = pad_sequences(tokenized_text, maxlen=maxlen, padding='post', truncating='post')

        return tokenized_text_pad

