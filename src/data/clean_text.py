import pandas as pd
import re


class TextCleaner(object):
    def __init__(self,
                 text: str
                 ):
        self.text = text

    def remove_links(self):
        regex = re.compile(r'http\S+')
        clean_text = regex.sub('', self.text)
        return clean_text