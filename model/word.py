# -*- coding: utf-8 -*-
# !/usr/bin/env python
import sys
import os
import time
import tensorflow as tf
from aip import AipNlp
import pandas as pd

reload(sys)
sys.setdefaultencoding('utf8')

APP_ID = '9585999'
API_KEY = 'EHjkhXXsbLXATesvGBLu7PgB'
SECRET_KEY = 'tiTrCDQTqMQBbsarcn52CeP0Dc7bgmzT'

class Word(object):
    def __init__(self):
        self.aip = AipNlp(APP_ID, API_KEY, SECRET_KEY)
        self.vecdf = pd.read_csv("./data/vec.csv", encoding="utf_8")

    def word_sim(self, word1, word2):
        result = self.aip.wordembedding(unicode(word1), unicode(word2))
        sim = 0
        tf.logging.warning("wordsim result:{}".format(unicode(result)))
        if result.get('sim'):
            sim = result['sim']['sim']
        return sim

    def add_vec(self, word, vec):
        columns = ["word", "vec"]
        df = pd.DataFrame([[word, vec]], columns=columns)
        self.vecdf = self.vecdf.append(df, ignore_index=True)
        os.remove("./data/vec.csv")
        self.vecdf.to_csv("./data/vec.csv")

    def word_vec(self, word):
        df = self.vecdf.loc[self.vecdf["word"] == word]
        if df.shape[0]:
            vec = df.iloc[0]["vec"]
            return vec

        result = {}
        for i in range(10):
            result = self.aip.wordembedding(unicode(word))
            tf.logging.warning("wordvec result:{}".format(unicode(result)))
            if result.get('error_code'):
                time.sleep(1)
            else:
                break
        vec = None
        if result.get('vec'):
            vec = result['vec']['vec']
            self.add_vec(word, vec)
        return vec