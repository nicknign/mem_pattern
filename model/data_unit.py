# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import os
import re
import numpy as np
import pandas as pd
import jieba as jb
import json

exclude_re = re.compile(u"[,，【】<>{};?？'\"]")

filepath = os.path.split(os.path.realpath(__file__))[0]

jb.load_userdict("{}/fenci.txt".format(filepath))


def save_file(data, fpath):
    with open(fpath, "w") as pf:
        json.dump(data, pf)


def load_task():
    trainfile = "{}/data/pattern.csv".format(filepath)
    traindf = pd.read_csv(trainfile, encoding="utf_8")
    train = []
    test = []
    for index, row in traindf.iterrows():
        train.append([row["question"], row["category"]])
    return train, test


def cut2list(string):
    result = []
    string = exclude_re.sub("", string)
    cutgen = jb.cut(string)
    for i in cutgen:
        if i != " ":
            result.append(i)
    return result


def vectorize_data(data, word_idx, sentence_size):
    Q = []
    C = []
    Ans = []

    for question, category in data:
        quesl = cut2list(question)
        lq = max(0, sentence_size - len(quesl))
        q = [word_idx[w] for w in quesl] + [0] * lq
        if category not in Ans:
            Ans.append(category)
        Q.append(q)

    answer_size = len(Ans)

    for _, category in data:
        y = np.zeros(answer_size)
        index = Ans.index(category)
        y[index] = 1
        C.append(y)

    save_file(word_idx, "./data/vocab.json")
    save_file(Ans, "./data/ans.json")

    return np.array(Q), np.array(C), answer_size
