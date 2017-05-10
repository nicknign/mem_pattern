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
import word

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


def vectorize_data(data):
    wordaip = word.Word()
    Q = []
    C = []
    Ans = []
    maxsize = 0
    for question, _ in data:
        size = len(cut2list(question))
        if size > maxsize:
            maxsize = size
    sentence_size = maxsize
    for question, category in data:
        quesl = cut2list(question)
        lq = max(0, sentence_size - len(quesl))
        q = []
        for w in quesl:
            vecjs = wordaip.word_vec(w)
            vec = json.loads(vecjs)
            while isinstance(vec, unicode):
                vec = json.loads(vec)
            q.append(vec)
        q.extend([[0]*len(q[0])]*lq)
        Q.append(q)

        if category not in Ans:
            Ans.append(category)


    answer_size = len(Ans)

    for _, category in data:
        y = np.zeros(answer_size)
        index = Ans.index(category)
        y[index] = 1
        C.append(y)

    save_file(Ans, "./data/ans.json")

    return np.array(Q), np.array(C), answer_size, sentence_size
