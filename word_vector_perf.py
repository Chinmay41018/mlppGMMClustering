import torch
import gensim.downloader as api
import itertools
import numpy as np
import pandas as pd
import ast


def similarity(word1, word2, expt_no=1):
    if expt_no not in [1, 2]:
        model = api.load('glove-wiki-gigaword-100')
        print("Similarity using doc2vec is : ", model.similarity(word1, word2))
    else:
        dat = pd.read_csv('./data/cleaned_documents.csv',
                          converters={"text": ast.literal_eval})
        words = list(itertools.chain.from_iterable(dat.text))
        words = set(words)
        word2idx = {word: idx for idx, word in enumerate(words)}
        U = torch.load(
            './expts/expt'+str(expt_no)+'/U',
            map_location=torch.device('cpu')).detach().numpy()
        print(
            "Similarity using experiment" + str(expt_no) + "is : ", np.dot(
                U[word2idx[word1]], U[word2idx[word2]])/(np.linalg.norm(
                    U[word2idx[word1]]) * np.linalg.norm(
                    U[word2idx[word2]])))
