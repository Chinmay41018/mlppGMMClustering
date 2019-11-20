import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression


color_map = {0: 'red',
             1: 'green',
             2: 'blue',
             3: 'yellow',
             4: 'pink'}
en = {'business': 0,
      'entertainment': 1,
      'politics': 2,
      'sport': 3,
      'tech': 4}


def tsne_plot(emb, labels, color_map=color_map, n_comp=2):
    low_emb = TSNE(n_components=2).fit_transform(emb)
    plt.scatter(
        low_emb[:, 0], low_emb[:, 1], c=[color_map[y] for y in labels])
    plt.show()


def criticise(path_to_d):
    dat = torch.load(path_to_d, map_location=torch.device('cpu'))
    labels = pd.read_csv(
        './data/cleaned_documents.csv',
        header=0,
        names=['text', 'embedding', 'label']).label
    encoded_label = [en[y] for y in labels]
    model = LogisticRegression()
    cv_score = cross_validate(model, np.array([x.data.numpy() for x in dat]),
                              encoded_label, cv=3)
    print("Classification performance : ", np.mean(cv_score['test_score']))
    return np.array([x.data.numpy() for x in dat]), encoded_label


def criticise_gold():
    def lam_doc2vec(x): return [float(y) for y in (x.strip("[]").split(", "))]
    doc2vec_dat = pd.read_csv(
                        './data/embedded_doc2vec_bbc.csv',
                        header=0,
                        names=['text', 'label', 'embedding'],
                        converters={"embedding": lam_doc2vec})
    doc2vec_emb = doc2vec_dat.embedding.values
    labels = doc2vec_dat.label.values
    encoded_label = [en[y] for y in labels]
    model = LogisticRegression()
    cv_score = cross_validate(model, doc2vec_emb.tolist(), encoded_label, cv=3)
    print(
        "Doc2Vec classification performance : ",
        np.mean(cv_score['test_score']))
    return doc2vec_emb.tolist(), encoded_label
