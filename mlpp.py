import ast
import sys
import pickle
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import pyro
from torch.autograd import Variable
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine


def get_pos_neg_sample(dat):
    x_plus = []
    x_neg = []
    uni_keys = list(uni_freq.keys())
    uni_values = list(uni_freq.values())
    for each in dat.text:
        x_plus_n = []
        x_neg_n = []
        sentence = [x for x in each if x in words]
        n_sample_draw = len(sentence) * neg_samples
        random_sample = np.random.choice(uni_keys, n_sample_draw, p=uni_values)
        r_i = 0
        for idx, token in enumerate(sentence):
            context = sentence[max(idx-c, 0):min(idx+c+1, len(sentence))]
            context.remove(token)
            for every in context:
                x_plus_n.append((word2idx[token], word2idx[every]))

            for sample in random_sample[r_i:r_i+len(context)]:
                x_neg_n.append((word2idx[token], word2idx[sample]))
            r_i += len(context)

        x_plus.append(x_plus_n)
        x_neg.append(x_neg_n)
    return x_plus, x_neg


def get_models():
    lamb = 1
    phi = 1
    if is_cuda:
        U_model = torch.distributions.MultivariateNormal(
            torch.zeros(W, E).cuda(), lamb**2 * torch.eye(E).cuda())
        V_model = torch.distributions.MultivariateNormal(
            torch.zeros(W, E).cuda(), lamb**2 * torch.eye(E).cuda())
        di_model = torch.distributions.MultivariateNormal(
            torch.zeros(E).cuda(), phi*torch.eye(E).cuda())
        return U_model, V_model, di_model
    U_model = torch.distributions.MultivariateNormal(
        torch.zeros(W, E), lamb**2 * torch.eye(E))
    V_model = torch.distributions.MultivariateNormal(
        torch.zeros(W, E), lamb**2 * torch.eye(E))
    di_model = torch.distributions.MultivariateNormal(
        torch.zeros(E), phi * torch.eye(E))
    return U_model, V_model, di_model


def get_loss(U, V, d_i, X_plus, X_neg):
    t1 = -torch.sum(
        -torch.log(
            1+torch.exp(
                -torch.bmm(
                    U[X_plus[:, 0]].view(U[X_plus[:, 0]].shape[0], 1, E),
                    (V[X_plus[:, 1]]+d_i).view(
                        (V[X_plus[:, 1]]+d_i).shape[0], E, 1)))))
    - torch.sum(
        -torch.log(
            1+torch.exp(
                -torch.bmm(
                    -U[X_neg[:, 0]].view(U[X_neg[:, 0]].shape[0], 1, E),
                    (V[X_neg[:, 1]]+d_i).view(
                        (V[X_neg[:, 1]]+d_i).shape[0], E, 1)))))
    t2 = - torch.sum(U_model.log_prob(U))
    t3 = - torch.sum(V_model.log_prob(V))
    t4 = - di_model.log_prob(d_i)
    return t1 + t2 + t3 + t4


def learn_UV(x_plus, x_neg):
    alpha = 0.99996
    n_iter = 100
    loss = []
    U = U_model.rsample()
    if is_cuda:
        U = U.cuda()
    V = V_model.rsample()
    if is_cuda:
        V = V.cuda()
    U.requires_grad = True
    V.requires_grad = True
    optimizer = torch.optim.Adam([V], lr=0.01)
    len_ = len(x_plus)
    for it in range(1):
        for n in range(len_):
            for g in optimizer.param_groups:
                g['lr'] *= alpha
            if is_cuda:
                d_i = di_model.rsample()
                d_i = d_i.to(device)
                d_i.requires_grad = True
                d_i.cuda = True
            else:
                d_i = Variable(di_model.rsample(), requires_grad=True)
            opti = torch.optim.Adam([d_i], lr=1)
            X_plus = torch.tensor(x_plus[n])
            X_neg = torch.tensor(x_neg[n])
            # SGD for d_i
            for iterat in range(n_iter):
                # Loss function
                opti.zero_grad()
                logProb = get_loss(U, V, d_i, X_plus, X_neg)
                logProb.backward()
                opti.step()
            optimizer.zero_grad()
            log_loss = get_loss(U, V, d_i, X_plus, X_neg)
            log_loss.backward()
            optimizer.step()
            loss.append(log_loss)
            if n % 100 == 0:
                print("Iteration : {}, logProb : {}".format(n+1, logProb))
    return U, V


def model(X, Y, U, V):
    phi = 1
    if is_cuda:
        d_i = pyro.sample("d_i", dist.MultivariateNormal(
            torch.zeros(E).cuda(), phi * torch.eye(E).cuda())).cuda()
    else:
        d_i = pyro.sample("d_i", dist.MultivariateNormal(
            torch.zeros(E), phi * torch.eye(E)))

    with pyro.plate('observations', len(X)):
        if is_cuda:
            logit = torch.sum(torch.bmm(
                U[X[:, 0]].view(U[X[:, 0]].shape[0], 1, E),
                (V[X[:, 1]]+d_i).view((V[X[:, 1]]+d_i).shape[0], E, 1)).cuda(),
                axis=1).cuda()
        else:
            logit = torch.sum(torch.bmm(
                U[X[:, 0]].view(U[X[:, 0]].shape[0], 1, E),
                (V[X[:, 1]]+d_i).view((V[X[:, 1]]+d_i).shape[0], E, 1)),
                axis=1)
        target = pyro.sample('obs', dist.Bernoulli(logits=logit), obs=Y)
        if is_cuda:
            target = target.cuda()


def train_d_i(x_plus, x_neg, U, V, idx=0):
    guide_di = AutoDelta(poutine.block(model, expose=['d_i']))
    optim = Adam({"lr": 0.1})
    svi_di = SVI(model, guide_di, optim, loss=Trace_ELBO())
    n_iter = 300
    losses = []
    pyro.clear_param_store()
    X_plus = torch.tensor(x_plus[idx])
    X_neg = torch.tensor(x_neg[idx])
    X_n = torch.cat((X_plus, X_neg))
    Y_n = torch.tensor(
        [float(1) for each in range(len(X_plus))] +
        [float(0) for each in range(len(X_neg))])
    if is_cuda:
        X_plus = X_plus.cuda()
        X_neg = X_neg.cuda()
        X_n = X_n.cuda()
        Y_n = Y_n.cuda()
    for i in range(n_iter):
        loss = svi_di.step(X_n, Y_n, U, V)
        losses.append(loss)
    d_i = guide_di(X_n, Y_n, U, V)["d_i"]
    return d_i, losses


def train(x_plus, x_neg, U, V):
    train_losses = []
    d_is = []
    for i in range(len(x_plus)):
        d_i, losses = train_d_i(x_plus, x_neg, U, V, i)
        train_losses.append(losses)
        d_is.append(d_i)
        if i % 100 == 0:
            print("ENDED FOR ITERATION {}".format(i))
    return d_is, train_losses

#############################################################
#                                                           #
#  Calling functions defined above here.                    #
#############################################################


# Hard coding context window size.
c = 2
neg_samples = 2*c


# Check if cuda enabled
is_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if sys.argv is not None and sys.argv[1]:
    dat_file = 'cleaned_documents_imdb.csv'
    postfix = '_imdb'
else:
    dat_file = 'cleaned_documents.csv'
    postfix = ''

# Reading  dataset
dat = pd.read_csv(dat_file, converters={"text": ast.literal_eval})
print("Read cleaned doc, size = ", len(dat))


#  Build vocabulary
E = 100  # size of embedding
words = list(itertools.chain.from_iterable(dat.text))
uni_freq = Counter(words)
words = set(words)
word2idx = {word: idx for idx, word in enumerate(words)}
idx2word = {idx: word for idx, word in enumerate(words)}
W = len(set(words))

print("Vocab size : ", W)
for each in uni_freq:
    uni_freq[each] = uni_freq[each]**0.75
deno = sum(uni_freq.values())
for each in uni_freq:
    uni_freq[each] = uni_freq[each]/deno
print("Unigram Frequencies obtained.")


# Build neg and pos samples.
try:
    x_plus = pickle.load('x_plus' + postfix)
    x_neg = pickle.load('x_neg' + postfix)
    print("loaded pre-saved")
except Exception:
    x_plus, x_neg = get_pos_neg_sample(dat)
    print("Computed")
    filehandler = open('x_plus' + postfix, 'wb')
    pickle.dump(x_plus, filehandler)

    filehandler = open('x_neg' + postfix, 'wb')
    pickle.dump(x_neg, filehandler)
    print("Saved for future")


# Get model definitions.
U_model, V_model, di_model = get_models()


# Learning U and V.
print("Lets learn U, V!")
try:
    U = torch.load('u' + postfix)
    V = torch.load('v' + postfix)
    print("Loaded U and V.")
except Exception:
    U, V = learn_UV(x_plus, x_neg)
    print("Computed U,V")
    torch.save(U, 'u')
    torch.save(V, 'v')
    print("Saved U,V")


# Learning doc vectors.
print("Learning d_i.")
dis, trainingLosses = train(x_plus, x_neg, U, V)
print("Finished learning d_i.")

torch.save(dis, 'di' + postfix)

# Training loss for learning of document vectors.
n_data = len(trainingLosses[0])
training_losses = np.mean(np.array(trainingLosses), axis=0)

fig = plt.figure()
plt.plot(training_losses)
fig.savefig('training_losses_di.png', dpi=fig.dpi)
