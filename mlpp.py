import ast
import sys
import pickle
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import gensim.downloader as api
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


def get_loss_no_di(U, V, X_plus, X_neg):
    t1 = -torch.sum(
        -torch.log(
            1+torch.exp(
                -torch.bmm(
                    U[X_plus[:, 0]].view(U[X_plus[:, 0]].shape[0], 1, E),
                    (V[X_plus[:, 1]]).view(
                        (V[X_plus[:, 1]]).shape[0], E, 1)))))
    - torch.sum(
        -torch.log(
            1+torch.exp(
                -torch.bmm(
                    -U[X_neg[:, 0]].view(U[X_neg[:, 0]].shape[0], 1, E),
                    (V[X_neg[:, 1]]).view(
                        (V[X_neg[:, 1]]).shape[0], E, 1)))))
    t2 = - torch.sum(U_model.log_prob(U))
    t3 = - torch.sum(V_model.log_prob(V))
    return t1 + t2 + t3


def learn_UV(x_plus, x_neg, expt=0, U_val=None):
    alpha = 0.99996
    n_iter = 100
    loss = []
    if expt != 3:
        U = U_model.rsample()
        if is_cuda:
            U = U.cuda()
        U.requires_grad = True
    else:
        U = U_val
    V = V_model.rsample()
    if is_cuda:
        V = V.cuda()
    V.requires_grad = True
    if expt != 3:
        optimizer = torch.optim.Adam([U, V], lr=0.01)
    else:
        optimizer = torch.optim.Adam([V], lr=0.01)
    len_ = len(x_plus)
    for it in range(1):
        for n in range(len_):
            for g in optimizer.param_groups:
                g['lr'] *= alpha
            if expt != 2:
                # Learning of d_i
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
                # End learning of d_i.
            X_plus = torch.tensor(x_plus[n])
            X_neg = torch.tensor(x_neg[n])

            optimizer.zero_grad()
            if expt != 2:
                log_loss = get_loss(U, V, d_i, X_plus, X_neg)
            else:
                log_loss = get_loss_no_di(U, V, X_plus, X_neg)
            log_loss.backward()
            optimizer.step()
            loss.append(log_loss)
            if n % 100 == 0:
                print("Iteration : {}, log_loss : {}".format(n+1, log_loss))
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
# Reading  dataset
dat_file = './data/cleaned_documents.csv'
dat = pd.read_csv(dat_file, converters={"text": ast.literal_eval})
print("Read cleaned doc, size = ", len(dat))


#  Build vocabulary
# If experiment 3 or 4 then partial vocabulary.
if len(sys.argv) > 1 and sys.argv[1] in ['3', '4']:
    gen_model = api.load('glove-wiki-gigaword-100')

E = 100  # size of embedding
words = list(itertools.chain.from_iterable(dat.text))

if len(sys.argv) > 1 and sys.argv[1] in ['3', '4']:
    words = [x for x in words if x in gen_model]

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
if len(sys.argv) > 1:
    x_plus, x_neg = get_pos_neg_sample(dat)
    print("Computed")
    filehandler = open('x_plus', 'wb')
    pickle.dump(x_plus, filehandler)

    filehandler = open('x_neg', 'wb')
    pickle.dump(x_neg, filehandler)
    print("Saved for future")
else:
    try:
        x_plus = pickle.load('x_plus')
        x_neg = pickle.load('x_neg')
        print("loaded pre-saved")
    except Exception:
        x_plus, x_neg = get_pos_neg_sample(dat)
        print("Computed")
        filehandler = open('x_plus', 'wb')
        pickle.dump(x_plus, filehandler)

        filehandler = open('x_neg', 'wb')
        pickle.dump(x_neg, filehandler)
        print("Saved for future")

# Get model definitions.
U_model, V_model, di_model = get_models()


if len(sys.argv) > 1 and sys.argv[1] in ['3', '4']:
    U = torch.tensor(
        [gen_model.get_vector(word)
         if word in gen_model else [0]*300 for word in words])
    if is_cuda:
        U = U.cuda()
    if sys.argv[1] == '4':
        V = U
    else:
        U, V = learn_UV(x_plus, x_neg, 3, U)

else:
    # Learning U and V.
    print("Lets learn U, V!")
    if len(sys.argv) > 1:
        if len(sys.argv) > 1 and sys.argv[1] == '2':
            U, V = learn_UV(x_plus, x_neg, 2)
        else:
            U, V = learn_UV(x_plus, x_neg)
        print("Computed U,V")
        torch.save(U, 'u')
        torch.save(V, 'v')
        print("Saved U,V")
    else:
        try:
            U = torch.load('u')
            V = torch.load('v')
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

torch.save(dis, 'di')

# Training loss for learning of document vectors.
n_data = len(trainingLosses[0])
training_losses = np.mean(np.array(trainingLosses), axis=0)

fig = plt.figure()
plt.plot(training_losses)
fig.savefig('training_losses_di.png', dpi=fig.dpi)
