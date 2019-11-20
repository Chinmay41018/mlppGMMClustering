from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import ast

model = Doc2Vec.load('./data/wikiDoc2Vec.model')
dat = pd.read_csv(
    './data/cleaned_documents.csv', names=['text', 'label'], header=0)
vectors = [ast.literal_eval(each) for each in dat['text']]
labels = dat['label'].values
embed = []
for each in vectors:
    embed.append(list(model.infer_vector(list(each))))
dat['embedding'] = embed
dat.to_csv('./data/embedded_doc2vec.csv')
