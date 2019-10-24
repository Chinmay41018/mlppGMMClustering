from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing


wiki = WikiCorpus("enwiki-latest-pages-articles.xml.bz2")


class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c.decode("utf-8") for c in content], [title])


documents = TaggedWikiDocument(wiki)
cores = multiprocessing.cpu_count()
model = Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19,
                iter=10, workers=cores)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
fname = "wikiDoc2Vec.model"
model.save(fname)
