import pandas as pd
import zipfile
import os
import re
import nltk
from nltk.stem.porter import PorterStemmer
import html
import inflect
from nltk.corpus import stopwords
import unicodedata
import contractions as c
from collections import Counter
nltk.download('punkt')


def create_dataframe(zip_path='../data/bbc.zip'):
    datadir = './bbc'
    zip = zipfile.ZipFile(zip_path)
    zip.extractall()

    dirs = os.listdir(datadir)
    data = []
    for each in dirs:
        if each == 'README.TXT':
            continue
        sub_dir = os.listdir(datadir+'/'+each)
        for every in sub_dir:
            path = datadir+'/'+each+'/'+every
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
            data.append([text, each])
    return pd.DataFrame(data, columns=['text', 'label'])


def fixup(x):
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'"). \
        replace('amp;', '&'). \
        replace('#146;', "'"). \
        replace('nbsp;', ' '). \
        replace('#36;', '$'). \
        replace('\\n', "\n"). \
        replace('quot;', "'"). \
        replace('<br />', "\n"). \
        replace('\\"', '"'). \
        replace('<unk>', 'u_n'). \
        replace(' @.@ ', '.'). \
        replace(' @-@ ', '-'). \
        replace('\\', ' \\ '). \
        replace('>', ''). \
        replace('-', '')
    return re1.sub(' ', html.unescape(x))


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    # tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [w for w in tokens if len(w) >= 2]
    tokens = [fixup(x) for x in tokens]
    # stem
    # tokens = stem_tokens(tokens, stemmer)
    return tokens


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word). \
            encode('ascii', 'ignore'). \
            decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of
    tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def contract(sent):
    ans = ''
    for each in sent.split():
        ans = ans + c.fix(each) + ' '
    return ans.strip()


def normalize(docs):
    docs = [fixup(x) for x in docs]
    docs = to_lowercase(docs)
    docs = remove_non_ascii(docs)
    docs = [contract(x) for x in docs]
    docs = remove_punctuation(docs)
    docs = replace_numbers(docs)
    docs = remove_stopwords(docs)
    docs = [tokenize(x) for x in docs]
    sent = []
    for each in docs:
        sent = sent + each
    counter = Counter(sent)
    docs = [[x for x in doc if counter[x] >= 3] for doc in docs]
    return docs


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
dat = create_dataframe()
dat.text = normalize(dat.text)
dat.to_csv('../data/cleaned_documents.csv')
