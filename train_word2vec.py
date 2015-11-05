#!/usr/bin/env python
# encoding: utf-8

from gensim.models import Word2Vec
import jieba


def get_stopWords(stopWords_fn):
    with open(stopWords_fn, 'rb') as f:
        stopWords_set = {line.strip('\r\t').decode('utf-8') for line in f}
    return stopWords_set


def sentence2words(sentence, stopWords=False, stopWords_set=None):
    """
    split a sentence into words based on jieba
    """
    # seg_words is a generator
    seg_words = jieba.cut(sentence)
    if stopWords:
        words = [word for word in seg_words if word not in stopWords_set and word != ' ']
    else:
        words = [word for word in seg_words]
    return words


class MySentences(object):
    def __init__(self, list_csv):
        stopWords_fn = 'all_stopword.txt'
        self.stopWords_set = get_stopWords(stopWords_fn)
        with open(list_csv, 'r') as f:
            self.fns = [line.strip() for line in f]

    def __iter__(self):
        for fn in self.fns:
            with open(fn, 'r') as f:
                for line in f:
                    yield sentence2words(line.strip(), True, self.stopWords_set)


def train_save(list_csv, model_fn):
    sentences = MySentences(list_csv)
    num_features = 100
    min_word_count = 10
    num_workers = 48
    context = 20
    epoch = 20
    sample = 1e-5
    model = Word2Vec(
        sentences,
        size=num_features,
        min_count=min_word_count,
        workers=num_workers,
        sample=sample,
        window=context,
        iter=epoch,
    )
    model.save(model_fn)
    return model


if __name__ == "__main__":
    model = train_save('sougou_list.csv', 'word2vec_model_0925')

    # get the word vector
    for w in model.most_similar(u'互联网'):
        print w[0], w[1]

    print model.syn0.shape

    print model.similarity(u'网络', u'互联网')

    country_vec = model[u"国家"]
    print type(country_vec)
    print country_vec
