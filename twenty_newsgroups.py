import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file

import gensim
from gensim.corpora.svmlightcorpus import SvmLightCorpus
from gensim.models.ldamodel import LdaModel as gensim_LDA

cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset="train", categories=cats)

vectorizer = CountVectorizer(stop_words="english", max_features=5000)
vectors = vectorizer.fit_transform(newsgroups_train.data)
ndarrays = vectors_to_list(vectors)

# this was the easiest way to interface with gensim... obviously
# wasteful to go to disk (and back), but good for now.
with open("newsgroups.txt", 'w') as f:
    dump_svmlight_file(vectors, newsgroups_train.target, f)

corp = SvmLightCorpus("newsgroups.txt")
gLDA = gensim_LDA(corp, alpha=5, eta=20, num_topics=2)
gLDA_lambda = gLDA.state.get_lambda()

print "ok -- gensim model fit"

yalda_model = lda.CollapsedSampler(
        alpha = 5., # number-of-topics-in-document concentration parameter
        beta = 20., # number-of-words-in-topic concentration parameter
        num_topics = 2,
        num_vocab = 5000)


def vectors_to_list(feature_vectors):
    '''
    for testing only! not exactly efficient. 
    '''
    array_ls = []
    for i in xrange(feature_vectors.shape[0]):
        array_ls.append(feature_vectors.getrow(i).toarray()[0].astype("int32"))
    return array_ls


