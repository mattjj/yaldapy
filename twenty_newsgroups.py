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

# this was the easiest way to interface with gensim... obviously
# wasteful to go to disk (and back), but good for now.
with open("newsgroups.txt", 'w') as f:
    dump_svmlight_file(vectors, newsgroups_train.target, f)

corp = SvmLightCorpus("newsgroups.txt")
gLDA = gensim_LDA(corp, alpha=1, eta=2, num_topics=2)
gLDA_lambda = gLDA.state.get_lambda()
print "ok -- gensim model fit"

#### yalda
import lda
model = lda.CollapsedSampler(
        alpha = 1., # number-of-topics-in-document concentration parameter
        beta = 2., # number-of-words-in-topic concentration parameter
        num_topics = 2,
        num_vocab = vectors.shape[1])
model.add_documents(vectors)
model.resample(100)

counts = model.topic_word_counts
# I believe the topic_word_counts is analogous to
# the \lambda in gensim.

