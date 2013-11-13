import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file

import lda 
import numpy
import pdb
import pylab

def sanity_check(nsteps=10, niters=25, plot_f="perplexity.png", normalize=False):
    cats = ['alt.atheism', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset="train", categories=cats)
    newsgroups_test = fetch_20newsgroups(subset="test", categories=cats)
    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)
    test_vectors = vectorizer.transform(newsgroups_test.data)

    yalda = lda.CollapsedSampler(
            alpha = 5., # number-of-topics-in-document concentration parameter
            beta = 20., # number-of-words-in-topic concentration parameter
            num_topics = 20,
            num_vocab = 5000)

    yalda.add_documents_spmat(train_vectors)

    Z_train, Z_test = 1.0, 1.0
    train_0, test_0 = perplexity(yalda, train_vectors), perplexity(yalda, test_vectors)
    if normalize:
        Z_train, Z_test = train_0, test_0
    
    train_perplexities = [train_0/Z_train]
    test_perplexities = [test_0/Z_test]
    for i in xrange(nsteps):
        yalda.resample(niters)
        test_perp = perplexity(yalda, test_vectors) 
        train_perp = perplexity(yalda, train_vectors)
        print "@ step {0}: train perplexity={1}; test perplexity={2}".format(i, train_perp, test_perp)
        # 'normalizing' to show change over time.
        train_perplexities.append(train_perp/Z_train)
        test_perplexities.append(test_perp/Z_test)

    X = range(0, nsteps*niters, niters)
    X = [i*niters for i in xrange(nsteps+1)]
    #pdb.set_trace()
    pylab.clf()
    pylab.plot(X, train_perplexities, label="train")
    pylab.plot(X, test_perplexities, ls="--", label="test")
    pylab.xlabel("iterations")
    pylab.ylabel("perplexity")
    pylab.legend(loc="best")
    if plot_f:
        pylab.savefig(plot_f)


def perplexity(model, D):
    word_ps = numpy.matrix([(model.topic_word_counts[i] + model.alpha)/float(sum(model.topic_word_counts[i] + model.alpha))
                                    for i in xrange(model.topic_word_counts.shape[0])])
    D_sparse = D.tocsr().astype(numpy.int32)
    n_test_docs = D_sparse.shape[0]
    perp = 0
    for doc_i in xrange(n_test_docs):
        doc_perp = 0
        doc_topic_ps = (model.document_topic_counts[doc_i] + model.beta)/float(
                                sum(model.document_topic_counts[doc_i] + model.beta))
        for w_i in D_sparse[doc_i].indices:
            w_n = D_sparse[doc_i,w_i]
            doc_perp += w_n * numpy.log(doc_topic_ps * word_ps[:,w_i])
        # divide by word count for doc
        perp += doc_perp / float(D_sparse[0].indices.shape[0])
    perp = -perp / float(n_test_docs) 
    return perp[0,0]
