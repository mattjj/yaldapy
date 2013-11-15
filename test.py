from __future__ import division
import numpy as np
import scipy.sparse as s
from matplotlib import pyplot as plt

import lda

def ibincount(counts):
    'returns an array a such that counts = np.bincount(a)'
    return np.repeat(np.arange(counts.shape[0]),counts)

def generate_synthetic(alpha, beta, num_topics, num_vocab, document_lengths):
    num_documents = len(document_lengths)
    document_topic_mat = np.random.dirichlet(
            np.repeat(alpha / num_topics, num_topics),size=num_documents)
    topic_word_mat = np.random.dirichlet(
            np.repeat(beta / num_vocab, num_vocab), size=num_topics)

    document_word_mat = document_topic_mat.dot(topic_word_mat)
    document_word_counts = [np.random.multinomial(n,wordprobs)
        for n,wordprobs in zip(document_lengths,document_word_mat)]
    document_word_counts = s.coo_matrix(document_word_counts)

    return document_topic_mat, topic_word_mat, document_word_counts

def test_initialize_at_truth():
    global alpha, beta, num_topics, num_vocab, document_lengths, \
            doc_topic, topic_word, docs, model
    alpha = 5.
    beta = 20.
    num_topics = 20
    num_vocab = 1000
    document_lengths = [100]*1000

    doc_topic, topic_word, docs = generate_synthetic(alpha,beta,
            num_topics,num_vocab,document_lengths)

    model = lda.CollapsedSampler(alpha,beta,num_topics,num_vocab)
    model.add_documents(docs)

    # initialize at truth
    model.document_topic_counts = (model.document_topic_counts.sum(1)[:,None] * doc_topic).round()
    model.topic_word_counts = (model.topic_word_counts.sum(1)[:,None] * topic_word).round()

    perplexities = []
    for itr in range(10):
        model.resample(100)
        perplexities.append(model.perplexity(docs))

    plt.matshow(topic_word[:20,:20])
    plt.title('true topic_word on first 20 words')
    plt.matshow(model.topic_word_counts[:20,:20])
    plt.title('topic_word counts on first 20 words')
    plt.figure()
    plt.plot(perplexities)
    plt.title('training perplexities')

def test_synthetic(niter=25,nsubiter=5):
    global alpha, beta, num_topics, num_vocab, document_lengths, \
            doc_topic, topic_word, docs, model
    alpha = 5. # topic-doc
    beta = 20. # topic-word
    num_topics = 20
    num_vocab = 1000
    document_lengths = [500]*5000

    doc_topic, topic_word, docs = generate_synthetic(alpha,beta,
            num_topics,num_vocab,document_lengths)

    model = lda.CollapsedSampler(alpha,beta,num_topics,num_vocab)
    model.add_documents(docs,'singletopic')

    perplexities = []
    for itr in range(niter):
        perplexities.append(model.perplexity(docs))
        print ''
        print perplexities[-1]
        model.resample(nsubiter)
    perplexities.append(model.perplexity(docs))
    print ''
    print perplexities[-1]

    plt.matshow(topic_word.T.dot(topic_word))
    plt.title('true word co-occurrence in topics')
    sampled_topic_word = model.topic_word_counts.astype('double')
    sampled_topic_word /= sampled_topic_word.sum(1)[:,None]
    plt.matshow(sampled_topic_word.T.dot(sampled_topic_word))
    plt.title('sampled word co-occurrence in topics')

    plt.figure()
    plt.plot(perplexities,label='train perplexities')

if __name__ == '__main__':
    test_synthetic()
    plt.show()

