from __future__ import division
import numpy as np

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

    documents = [ibincount(wordcounts) for wordcounts in document_word_counts]

    return document_topic_mat, topic_word_mat, documents

def test1():
    alpha = 5.
    beta = 20.
    num_toipcs = 50
    num_vocab = 1000
    document_lengths = [100]*1000

    true_doc_topic, true_topic_word, docs = generate_synthetic(alpha,beta,
            num_topics,num_vocab,document_lengths)
    true_doc_word = true_doc_topic.dot(true_topic_word)
    true_word_word = true_doc_word.T.dot(true_doc_word)
    true_word_word /= true_word_word.sum(1)[:,None]

    # TODO


