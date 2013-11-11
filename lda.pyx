# distutils: name = lda
# distutils: extra_compile_args = -O3 -march=native -w
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

import cython

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

cdef DTYPE_t sample_discrete(double[:] probs):
    cdef DTYPE_t i
    cdef double r = 0
    for i in range(probs.shape[0]):
        r += probs[i]
    r *= np.random.rand()
    for i in range(probs.shape[0]):
        r -= probs[i]
        if r < 0:
            break
    return i

cdef class CollapsedSampler(object):
    ### python-accessible attributes

    cdef public double alpha
    cdef public double beta
    cdef public int num_topics
    cdef public int num_vocab
    cdef public list docs
    cdef public list labels

    property topic_word_counts:
        def __get__(self):
            return np.asarray(self.topic_word_c)

    property document_topic_counts:
        def __get__(self):
            return np.asarray(self.document_topic_c[:len(self.docs)])

    ### internal counts

    cdef DTYPE_t[:,:] topic_word_c
    cdef DTYPE_t[:,:] document_topic_c
    cdef DTYPE_t[:] topic_c

    ### pre-allocated temporaries because cython can't do dynamic stack arrays

    cdef double[:] topic_scores_buf
    cdef double[:] word_scores_buf

    def __init__(self,
            double alpha, double beta,
            int num_topics, int num_vocab,
            int doc_capacity=0):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.num_vocab = num_vocab

        self.docs = []
        self.labels = []

        self.topic_word_c = np.zeros((num_topics,num_vocab),dtype=DTYPE)
        self.document_topic_c = np.zeros((doc_capacity,num_topics),dtype=DTYPE)
        self.topic_c = np.zeros(num_topics,dtype=DTYPE)

        self.topic_scores_buf = np.empty(self.num_topics,dtype=np.double)
        self.word_scores_buf = np.empty(self.num_vocab,dtype=np.double)

    ### Gibbs sampling

    def resample(self,niter=1):
        cdef int itr, doc_id, word_idx
        cdef DTYPE_t[:] doc, labels
        for itr in range(niter):
            for doc_id in range(len(self.docs)):
                doc = self.docs[doc_id]
                labels = self.labels[doc_id]
                for word_idx in range(doc.shape[0]):
                    self.count(labels[word_idx],doc[word_idx],doc_id,-1)
                    labels[word_idx] = self.gibbs_sample_topic(doc[word_idx],doc_id)
                    self.count(labels[word_idx],doc[word_idx],doc_id,1)

    cdef void count(self, int topic, DTYPE_t word, int doc_id, DTYPE_t inc):
        self.topic_c[topic] += inc
        self.topic_word_c[topic,word] += inc
        self.document_topic_c[doc_id,topic] += inc

    cdef DTYPE_t gibbs_sample_topic(self, DTYPE_t word, int doc_id):
        cdef double[:] scores = self.topic_scores_buf
        cdef int t
        for t in range(self.num_topics):
            scores[t] = self.gibbs_score(t,word,doc_id)
        return sample_discrete(scores)

    cdef double gibbs_score(self, int topic, DTYPE_t word, int doc_id):
        return (self.alpha/self.num_topics + self.document_topic_c[doc_id,topic]) \
                * (self.beta/self.num_vocab + self.topic_word_c[topic,word]) \
                  / (self.beta + self.topic_c[topic])

    ### adding documents and initialization

    def add_documents(self, list documents):
        self.ensure_capacity(len(self.docs) + len(documents))
        for doc in documents:
            self.labels.append(self.generate_labels(doc))
            self.docs.append(doc)

    def ensure_capacity(self, int required_capacity):
        cdef DTYPE_t[:,:] new_document_topic_c
        if self.document_topic_c.shape[0] < required_capacity:
            new_document_topic_c = \
                    np.zeros((required_capacity,self.document_topic_c.shape[1]),dtype=DTYPE)
            new_document_topic_c[:len(self.docs)] = self.document_topic_c[:len(self.docs)]
            self.document_topic_c = new_document_topic_c

    cdef DTYPE_t[:] generate_labels(self, np.ndarray[DTYPE_t,ndim=1,mode='c'] npdoc):
        cdef int i
        cdef int N = npdoc.shape[0]
        cdef int doc_id = len(self.docs)
        cdef np.ndarray[DTYPE_t,ndim=1,mode='c'] nplabels = np.empty(N,dtype=DTYPE)

        cdef DTYPE_t[:] labels = nplabels
        cdef DTYPE_t[:] doc = npdoc
        for i in range(N):
            labels[i] = self.gibbs_sample_topic(doc[i],doc_id)
            self.count(labels[i],doc[i],doc_id,1)
        return labels

    ### generating synthetic data

    def generate_documents(self, list document_lengths):
        self.ensure_capacity(len(self.docs) + len(document_lengths))
        for N in document_lengths:
            labels, doc = self.generate_labels_and_words(N)
            self.docs.append(doc)
            self.labels.append(labels)

    cdef tuple generate_labels_and_words(self, int N):
        cdef int i
        cdef int doc_id = len(self.docs)
        cdef np.ndarray[DTYPE_t,ndim=1,mode='c'] nplabels = np.empty(N,dtype=DTYPE)
        cdef np.ndarray[DTYPE_t,ndim=1,mode='c'] npwords = np.empty(N,dtype=DTYPE)

        cdef DTYPE_t[:] labels = nplabels
        cdef DTYPE_t[:] words = npwords

        for i in range(N):
            labels[i] = self.prior_sample_topic(doc_id)
            words[i] = self.prior_sample_word(labels[i])
            self.count(labels[i],words[i],doc_id,1)

        return nplabels, npwords

    cdef DTYPE_t prior_sample_topic(self, int doc_id):
        cdef double[:] scores = self.topic_scores_buf
        cdef int t
        for t in range(self.num_topics):
            scores[t] = self.alpha/self.num_topics + self.document_topic_c[doc_id,t]
        return sample_discrete(scores)

    cdef DTYPE_t prior_sample_word(self, int topic):
        cdef double[:] scores = self.word_scores_buf
        cdef int w
        for w in range(self.num_vocab):
            scores[w] = self.beta/self.num_vocab + self.topic_word_c[topic,w]
        return sample_discrete(scores)

