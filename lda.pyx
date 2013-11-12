# distutils: extra_compile_args = -O3 -w
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

cimport cython

from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf, fflush, stdout

TOPIC = np.uint16
ctypedef np.uint16_t TOPIC_t

WORD = np.uint16
ctypedef np.uint16_t WORD_t

COUNT = np.uint16
ctypedef np.uint16_t COUNT_t


cdef TOPIC_t sample_discrete(double[:] probs):
    cdef TOPIC_t i
    cdef double r = 0
    for i in range(probs.shape[0]):
        r += probs[i]
    r *= (<double> rand()) / RAND_MAX
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

    property topic_word_counts:
        def __get__(self):
            return np.asarray(self.word_topic_c).T

    property document_topic_counts:
        def __get__(self):
            return np.asarray(self.document_topic_c)

    property documents:
        @cython.wraparound(True)
        def __get__(self):
            return [np.asarray(self.words[start:stop])
                    for start,stop in zip(self.docstarts[:-1],self.docstarts[1:])]

    property topic_labels:
        @cython.wraparound(True)
        def __get__(self):
            return [np.asarray(self.labels[start:stop])
                    for start,stop in zip(self.docstarts[:-1],self.docstarts[1:])]

    ### internal document representation and cached counts

    cdef WORD_t[::1] words
    cdef TOPIC_t[::1] labels
    cdef int[::1] docstarts

    cdef COUNT_t[:,::1] word_topic_c
    cdef COUNT_t[:,::1] document_topic_c
    cdef COUNT_t[::1] topic_c

    ### pre-allocated temporaries because cython can't do dynamic stack arrays

    cdef double[::1] topic_scores_buf
    cdef double[::1] word_scores_buf

    def __init__(self, double alpha, double beta, int num_topics, int num_vocab):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.num_vocab = num_vocab

        self.words = np.empty(0,dtype=WORD)
        self.labels = np.empty(0,dtype=TOPIC)
        self.docstarts = np.zeros(1,dtype=np.int32)

        self.word_topic_c = np.zeros((num_vocab,num_topics),dtype=COUNT)
        self.document_topic_c = np.zeros((0,num_topics),dtype=COUNT)
        self.topic_c = np.zeros(num_topics,dtype=COUNT)

        self.topic_scores_buf = np.empty(self.num_topics,dtype=np.double)
        self.word_scores_buf = np.empty(self.num_vocab,dtype=np.double)

    ### Gibbs sampling

    def resample(self,int niter):
        cdef int itr, doc, i
        for itr in range(niter):
            for doc in range(self.docstarts.shape[0]-1):
                for i in range(self.docstarts[doc],self.docstarts[doc+1]):
                    self.count(self.labels[i],self.words[i],doc,-1)
                    self.labels[i] = self.sample_topic(self.words[i],doc)
                    self.count(self.labels[i],self.words[i],doc,1)
            printf("."); fflush(stdout)

    cdef inline void count(self, TOPIC_t topic, WORD_t word, int doc_id, int inc):
        self.topic_c[topic] += inc
        self.word_topic_c[word,topic] += inc
        self.document_topic_c[doc_id,topic] += inc

    cdef inline TOPIC_t sample_topic(self, WORD_t word, int doc_id):
        cdef TOPIC_t t
        for t in range(self.num_topics):
            self.topic_scores_buf[t] = self.score(t,word,doc_id)
        return sample_discrete(self.topic_scores_buf)

    cdef inline double score(self, TOPIC_t topic, WORD_t word, int doc_id):
        return (self.alpha + self.document_topic_c[doc_id,topic]) \
                * (self.beta + self.word_topic_c[word,topic]) \
                  / (self.beta*self.num_vocab + self.topic_c[topic])

    ### adding documents and initialization

    def add_documents_spmat(self, spmatrix):
        assert spmatrix.shape[1] == self.num_vocab, 'vocabulary size mismatch'
        csr_matrix = spmatrix.tocsr()
        prev_num_documents = self.document_topic_c.shape[0]

        # extend internal sparse document representation and counts array
        self.docstarts = np.concatenate((
            self.docstarts,
            np.cumsum(np.asarray(csr_matrix.sum(1)).squeeze()).astype(np.int32)))
        self.words = np.concatenate((
            self.words,
            np.repeat(csr_matrix.indices,csr_matrix.data).astype(WORD)))
        self.labels = np.concatenate((
            self.labels,
            np.empty(csr_matrix.data.sum(),dtype=TOPIC))) # filled in below

        self.document_topic_c = np.concatenate((
            self.document_topic_c,
            np.zeros((csr_matrix.shape[0],self.num_topics),dtype=COUNT)))

        # initialize newly added documents
        self.sample_forwards(prev_num_documents)

    cdef void sample_forwards(self, int prev_num_documents):
        cdef int doc, i
        for doc in range(prev_num_documents,self.docstarts.shape[0]-1):
            for i in range(self.docstarts[doc],self.docstarts[doc+1]):
                self.labels[i] = self.sample_topic(self.words[i],doc)
                self.count(self.labels[i],self.words[i],doc,1)

