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

TOPIC = np.uint32
ctypedef np.uint32_t TOPIC_t

WORD = np.uint32
ctypedef np.uint32_t WORD_t

COUNT = np.uint32
ctypedef np.uint32_t COUNT_t


cdef inline TOPIC_t sample_discrete(double[:] probs, double tot):
    cdef TOPIC_t i
    tot *= (<double> rand()) / RAND_MAX
    for i in range(probs.shape[0]):
        tot -= probs[i]
        if tot < 0.:
            return i
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

        def __set__(self,topic_word_counts):
            self.word_topic_c = topic_word_counts.T.astype(COUNT,order='C')

    property document_topic_counts:
        def __get__(self):
            return np.asarray(self.document_topic_c)

        def __set__(self,document_topic_counts):
            self.document_topic_c = document_topic_counts.astype(COUNT,order='C')

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
    # http://trac.cython.org/cython_trac/ticket/749

    cdef double[::1] topic_scores_buf

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
        cdef double score, tot = 0.
        for t in range(self.num_topics):
            score = self.score(t,word,doc_id)
            self.topic_scores_buf[t] = score
            tot += score
        return sample_discrete(self.topic_scores_buf,tot)

    cdef inline double score(self, TOPIC_t topic, WORD_t word, int doc_id):
        return (self.alpha + self.document_topic_c[doc_id,topic]) \
                * (self.beta + self.word_topic_c[word,topic]) \
                  / (self.beta*self.num_vocab + self.topic_c[topic])

    ### adding documents and initialization

    def add_documents(self, spmatrix, initialization='singletopic'):
        assert spmatrix.shape[1] == self.num_vocab, 'vocabulary size mismatch'
        assert initialization in ('forward','singletopic','uniform')
        csr_matrix = spmatrix.tocsr().astype(np.int32)
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
        if initialization == 'forward':
            self.init_sample_forwards(prev_num_documents)
        elif initialization == 'singletopic':
            self.init_single_topic(prev_num_documents)
        elif initialization == 'uniform':
            self.init_uniform(prev_num_documents)

    cdef void init_sample_forwards(self, int prev_num_documents):
        cdef int doc, i
        for doc in range(prev_num_documents,self.docstarts.shape[0]-1):
            for i in range(self.docstarts[doc],self.docstarts[doc+1]):
                self.labels[i] = self.sample_topic(self.words[i],doc)
                self.count(self.labels[i],self.words[i],doc,1)

    cdef void init_single_topic(self, int prev_num_documents):
        self.labels[self.docstarts[prev_num_documents]:] = 0
        self.init_count(prev_num_documents)

    @cython.wraparound(True)
    cdef void init_uniform(self, int prev_num_documents):
        cdef int firstidx = self.docstarts[prev_num_documents]
        self.labels[firstidx:] = \
                np.random.randint(self.num_topics,size=self.docstarts[-1] - firstidx)
        self.init_count(prev_num_documents)

    cdef void init_count(self, int prev_num_documents):
        cdef int doc, i
        for doc in range(prev_num_documents,self.docstarts.shape[0]-1):
            for i in range(self.docstarts[doc],self.docstarts[doc+1]):
                self.count(self.labels[i],self.words[i],doc,1)


    ### perplexity

    def perplexity(self, spmatrix):
        assert spmatrix.shape[1] == self.num_vocab, 'vocabulary size mismatch'
        csr_matrix = spmatrix.tocsr()

        word_topic_ps = np.asarray(self.word_topic_c) + self.beta
        word_topic_ps /= word_topic_ps.sum(0)
        word_ps = word_topic_ps.sum(1) / self.num_topics

        return np.exp(
                -(np.log(word_ps) * np.asarray(csr_matrix.sum(0)).ravel()
                    ).sum() / csr_matrix.sum())

