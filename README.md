# Build #

```bash
python setup.py build_ext --inplace
```

# Example #

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset="train", categories=cats)

vectorizer = CountVectorizer(stop_words="english", max_features=5000)
vectors = vectorizer.fit_transform(newsgroups_train.data)

### build model

import lda

model = lda.CollapsedSampler(
        alpha = 5., # number-of-topics-in-document concentration parameter
        beta = 20., # number-of-words-in-topic concentration parameter
        num_topics = 25,
        num_vocab = vectors.shape[1],
        )

model.add_documents_spmat(vectors)

### run inference

model.resample()
```

# TODO #
* test!
* split/merge moves
* hogwild
