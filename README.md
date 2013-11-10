# Build #

```bash
python setup.py build_ext --inplace
```

# Example #

```python
from __future__ import division
import numpy as np

import lda

### generate data

truemodel = lda.CollapsedSampler(
        alpha = 5., # number-of-topics-in-document concentration parameter
        beta = 20., # number-of-words-in-topic concentration parameter
        num_topics = 100,
        num_vocab = 1000,
        )

# generate 100 documents of length 1000
truemodel.generate_documents([1000]*100)

documents = truemodel.docs

### do some inference

model = lda.CollapsedSampler(
        alpha = 5., # number-of-topics-in-document concentration parameter
        beta = 20., # number-of-words-in-topic concentration parameter
        num_topics = 100,
        num_vocab = 1000,
        )

model.add_documents(documents)
model.resample(niter=10)
```

# TODO #
* test! all I know is it goes fast and doesn't segfault
* split/merge moves
* permute things (random scan)?
