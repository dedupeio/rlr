# rlr
L2 Regularized Logistic Regression With Case Weighting

Minimal dependency logistic regression classifer with L2 Regularization and optional case weighting.

[![Build Status](https://travis-ci.org/datamade/rlr.svg)](https://travis-ci.org/datamade/rlr)

```python
labels = numpy.array([1] * 6 + [0] * 6)
examples = numpy.array([1, 0] * 6).reshape(12, 1)

case_weights = numpy.arange(1, 13) * 1./12
case_weights = numpy.array([0.5] * 12)

classifier = rlr.RegularizedLogisticRegression(alpha = 0)
classifier.fit(examples, labels, case_weights=case_weights)

classifier.predict_proba(examples)
[0.5, ... 0.5]
```

