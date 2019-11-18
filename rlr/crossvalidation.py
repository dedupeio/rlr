#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from builtins import range

import numpy
import logging
import warnings
import collections

logger = logging.getLogger(__name__)


def gridSearch(examples,
               labels,
               learner,
               num_cores,
               k=3,
               search_space=[.00001, .0001, .001, .01, .1, 1],
               randomize=True):

    if num_cores < 2 :
        from multiprocessing.dummy import Pool
    else :
        from .backport import Pool

    repeats = max(1, int(150/len(labels)))

    pool = Pool()

    logger.info('using cross validation to find optimum alpha...')

    alpha_tester = AlphaTester(learner)
    alpha_scores = collections.defaultdict(list)

    for repeat in range(repeats):
        permutation = numpy.random.permutation(labels.size)

        examples = examples[permutation]
        labels = labels[permutation]

        labeled_examples = (examples, labels)
        
        for alpha in search_space:

            score_jobs = [pool.apply_async(alpha_tester,
                                           (subset, validation, alpha))
                          for subset, validation in
                          kFolds(labeled_examples, k)]

            scores = [job.get() for job in score_jobs]

            alpha_scores[alpha].extend(scores)

    best_alpha, score = max(alpha_scores.items(),
                            key=lambda x: reduceScores(x[1]))

    logger.info('optimum alpha: %f, score %s' % (best_alpha, reduceScores(score)))
    pool.close()
    pool.join()


    return best_alpha

# http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/
def kFolds(labeled_examples, k):
    examples, labels = labeled_examples

    if k < 2 :
        raise ValueError("Number of folds must be at least 2")
    
    if len(labels) < 2 :
        raise ValueError("At least two training datum are required")

    for i in range(k):
        selected_indices = range(i, examples.shape[0], k)
        
        validation = (examples[selected_indices, :], 
                      labels[selected_indices])
        training = (numpy.delete(examples, selected_indices, axis=0),
                    numpy.delete(labels, selected_indices))

        if len(training[1]) and len(validation[1]) :
            yield (training, validation)
        else :
            warnings.warn("Only providing %s folds out of %s requested" % 
                          (i, k))
            break

class AlphaTester(object) :
    def __init__(self, learner) : # pragma : no cover
        self.learner = learner

    def __call__(self, training, validation, alpha) :
        training_examples, training_labels = training
        self.learner.alpha = alpha
        self.learner.fit_alpha(training_examples, training_labels, None)

        validation_examples, validation_labels = validation
        predictions = self.learner.predict_proba(validation_examples) 

        return scorePredictions(validation_labels, predictions)
        
def scorePredictions(true_labels, predictions) :
    # http://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    true_dupes = int(numpy.sum(predictions[true_labels == 1] > 0.5))
    false_dupes = int(numpy.sum(predictions[true_labels == 0] > 0.5))

    true_distinct = int(numpy.sum(predictions[true_labels == 0] <= 0.5))
    false_distinct = int(numpy.sum(predictions[true_labels == 1] <= 0.5))

    if not (true_dupes + false_dupes) * (true_distinct + false_distinct) :
        return 0
    
    matthews_cc = ((true_dupes * true_distinct 
                    - false_dupes * false_distinct)
                   /numpy.sqrt((true_dupes + false_dupes)
                               * (true_dupes + false_distinct)
                               * (true_distinct + false_dupes)
                               * (true_distinct + false_distinct)))


    return matthews_cc

def reduceScores(scores) :

    scores = [score for score in scores
              if score is not None and not numpy.isnan(score)]

    if scores :
        average_score = sum(scores)/len(scores)
    else :
        average_score = 0

    return average_score


