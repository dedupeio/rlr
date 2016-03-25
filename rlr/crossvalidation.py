#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from builtins import range

import numpy
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gridSearch(examples,
               labels,
               learner,
               num_cores,
               k=3,
               search_space=[.00001, .0001, .001, .01, .1, 1],
               randomize=True):

    default_alpha = 0.01
    if len(labels) < k:
        logger.info(
            "gridSearch: Too few labels ({0}) for kfold {1}, "
            "returning default_alpha {2}"
            "".format(len(labels), k, default_alpha)
        )
        return default_alpha

    if num_cores < 2 :
        from multiprocessing.dummy import Pool
    else :
        from .backport import Pool

    pool = Pool()

    permutation = numpy.random.permutation(labels.size)

    examples = examples[permutation]
    labels = labels[permutation]

    labeled_examples = (examples, labels)

    logger.info('using cross validation to find optimum alpha...')
    best_score = 0
    best_alpha = 0.01

    alpha_tester = AlphaTester(learner)

    for alpha in search_space:

        score_jobs = [pool.apply_async(alpha_tester, 
                                       (subset, validation, alpha))
                      for subset, validation in 
                      kFolds(labeled_examples, k)]

        scores = [job.get() for job in score_jobs]
        
        average_score = reduceScores(scores)

        logger.debug("Average Score: %f, alpha: %s" % (average_score, alpha))

        if average_score >= best_score :
            best_score = average_score
            best_alpha = alpha

    logger.info('optimum alpha: %f' % best_alpha)
    pool.close()
    pool.join()

    return best_alpha


def kFolds(labeled_examples, k):
    # http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/

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

    true_dupes = numpy.sum(predictions[true_labels == 1] > 0.5)
    false_dupes = numpy.sum(predictions[true_labels == 0] > 0.5)

    true_distinct = numpy.sum(predictions[true_labels == 0] <= 0.5)
    false_distinct = numpy.sum(predictions[true_labels == 1] <= 0.5)

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
    
    scores = [score for score in scores if score is not None]

    if scores :
        average_score = sum(scores)/len(scores)
    else :
        average_score = 0

    return average_score


