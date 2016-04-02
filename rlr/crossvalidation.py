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
               reps=5,
               search_space=[.00001, .0001, .001, .01, .1, 1],
               default_alpha=0.01,
               randomize=True # this arg is ignored
               ):


    if len(labels) < k:
        logger.info(
            "gridSearch: Too few labels ({0}) for kfold {1} cross-validation, "
            "returning default_alpha {2}"
            "".format(len(labels), k, default_alpha)
        )
        return default_alpha

    if num_cores < 2 :
        from multiprocessing.dummy import Pool
    else :
        from .backport import Pool

    pool = Pool()

    logger.info('using cross validation to find optimum alpha...')
    best_score = 0
    best_alpha = default_alpha

    alpha_tester = AlphaTester(learner)

    for alpha in search_space:
        scores = []
        for _ in range(reps):
            permutation = numpy.random.permutation(len(labels))
            labeled_examples = (examples[permutation], labels[permutation])

            score_jobs = [pool.apply_async(alpha_tester,
                                           (training, validation, alpha))
                          for training, validation in
                          kFolds(labeled_examples, k)]

            scores.extend([job.get() for job in score_jobs])

        scores = [score for score in scores if score is not None]
        if scores:
            average_score = numpy.mean(scores)
            stdev = numpy.std(scores)
        else:
            average_score = 0
            stdev = 0

        logger.debug("alpha {0} mean {1} std {2}".format(alpha, average_score, stdev))

        if average_score >= best_score :
            best_score = average_score
            best_alpha = alpha

    logger.info('best score: %f' % best_score)
    logger.info('best alpha: %f' % best_alpha)
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

    denom_squared = ((true_dupes + false_dupes)
                     * (true_dupes + false_distinct)
                     * (true_distinct + false_dupes)
                     * (true_distinct + false_distinct))

    if denom_squared == 0:
        return 0

    matthews_cc = ((true_dupes * true_distinct
                    - false_dupes * false_distinct)
                   /numpy.sqrt(denom_squared))

    if numpy.isnan(matthews_cc):
        # Shouldn't happen
        logger.debug('matthews_cc is nan: ' + str([true_dupes, false_dupes, true_distinct, false_distinct]))
        return 0

    return matthews_cc

