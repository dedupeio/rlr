import unittest
import numpy
from rlr.lr import RegularizedLogisticRegression as RLR, phi, gridSearch
import itertools

class DataModelTest(unittest.TestCase) :

    def test_zero_alpha(self) :
        labels = numpy.array([0, 1] * 6).reshape(12, )
        examples = numpy.array([1, 0] * 6).reshape(12, 1)

        classifier = RLR(0, False)

        classifier.fit(examples, labels)

        numpy.testing.assert_almost_equal(classifier.predict_proba(examples)[:,-1],
                                          labels,
                                          3)

    def test_large_alpha(self) :
        labels = numpy.array([0, 1] * 6).reshape(12, )
        examples = numpy.array([1, 0] * 6).reshape(12, 1)

        classifier = RLR(10000, False)

        classifier.fit(examples, labels)

        numpy.testing.assert_almost_equal(classifier.predict_proba(examples)[:,-1],
                                          numpy.array([0.5]*12),
                                          3)

    def test_cv(self) :
        labels = (numpy.arange(120) < 50).astype(int)
        examples = numpy.sqrt(numpy.arange(120)).reshape(120, 1)

        classifier = RLR(0)

        classifier.fit(examples, labels)


    def test_rounding_error(self) :
        examples = numpy.array(
            [[ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               0.5       ,  1.04166663,  5.2750001 ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.5       ,  1.61556602,  1.        ,  0.5       ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               0.5       ,  0.5       ,  0.5       ,  1.70000005,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               1.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               4.78571415,  3.91745281,  1.        ,  0.5       ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               0.5       ,  4.375     ,  4.35227251,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               5.5       ,  3.58712125,  0.        ,  1.5       ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               4.25      ,  5.5       ,  4.60833311,  5.5       ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               1.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               5.5       ,  4.62812519,  0.        ,  2.5       ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               0.5       ,  1.04166663,  0.5       ,  5.01785707,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               1.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               3.35714293,  2.79500008,  1.        ,  0.5       ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               1.75      ,  1.04166663,  0.5       ,  1.70000005,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               1.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               5.5       ,  4.79391909,  1.        ,  0.5       ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               0.5       ,  0.5       ,  0.5       ,  0.5       ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               1.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.5       ,  1.        ,  0.5       ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               0.5       ,  0.95833331,  0.5       ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               4.07142878,  4.31931829,  1.        ,  0.5       ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               4.25      ,  0.5       ,  0.5       ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               5.5       ,  4.94444466,  1.        ,  0.5       ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               4.07142878,  2.5       ,  0.5       ,  0.5       ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               1.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.5       ,  0.63793105,  0.        ,  0.        ],
             [ 1.        ,  0.        ,  1.        ,  0.        ,  0.,
               0.5       ,  0.5       ,  0.5       ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  1.        ,  1.        ,  1.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               0.        ,  0.        ,  0.        ,  0.        ,  0.,
               4.07142878,  2.38010216,  1.        ,  0.5       ]])

        labels = numpy.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

        classifier = RLR(0, False)

        classifier.fit(examples, labels)


    def test_stability_no_cv(self):
        test_data = [{
                        'examples': numpy.array([[ 0.32363278,  0.40278021,  0.0999007 ],
                                                [ 0.32363278,  0.65415895,  0.06500483],
                                                [ 0.32363278,  0.43124139,  0.33864626],
                                                [ 0.71153408,  0.98082042,  0.97221285],
                                                [ 0.23200932,  0.37879705,  0.87567651]]),
                        'labels': numpy.array([0, 0, 1, 1, 1])
                      },
                      {
                        'examples': numpy.array([[0.32363277673721313, 0.4574252665042877, 0.07390517741441727],
                                                 [0.32363277673721313, 0.2044195830821991, 0.1042337492108345],
                                                 [0.2919190526008606, 0.42525577545166016, 0.42221683263778687],
                                                 [0.19814740121364594, 0.2475186437368393, 0.4313957393169403],
                                                 [0.6528099775314331, 0.7313898801803589, 0.4521579444408417],
                                                 [0.32363277673721313, 0.6541589498519897, 0.06500483304262161],
                                                 [0.32363277673721313, 0.4841150641441345, 0.0396646186709404],
                                                 [0.23200932145118713, 0.512259840965271, 0.10688728839159012],
                                                 [0.32363277673721313, 0.3057943284511566, 0.14972583949565887],
                                                 [0.32363277673721313, 0.4090577960014343, 0.19944415986537933]]),
                        'labels': numpy.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0])
                      }]

        for data in test_data:
            weights = []
            classifier = RLR(cv=0)
            for _ in range(10):
                classifier.fit(data['examples'], data['labels'])
                weights.append(classifier.weights)

            for a, b in itertools.combinations(weights, 2):
                numpy.testing.assert_almost_equal(a, b)


    def test_gridsearch(self):
        test_data = [{
                        'examples': numpy.array([[ 0.32363278,  0.40278021,  0.0999007 ],
                                                [ 0.32363278,  0.65415895,  0.06500483],
                                                [ 0.32363278,  0.43124139,  0.33864626],
                                                [ 0.71153408,  0.98082042,  0.97221285],
                                                [ 0.23200932,  0.37879705,  0.87567651]]),
                        'labels': numpy.array([0, 0, 1, 1, 1])
                      },
                      {
                        'examples': numpy.array([[0.32363277673721313, 0.4574252665042877, 0.07390517741441727],
                                                 [0.32363277673721313, 0.2044195830821991, 0.1042337492108345],
                                                 [0.2919190526008606, 0.42525577545166016, 0.42221683263778687],
                                                 [0.19814740121364594, 0.2475186437368393, 0.4313957393169403],
                                                 [0.6528099775314331, 0.7313898801803589, 0.4521579444408417],
                                                 [0.32363277673721313, 0.6541589498519897, 0.06500483304262161],
                                                 [0.32363277673721313, 0.4841150641441345, 0.0396646186709404],
                                                 [0.23200932145118713, 0.512259840965271, 0.10688728839159012],
                                                 [0.32363277673721313, 0.3057943284511566, 0.14972583949565887],
                                                 [0.32363277673721313, 0.4090577960014343, 0.19944415986537933]]),
                        'labels': numpy.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0])
                      }]
        
        for data in test_data:
            alphas = []
            classifier = RLR()
            for _ in range(10):
                alpha = gridSearch(data['examples'], data['labels'], classifier,
                                   classifier.num_cores, classifier.cv)
                alphas.append(alpha)

            assert len(set(alphas)) == 1
            
if __name__ == '__main__':
    unittest.main()
