import unittest
import numpy
from rlr.glr import blr

class DataModelTest(unittest.TestCase) :

    def test_a_alpha(self) :
        labels = numpy.array([1] * 6 + [0] * 6)
        examples = numpy.array([1, 0] * 6).reshape(12, 1)
        

        print blr(labels, examples, 0)
        assert 1 == 0


    def test_b_alpha(self) :
        labels = numpy.array([0] * 6 + [0] * 6)
        examples = numpy.array([1, 0] * 6).reshape(12, 1)
        

        print blr(labels, examples, 0)
        assert 1 == 0


    def test_c_alpha(self) :
        labels = numpy.array([1] * 6 + [1] * 6)
        examples = numpy.array([1, 0] * 6).reshape(12, 1)
        
        print blr(labels, examples, 0)
        assert 1 == 0


    def test_d_alpha(self) :
        labels = numpy.array([1] * 6 + [0] * 6)
        examples = numpy.array([1, 0] * 6).reshape(12, 1)

        case_weights = numpy.arange(1, 13) * 1./12
        case_weights = numpy.array([0.5] * 12)

        print blr(labels, examples, 0, case_weights)
        assert 1 == 0
