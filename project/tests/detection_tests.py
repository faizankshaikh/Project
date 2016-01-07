from nose.tools import *
from packages.detection import Detection

detect = Detection()

def object_tester():
    assert_equal(detect.tester, 100)
    
def data_loader_tester():
    #TODO how to print in tests
    x, y = detect.data_loader()
    
    #assert_equal(str(type(detect.data)), "<class 'pandas.core.frame.DataFrame'>")
    
    assert_equal(str(type(x)), "<type 'numpy.ndarray'>")
    assert_equal(str(type(y)), "<type 'numpy.ndarray'>")
    
    assert_equal(x.shape, (7705, 1, 32, 32))
    assert_equal(y.shape, (7705, ))
    
    proc_x = detect.preprocess(x)
    assert_equal(proc_x.shape, (7705, 1, 32, 32))
    
    # assert_less(proc_x[0, 0, 0, 0], 1)
    
def visualize_tester():
    # detect.visualize()
    pass
    
#TODO preprocess not checked
#TODO augment_creator not checked    
