from nose.tools import *
from packages.detection import Detection

detect = Detection()

def object_tester():
    assert_equal(detect.tester, 100)
    
def data_loader_tester():
    detect.data_loader()
    assert_equal(str(type(detect.data)), "<class 'pandas.core.frame.DataFrame'>")
    
def visualize_tester():
    detect.visualize()
