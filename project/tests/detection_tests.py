from nose.tools import *
from packages import detection

def printer_test():
    assert_equal(detection.printer(), 'All is Well')
