from nose.tools import *
from packages.recognition import Recognition as rg

def visualize_test():
    recognize = rg()
    #recognize.load_dataset()
    recognize.visualize_dataset()
