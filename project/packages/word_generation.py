#TODO Write module 3 here
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

class word_generation(object):

    def __init__(self):
        self.threshold=80
        
    def words(self,file_name):
        try:
            fp=open(file_name,'r')
        except:
            raise 
        text=fp.read()
        return re.findall('[a-z]+',text.lower())
        
