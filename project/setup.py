try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    
#TODO add requirements, author_email
    
config = {
    'description' : 'Pattern Detection and Recognition using Deep Learning', 
    'author' : ['jalFaizy', 'manasikhapke', 'shraddhabgunjal'], 
    'url' : '',
    'author_email' : 'faizankshaikh@gmail.com',
    'version' : '0.1',
    'license' : 'MIT'
    'install_requires' : [],
    'packages' : [''],
    'scripts' : [],
    'name' : 'Text Spotting'
}

setup(**config)
