"""PopSearch

:author: Sebastian Flennerhag
:copyright: 2018
:license: MIT

PopSearch - Population Based Hyperparameter Search.
"""

from setuptools import setup, find_packages
import popsearch

VERSION = popsearch.__version__

setup(name='popsearch',
      version=VERSION,
      description='Population Based Hyper-parameter Search',
      author='Sebastian Flennerhag',
      author_email='sebastianflennerhag@hotmail.com',
      url='https://github.com/flennerhag/popsearch',
      packages=find_packages(),
      package_data={'': ['LICENSE.txt',
                         'README.md',
                         'requirements.txt'
                         ]
                    },
      include_package_data=True,
      install_requires=['numpy>=1.11',
                        'scipy>=0.17'],
      license='MIT',
      platforms='any',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Development Status :: 4 - Beta',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'
                   ],
      long_description="""
A package for population-based hyper-parameter search.

This project is hosted at https://github.com/flennerhag/popsearch
""")
