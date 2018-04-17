#!/usr/bin/env python

from __future__ import absolute_import
import io
import re
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def find_version():
    version_file = io.open(os.path.join(os.path.dirname(__file__), 'MCEvidence.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='MCEvidence',
      version=find_version(),
      description='MCEvidence evidence estimation from MCMC chains',
      author='Yabebal Fantaye',
      author_email='yabi@aims.ac.za',
      url="https://github.com/yabebalFantaye/MCEvidence",
      packages=[''],
      scripts=['MCEvidence.py'],
      test_suite='example.py',
      #package_data={'planck_fullgrid_R2': ['AllChains','SingleChains']}
      install_requires=[
          'numpy',
          'statistics',
          'sklearn',
          "scipy (>=0.11.0)",
          'pandas (>=0.14.0)',
          ],
          
      classifiers=[
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
      ],
      keywords=['MCMC', 'Evidence', 'bayesian evidence', 'marginal likelihood']
      )
