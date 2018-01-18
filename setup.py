from setuptools import setup
from setuptools import find_packages
import unittest


def readme():
    with open('README.md') as f:
        return f.read()


def license():
    with open('LICENSE.md') as f:
        return f.read()


setup(name='sparsecomputation',
      version='0.1dev',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
      ],
      description='Sparse Computation for sparsifying similarity matrices',
      keywords='pairwise similarity, classification, sparsification,' +
      'clustering',
      url='https://github.com/hochbaumGroup/sparsecomputation',
      author='Titouan Jehl, Quico Spaen, Philipp Baumann',
      author_email='qspaen@berkeley.edu',
      license=license(),
      long_description=readme(),
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=['numpy', 'scipy', 'sklearn', 'six'],
      setup_requires=['pytest-runner', ],
      tests_require=['pytest', ],
      zip_safe=False
      )
