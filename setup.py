# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='assignment1',
    version='0.1.0',
    description='My solutions to Stanford University NLP and Deep Learning cs224 class',
    long_description=readme,
    author='Junior Teudjio Mbativou',
    author_email='teudjiombativou@gmail.com',
    url='https://github.com/teudjio',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'nose',
        'sphinx',
        'matplotlib',
        'scipy',
        'numpy',
        'sklearn',
    ]
)

