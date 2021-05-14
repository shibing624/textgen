# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

from textgen import __version__

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name='textgen',
    version=__version__,
    description='Text Generation Model',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/text-generation',
    license="Apache 2.0",
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        "License :: OSI Approved :: Apache Software License",
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    keywords='textgen,text-generation,Text Generation Tool,ernie-gen,chinese text generation',
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(exclude=['tests']),
    package_dir={'textgen': 'textgen'},
    package_data={
        'textgen': ['*.*', '../LICENSE', '../README.*', '../*.txt', '../examples/*', ],
    }
)
