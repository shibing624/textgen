# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__version__ = '0.0.4'
with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

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
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires=">=3.6",
    keywords='textgen,text-generation,Text Generation Tool,ernie-gen,chinese text generation',
    install_requires=[
        'colorlog',
        'jieba>=0.39',
        'transformers',
        'datasets',
        'gensim>=4.0.0',
        'text2vec',
        'tensorboardX',
        'tqdm',
        'pandas',
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'textgen': 'textgen'},
    package_data={
        'textgen': ['*.*', ],
    }
)
