#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='lddl',
    version='0.1.0',
    description=
    'Language Datasets and Data Loaders for NVIDIA Deep Learning Examples',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='github.com/NVIDIA/DeepLearningExamples/tools/lddl',
    author='Shang Wang',
    author_email='shangw@nvidia.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'dask[complete]==2021.7.1',
        'distributed==2021.7.1',
        'dask-mpi==2021.11.0',
        'bokeh==2.4.3',
        'pyarrow>=4.0.1',
        'mpi4py==3.1.3',
        'transformers==4.30.0',
        'wikiextractor==3.0.6',
        'news-please @ git+https://github.com/fhamborg/news-please.git@3b7d9fdfeb148ef73f393bb2f2557e6bd878a09f',
        'cchardet==2.1.7',
        'awscli>=1.22.55',
        'wikiextractor @ git+https://github.com/attardi/wikiextractor.git@v3.0.6',
        'gdown==4.5.3',
    ],
    entry_points={
        'console_scripts': [
            'download_wikipedia=lddl.download.wikipedia:console_script',
            'download_books=lddl.download.books:console_script',
            'download_common_crawl=lddl.download.common_crawl:console_script',
            'download_open_webtext=lddl.download.openwebtext:console_script',
            'preprocess_bert_pretrain=lddl.dask.bert.pretrain:console_script',
            'preprocess_bart_pretrain=lddl.dask.bart.pretrain:console_script',
            'balance_dask_output=lddl.dask.load_balance:console_script',
            'generate_num_samples_cache=lddl.dask.load_balance:generate_num_samples_cache',
        ],
    },
)
