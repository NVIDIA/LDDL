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

import argparse
import dask
import dask.bag as db
import dask.distributed
import functools
import nltk
import os
import pyarrow as pa
import time

from lddl.dask.readers import (read_open_webtext, read_wikipedia, read_books,
                               read_common_crawl, estimate_block_size)
from lddl.utils import expand_outdir_and_mkdir
from lddl.download.utils import parse_str_of_num_bytes


def _get_sequences(wikipedia_path=None,
                   books_path=None,
                   common_crawl_path=None,
                   open_webtext_path=None,
                   wikipedia_lang='en',
                   target_seq_length=128,
                   short_seq_prob=0.1,
                   blocksize=None,
                   num_blocks=None):
  if num_blocks is not None:
    if blocksize is not None:
      raise ValueError('Only one of num_blocks or blocksize needs to be set!')
    blocksize = estimate_block_size(
        (wikipedia_path, books_path, common_crawl_path, open_webtext_path),
        num_blocks,
    )
  bags = []
  if wikipedia_path is not None:
    bags.append(
        read_wikipedia(
            wikipedia_path,
            lang=wikipedia_lang,
            blocksize=blocksize,
        ))
  if books_path is not None:
    bags.append(read_books(
        books_path,
        blocksize=blocksize,
    ))
  if common_crawl_path is not None:
    bags.append(read_common_crawl(
        common_crawl_path,
        blocksize=blocksize,
    ))

  if open_webtext_path is not None:
    bags.append(read_open_webtext(
        open_webtext_path,
        blocksize=blocksize,
    ))

  def _segment(article):
    return filter(
        None,
        map(lambda s: s.strip(), nltk.tokenize.sent_tokenize(article)),
    )

  def _aggregate_sentences(sentences):
    # Cutting sentences into chunks that are close to target_seq_length
    # results is in the format of
    # [
    #   {
    #     'sentences': [sent1, sent2],
    #     'num_tokens': [num_tokens1, num_tokens2],
    #   },
    #   {
    #     'sentences': [sent1, sent2, sent3],
    #     'num_tokens': [num_tokens1, num_tokens2, num_tokens3],
    #   },
    #   {
    #     'sentences': [sent1],
    #     'num_tokens': [num_tokens1],
    #   },
    #   ...
    # ]
    results = []
    # Excluding [CLS], [SEP], [SEP]
    target_length = target_seq_length - 3
    chunk = ""
    num_tokens = 0
    for sentence in sentences:
      chunk += " " + sentence
      num_tokens += len(list(sentence.split()))
      if num_tokens >= target_length:
        results.append({
            'sentences': chunk,
            'num_tokens': num_tokens,
            'target_length': target_length,
        })
        chunk = ""
        num_tokens = 0
    if num_tokens > 0:
      results.append({
          'sentences': chunk,
          'num_tokens': num_tokens,
          'target_length': target_length,
      })
    return results

  def _generate_sequences(article):
    return _aggregate_sentences(_segment(article))

  return db.concat(bags).map(_generate_sequences).flatten()


def save(pairs, path, output_format='parquet'):
  if output_format == 'parquet':
    pairs.to_dataframe(meta={
        'sentences': str,
    }).to_parquet(
        path,
        engine='pyarrow',
        write_index=False,
        schema={
            'sentences': pa.string(),
        },
    )
  elif output_format == 'txt':
    pairs = pairs.map(lambda p: '{}'.format(p['sentences'],)).to_textfiles(
        os.path.join(path, '*.txt'))
  else:
    raise ValueError('Format {} not supported!'.format(output_format))


def main(args):

  if args.schedule == 'mpi':
    from dask_mpi import initialize
    initialize()
    client = dask.distributed.Client()
  else:
    client = dask.distributed.Client(
        n_workers=args.local_n_workers,
        threads_per_worker=args.local_threads_per_worker,
    )

  nltk.download('punkt')

  tic = time.perf_counter()
  sequences = _get_sequences(
      wikipedia_path=args.wikipedia,
      books_path=args.books,
      common_crawl_path=args.common_crawl,
      open_webtext_path=args.open_webtext,
      wikipedia_lang=args.wikipedia_lang,
      target_seq_length=args.target_seq_length,
      short_seq_prob=args.short_seq_prob,
      blocksize=args.block_size,
      num_blocks=args.num_blocks,
  )

  args.sink = expand_outdir_and_mkdir(args.sink)
  save(sequences, args.sink, output_format=args.output_format)
  print('Running the dask pipeline took {} s'.format(time.perf_counter() - tic))


def attach_args(
    parser=argparse.ArgumentParser('BART pretrain dataset dask pipeline')):
  parser.add_argument(
      '--schedule',
      type=str,
      default='mpi',
      choices=['mpi', 'local'],
      help='how the dask pipeline is scheduled',
  )
  parser.add_argument(
      '--local-n-workers',
      type=int,
      default=os.cpu_count(),
      help='number of worker processes for the local cluster; '
      'only used when --schedule=local',
  )
  parser.add_argument(
      '--local-threads-per-worker',
      type=int,
      default=1,
      help='number of Python user-level threads per worker process for the '
      'local cluster; only used when --schedule=local',
  )
  parser.add_argument(
      '--wikipedia',
      type=str,
      default=None,
      help='path to the Wikipedia corpus',
  )
  parser.add_argument(
      '--books',
      type=str,
      default=None,
      help='path to the Toronto books corpus',
  )
  parser.add_argument(
      '--common-crawl',
      type=str,
      default=None,
      help='path to the Common Crawl news corpus',
  )
  parser.add_argument(
      '--open-webtext',
      type=str,
      default=None,
      help='path to the Open WebText Corpus',
  )
  parser.add_argument(
      '--sink',
      type=str,
      default=None,
      required=True,
      help='path to the dir to store output files',
  )
  parser.add_argument(
      '--output-format',
      type=str,
      default='parquet',
      choices=['parquet', 'txt'],
      help='output file format',
  )
  parser.add_argument(
      '--wikipedia-lang',
      type=str,
      default='en',
      choices=['en', 'zh'],
      help='wikipedia language type',
  )
  parser.add_argument(
      '--target-seq-length',
      type=int,
      default=128,
      help='target sequence length',
  )
  parser.add_argument(
      '--short-seq-prob',
      type=float,
      default=0.1,
      help='probability to use sequences shorter than --target-seq-length',
  )
  parser.add_argument(
      '--block-size',
      type=functools.partial(parse_str_of_num_bytes, return_str=False),
      default=None,
      metavar='n[KMG]',
      help='The size of each output parquet/txt shard. Since Dask cannot '
      'guarantee perfect load balance, this value is only used as an estimate. '
      'Only one of --block-size and --num-blocks needs to be set, since one '
      'value can be derived from the other. Default: {}'.format(None),
  )
  parser.add_argument(
      '--num-blocks',
      type=int,
      default=None,
      help='The total number of the output parquet/txt shards. Since Dask '
      'cannot guarantee perfect load balance, this value is only used as an '
      'estimate. Only one of --block-size or --num-blocks needs to be set, '
      'since one value can be derived from the other. Default: {}'.format(None),
  )
  return parser


def console_script():
  main(attach_args().parse_args())
