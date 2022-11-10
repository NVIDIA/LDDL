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

import dask.bag as db
import nltk
import os
import random


def _filter_empty_strs(bag_strs):
  return bag_strs.map(lambda s: s.strip()).filter(lambda s: len(s) > 0)


def _find_files_under(path, extensions={'.txt'}):
  all_files = []
  for current_dir, sub_dirs, file_names in os.walk(path):
    for file_name in file_names:
      if os.path.splitext(file_name)[1] in extensions:
        all_files.append(os.path.join(current_dir, file_name))
  return list(sorted(all_files))


def _total_bytes_of(files):
  return sum(map(os.path.getsize, files))


def estimate_block_size(paths, num_blocks):
  total_bytes = 0
  for p in paths:
    if p is None:
      continue
    total_bytes += _total_bytes_of(_find_files_under(p))
  print('total_bytes = {}, num_blocks = {}'.format(total_bytes, num_blocks))
  block_size = round(total_bytes / num_blocks)
  print('block_size = {} bytes'.format(block_size))
  return block_size


def _read_bag_of_text(
    path,
    blocksize=None,
    sample_ratio=1.0,
    sample_seed=12345,
):
  input_files = _find_files_under(path)
  bag_strs = db.read_text(input_files, blocksize=blocksize)
  bag_strs = _filter_empty_strs(bag_strs)
  if sample_ratio < 1.0:
    bag_strs = bag_strs.random_sample(sample_ratio, random_state=sample_seed)
  return bag_strs


def read_wikipedia(
    path,
    lang='en',
    blocksize=None,
    sample_ratio=1.0,
    sample_seed=12345,
):
  return _read_bag_of_text(
      os.path.join(path, lang),
      blocksize=blocksize,
      sample_ratio=sample_ratio,
      sample_seed=sample_seed,
  )


def read_books(
    path,
    blocksize=None,
    sample_ratio=1.0,
    sample_seed=12345,
):
  return _read_bag_of_text(
      path,
      blocksize=blocksize,
      sample_ratio=sample_ratio,
      sample_seed=sample_seed,
  )


def read_common_crawl(
    path,
    blocksize=None,
    sample_ratio=1.0,
    sample_seed=12345,
):
  return _read_bag_of_text(
      path,
      blocksize=blocksize,
      sample_ratio=sample_ratio,
      sample_seed=sample_seed,
  )

def read_open_webtext(
    path,
    blocksize=None,
    sample_ratio=1.0,
    sample_seed=12345,
):
  return _read_bag_of_text(
      path,
      blocksize=blocksize,
      sample_ratio=sample_ratio,
      sample_seed=sample_seed,
  )

def split_id_text(raw_text):
  # The first token is the document id.
  i = 0
  while i < len(raw_text) and not raw_text[i].isspace():
    i += 1
  return raw_text[:i], raw_text[i + 1:]
