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

import os
import io
import numpy as np
import pathlib
import pyarrow.parquet as pq


def mkdir(d):
  pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def expand_outdir_and_mkdir(outdir):
  outdir = os.path.abspath(os.path.expanduser(outdir))
  mkdir(outdir)
  return outdir


def get_all_files_paths_under(root):
  return (
      os.path.join(r, f) for r, subdirs, files in os.walk(root) for f in files)


def get_all_parquets_under(path):
  return sorted([
      p for p in get_all_files_paths_under(path)
      if '.parquet' in os.path.splitext(p)[1]
  ])


def get_all_bin_ids(file_paths):

  def is_binned_parquet(p):
    return '_' in os.path.splitext(p)[1]

  def get_bin_id(p):
    return int(os.path.splitext(p)[1].split('_')[-1])

  bin_ids = list(
      sorted(set((get_bin_id(p) for p in file_paths if is_binned_parquet(p)))))
  for a, e in zip(bin_ids, range(len(bin_ids))):
    if a != e:
      raise ValueError('bin id must be contiguous integers starting from 0!')
  return bin_ids


def get_file_paths_for_bin_id(file_paths, bin_id):
  return [
      p for p in file_paths
      if '.parquet_{}'.format(bin_id) == os.path.splitext(p)[1]
  ]


def get_num_samples_of_parquet(path):
  return len(pq.read_table(path))


def attach_bool_arg(parser, flag_name, default=False, help_str=None):
  attr_name = flag_name.replace('-', '_')
  parser.add_argument(
      '--{}'.format(flag_name),
      dest=attr_name,
      action='store_true',
      help=flag_name.replace('-', ' ') if help_str is None else help_str,
  )
  parser.add_argument(
      '--no-{}'.format(flag_name),
      dest=attr_name,
      action='store_false',
      help=flag_name.replace('-', ' ') if help_str is None else help_str,
  )
  parser.set_defaults(**{attr_name: default})


def serialize_np_array(a):
  memfile = io.BytesIO()
  np.save(memfile, a)
  memfile.seek(0)
  return memfile.read()


def deserialize_np_array(b):
  memfile = io.BytesIO()
  memfile.write(b)
  memfile.seek(0)
  return np.load(memfile)
