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
import functools
import multiprocessing
import os
import shutil
from glob import glob
import subprocess
import tqdm
import gdown

from lddl.utils import (
    expand_outdir_and_mkdir,
    mkdir,
    get_all_files_paths_under,
    attach_bool_arg,
)


def attach_args(parser=argparse.ArgumentParser("""
OpenWebTextCorpus Downloader performs the following steps:
- Step 1: Download OpenWebTextCorpus
  (https://skylion007.github.io/OpenWebTextCorpus/)
  from provided google drive url and extract the raw text of the articles to
  the directory specified by the --outdir flag.
- Step 2: Prepare and aggregate the raw text into text shards in the 'source'
  subdirectory under the directory specified by the --outdir flag. The text
  shards under the 'source' subdirectory can then be used as the input to the
  LDDL preprocessor.
All steps are executed by default. Each step, before it starts, expects the
previous steps already finish. You can turn Step 1 off by --no-download, and
turn Step 2 off by --no-unzip and --no-shard.
""")):
  parser.add_argument(
      '--outdir',
      type=str,
      default=None,
      required=True,
      help='path to the output dir',
  )
  attach_bool_arg(
      parser,
      'download',
      default=True,
      help_str='--download is set by default. To skip download, explicitly set '
      '--no-download.',
  )
  attach_bool_arg(
      parser,
      'unzip',
      default=True,
      help_str='--unzip is set by default. To skip unzip, explicitly set '
      '--no-unzip.',
  )
  attach_bool_arg(
      parser,
      'shard',
      default=True,
      help_str='--shard is set by default. To skip shard, explicitly set '
      '--no-shard.',
  )
  parser.add_argument(
      '--num-shards',
      type=int,
      default=32,
      help='number of shards',
  )
  parser.add_argument(
      '--shard-num-processes',
      type=int,
      default=os.cpu_count(),
      help='num of processes used to shard OpenWebTextCorpus',
  )
  parser.add_argument(
      '--url',
      type=str,
      default='https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx',
      help='the google drive url of OpenWebTextCorpus',
  )
  return parser


def _shard_pages(shard):
  shard_path, pages = shard
  with open(shard_path, 'w', newline='\n') as shard_file:
    one_line_pages = []
    for page in pages:
      text_paths = [
          f for f in get_all_files_paths_under(page)
          if os.path.splitext(f)[1] == '.txt'
      ]
      page_lines = []
      for text in text_paths:
        with open(text, 'r', encoding='utf-8-sig', newline='\n') as page_file:
          sub_page_lines = (pg.strip() for pg in page_file)
          sub_page_lines = [pg for pg in sub_page_lines if len(pg) > 0]
          page_lines.extend(sub_page_lines)
      # The first token is the name of the page.
      page_name = os.path.splitext(os.path.basename(page))[0]
      one_line_pages.append(' '.join([page_name] + page_lines))
    shard_file.write('\n'.join(one_line_pages))


def unzip_subset(subset, text_dir):
  try:
    subdir_name = subset.split('.xz')[0].split('/')[-1]
    tmpdir_name = os.path.join('/tmp', subdir_name)
    subdir_name = os.path.join(text_dir, subdir_name)
    mkdir(subdir_name)
    mkdir(tmpdir_name)
    out_path = os.path.join(tmpdir_name, 'tar.out')
    err_path = os.path.join(tmpdir_name, 'tar.err')
    subprocess.run(
        ['tar', '-xvf', subset, '-C', subdir_name],
        check=True,
        stdout=open(out_path, 'w'),
        stderr=open(err_path, 'w'),
    )
    shutil.rmtree(tmpdir_name)
  except subprocess.CalledProcessError as e:
    print(e, 'Please check {} and {}'.format(out_path, err_path))
    raise


def unzip_merge_txt(openweb_dir, text_dir, num_processes):
  subset_paths = [
      f for f in get_all_files_paths_under(openweb_dir)
      if os.path.splitext(f)[1] == '.xz'
  ]
  with multiprocessing.Pool(num_processes) as p:
    list(
        tqdm.tqdm(p.map(functools.partial(unzip_subset, text_dir=text_dir),
                        subset_paths),
                  total=len(subset_paths)))


def _shard_openwebs(text_dir, shards_dir, num_shards, num_processes):
  dir_paths = [d for d in glob(text_dir + '/*')]
  shards = [(
      os.path.join(shards_dir, '{}.txt'.format(shard_idx)),
      dir_paths[shard_idx::num_shards],
  ) for shard_idx in range(num_shards)]
  with multiprocessing.Pool(num_processes) as p:
    list(tqdm.tqdm(p.imap(_shard_pages, shards), total=len(shards)))


def main(args):
  args.outdir = expand_outdir_and_mkdir(args.outdir)
  target_path = os.path.join(args.outdir, 'openwebtext.tar.xz')
  if args.download:
    gdown.download(args.url, target_path, quiet=False)
  if args.unzip:
    print('Unzipping {} ...'.format(target_path))
    out_path = os.path.join(args.outdir, 'tar.out')
    err_path = os.path.join(args.outdir, 'tar.err')
    try:
      subprocess.run(
          ['tar', '-xvf', target_path, '-C', args.outdir],
          check=True,
          stdout=open(out_path, 'w'),
          stderr=open(err_path, 'w'),
      )
    except subprocess.CalledProcessError as e:
      print(e, 'Please check {} and {}'.format(out_path, err_path))
      raise
    openweb_dir = os.path.join(args.outdir, 'openwebtext')
    text_dir = os.path.join(args.outdir, 'txt')
    mkdir(text_dir)
    unzip_merge_txt(openweb_dir, text_dir, args.shard_num_processes)

  if args.shard:
    text_dir = os.path.join(args.outdir, 'txt')
    print('Sharding {} ...'.format(text_dir))
    dask_source_path = os.path.join(args.outdir, 'source')
    mkdir(dask_source_path)
    _shard_openwebs(
        text_dir,
        dask_source_path,
        args.num_shards,
        args.shard_num_processes,
    )
    print('Dask source prepared at {} !'.format(dask_source_path))


def console_script():
  main(attach_args().parse_args())
