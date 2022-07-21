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
import requests
import tqdm


def download(url, path, chunk_size=16 * 1024 * 1024):
  with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total_size = int(r.headers.get('content-length', 0))
    progress_bar = tqdm.tqdm(total=total_size, unit='Bytes', unit_scale=True)
    with open(path, 'wb') as f:
      for chunk in r.iter_content(chunk_size=chunk_size):
        progress_bar.update(len(chunk))
        f.write(chunk)
    progress_bar.close()


def parse_str_of_num_bytes(s, return_str=False):
  try:
    power = 'kmg'.find(s[-1].lower()) + 1
    size = float(s[:-1]) * 1024**power
  except ValueError:
    raise ValueError('Invalid size: {}'.format(s))
  if return_str:
    return s
  else:
    return int(size)
