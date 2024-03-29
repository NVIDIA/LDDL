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

import logging
import os
import pathlib
from .utils import (get_local_rank, get_node_rank)


def _get_logger_name(node_rank, local_rank=None, worker_rank=None):
  if local_rank is None and worker_rank is None:
    return 'node-{}'.format(node_rank)
  elif worker_rank is None:
    return 'node-{}_local-{}'.format(node_rank, local_rank)
  else:
    return 'node-{}_local-{}_worker-{}'.format(node_rank, local_rank,
                                               worker_rank)


class DummyLogger:

  def debug(self, msg, *args, **kwargs):
    pass

  def info(self, msg, *args, **kwargs):
    pass

  def warning(self, msg, *args, **kwargs):
    pass

  def error(self, msg, *args, **kwargs):
    pass

  def critical(self, msg, *args, **kwargs):
    pass

  def log(self, msg, *args, **kwargs):
    pass

  def exception(self, msg, *args, **kwargs):
    pass


class DatasetLogger:

  def __init__(
      self,
      log_dir=None,
      log_level=logging.INFO,
  ):
    self._log_dir = log_dir
    self._node_rank = get_node_rank()
    self._local_rank = get_local_rank()
    self._worker_rank = None
    self._log_level = log_level

    if log_dir is not None:
      pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    # Create node level logger.
    if self._local_rank == 0:
      self._create_logger(_get_logger_name(self._node_rank))
    # Create local_rank level logger.
    self._create_logger(
        _get_logger_name(self._node_rank, local_rank=self._local_rank))

  def _create_logger(self, name):
    logger = logging.getLogger(name)
    fmt = logging.Formatter(
        'LDDL - %(asctime)s - %(filename)s:%(lineno)d:%(funcName)s - %(name)s '
        '- %(levelname)s : %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    if self._log_dir is not None:
      path = os.path.join(self._log_dir, '{}.txt'.format(name))
      file_handler = logging.FileHandler(path)
      file_handler.setFormatter(fmt)
      logger.addHandler(file_handler)
    logger.setLevel(self._log_level)
    return logger

  def init_for_worker(self, worker_rank):
    if self._worker_rank is None:
      self._worker_rank = worker_rank
      self._create_logger(
          _get_logger_name(
              self._node_rank,
              local_rank=self._local_rank,
              worker_rank=worker_rank,
          ))

  def to(self, which):
    assert which in {'node', 'rank', 'worker'}
    if which == 'node':
      if (self._local_rank == 0 and
          (self._worker_rank is None or self._worker_rank == 0)):
        return logging.getLogger(_get_logger_name(self._node_rank))
      else:
        return DummyLogger()
    elif which == 'rank':
      if self._worker_rank is None or self._worker_rank == 0:
        return logging.getLogger(
            _get_logger_name(self._node_rank, local_rank=self._local_rank))
      else:
        return DummyLogger()
    else:  # which == 'worker'
      return logging.getLogger(
          _get_logger_name(
              self._node_rank,
              local_rank=self._local_rank,
              worker_rank=self._worker_rank,
          ))
