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

import random
import torch

from lddl.random import choices
from .datasets import ParquetDataset


class Binned:

  def __init__(self, dataloaders, base_seed=12345, start_epoch=0, logger=None):
    self._dataloaders = dataloaders

    self._base_seed = base_seed
    self._epoch = start_epoch - 1

    self._logger = logger

    self._world_rng_state = None

  def _init_rng_states(self):
    orig_rng_state = random.getstate()

    random.seed(self._base_seed + self._epoch)
    self._world_rng_state = random.getstate()

    random.setstate(orig_rng_state)

  def _init_iter(self):
    self._init_rng_states()
    num_samples_remaining = [len(dl.dataset) for dl in self._dataloaders]
    dataiters = [iter(dl) for dl in self._dataloaders]
    return num_samples_remaining, dataiters

  def __len__(self):
    return sum((len(dl) for dl in self._dataloaders))

  def _get_batch_size(self, batch):
    raise NotImplementedError('Binned is an abstract class!')

  def _choices(self, population, weights=None, cum_weights=None, k=1):
    c, self._world_rng_state = choices(
        population,
        weights=weights,
        cum_weights=cum_weights,
        k=k,
        rng_state=self._world_rng_state,
    )
    return c

  def __iter__(self):
    self._epoch += 1
    num_samples_remaining, dataiters = self._init_iter()

    for i in range(len(self)):
      bin_id = self._choices(
          list(range(len(dataiters))),
          weights=num_samples_remaining,
          k=1,
      )[0]
      self._logger.to('rank').info('{}-th iteration selects bin_id = {}'.format(
          i, bin_id))
      assert num_samples_remaining[bin_id] > 0
      batch = next(dataiters[bin_id])
      num_samples_remaining[bin_id] -= self._get_batch_size(batch)
      yield batch

    assert sum((nsr for nsr in num_samples_remaining)) == 0


class DataLoader(torch.utils.data.DataLoader):

  def __len__(self):
    if isinstance(self.dataset, ParquetDataset):
      num_workers_per_rank = max(self.num_workers, 1)
      num_files_per_worker = self.dataset.num_files_per_rank // num_workers_per_rank
      num_samples_per_worker = self.dataset.num_samples_per_file * num_files_per_worker
      num_batches_per_worker = (
          (num_samples_per_worker - 1) // self.batch_size + 1)
      return num_batches_per_worker * num_workers_per_rank
    else:
      super().__len__()
