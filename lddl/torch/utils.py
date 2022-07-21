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

import torch


def barrier():
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.barrier()


def get_rank():
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    rank = torch.distributed.get_rank()
  else:
    rank = 0
  return rank


def get_world_size():
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    world_size = torch.distributed.get_world_size()
  else:
    world_size = 1
  return world_size


def get_nproc_per_node(local_rank):
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    max_local_rank = torch.tensor(
        local_rank,
        device='cuda' if torch.distributed.get_backend() == 'nccl' else 'cpu',
    )
    torch.distributed.all_reduce(
        max_local_rank,
        op=torch.distributed.ReduceOp.MAX,
    )
    nproc_per_node = max_local_rank.item() + 1
  else:
    nproc_per_node = 1
  return nproc_per_node


def get_num_nodes(local_rank=None, nproc_per_node=None):
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    if nproc_per_node is None:
      assert local_rank is not None
      nproc_per_node = get_nproc_per_node(local_rank)
    num_nodes = get_world_size() // nproc_per_node
  else:
    num_nodes = 1
  return num_nodes


def get_node_rank(local_rank=None, nproc_per_node=None):
  """ This assume the training processes are launched via
  torch.distributed.launch.py. Therefore, the ordering scheme of
  rank             -> (node_rank, local_rank) mapping is:
  0                -> (0, 0)
  1                -> (0, 1)
  ...
  nproc_per_node   -> (1, 0)
  nproc_per_node+1 -> (1, 1)
  ...
  """
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    if nproc_per_node is None:
      assert local_rank is not None
      nproc_per_node = get_nproc_per_node(local_rank)
    node_rank = get_rank() // nproc_per_node
  else:
    node_rank = 0
  return node_rank
