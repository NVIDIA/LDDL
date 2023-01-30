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
import paddle
from paddle.fluid.framework import _non_static_mode
from paddle.distributed.fleet.base.private_helper_function import wait_server_ready


def get_rank():
  return int(os.getenv("PADDLE_TRAINER_ID", "0"))


def get_local_rank():
  return int(os.getenv('PADDLE_RANK_IN_NODE', '0'))


def get_world_size():
  return int(os.getenv('PADDLE_TRAINERS_NUM', '1'))


def barrier():
  if get_world_size() > 1:
    paddle.distributed.barrier()


def get_endpoints():
  endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS')
  return endpoints.split(",")


def get_current_endpoint():
  return os.getenv("PADDLE_CURRENT_ENDPOINT")


def get_other_endpoints():
  other_endpoints = get_endpoints()[:]
  current_endpoint = get_current_endpoint()
  other_endpoints.remove(current_endpoint)
  return other_endpoints


def get_num_nodes():
  # paddle_local_size = int(os.getenv('PADDLE_LOCAL_SIZE', '-1'))
  endpoints = get_endpoints()[:]
  ips = set()
  for endpoint in endpoints:
    ip = endpoint.split(":")[0]
    ips.add(ip)
  return len(ips)


def get_nproc_per_node():
  return get_world_size() // get_num_nodes()


def get_node_rank():
  """ This assume the training processes are launched via
  paddle.distributed.launch.py. Therefore, the ordering scheme of
  rank             -> (node_rank, local_rank) mapping is:
  0                -> (0, 0)
  1                -> (0, 1)
  ...
  nproc_per_node   -> (1, 0)
  nproc_per_node+1 -> (1, 1)
  ...
  """
  nproc_per_node = get_nproc_per_node()
  node_rank = get_rank() // nproc_per_node
  return node_rank


def all_reduce_in_static_mode(local_tensor, reduce_op):
  assert not _non_static_mode(), "this function can only be used in static mode"
  rank = get_rank()
  local_rank = get_local_rank()
  nranks = get_world_size()
  current_endpoint = get_current_endpoint()
  other_endpoints = get_other_endpoints()
  device = paddle.set_device("gpu")
  if rank == 0:
    wait_server_ready(other_endpoints)

  startup_program = paddle.static.Program()
  main_program = paddle.static.Program()
  exe = paddle.static.Executor(device)

  block = startup_program.global_block()
  nccl_id_var = block.create_var(
      name=paddle.fluid.unique_name.generate('nccl_id'),
      persistable=True,
      type=paddle.fluid.core.VarDesc.VarType.RAW,
  )

  block.append_op(
      type='c_gen_nccl_id',
      inputs={},
      outputs={'Out': nccl_id_var},
      attrs={
          'rank': rank,
          'endpoint': current_endpoint,
          'other_endpoints': other_endpoints,
      },
  )

  block.append_op(
      type='c_comm_init',
      inputs={'X': nccl_id_var},
      outputs={},
      attrs={
          'nranks': nranks,
          'rank': rank,
          'ring_id': 0
      },
  )

  with paddle.static.program_guard(main_program, startup_program):
    data = paddle.static.data(name='local_value', shape=[-1], dtype='int64')
    paddle.distributed.all_reduce(data, op=reduce_op)

  exe.run(startup_program)
  results = exe.run(main_program,
                    feed={'local_value': local_tensor},
                    fetch_list=[data.name])
  return results[0]
