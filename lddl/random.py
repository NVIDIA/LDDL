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


def _swap_rng_state(new_state):
  old_state = random.getstate()
  random.setstate(new_state)
  return old_state


def randrange(stop, rng_state=None):
  orig_rng_state = _swap_rng_state(rng_state)
  n = random.randrange(stop)
  return n, _swap_rng_state(orig_rng_state)


def shuffle(x, rng_state=None):
  orig_rng_state = _swap_rng_state(rng_state)
  random.shuffle(x)
  return _swap_rng_state(orig_rng_state)


def sample(population, k, rng_state=None):
  orig_rng_state = _swap_rng_state(rng_state)
  s = random.sample(population, k)
  return s, _swap_rng_state(orig_rng_state)


def choices(population, weights=None, cum_weights=None, k=1, rng_state=None):
  orig_rng_state = _swap_rng_state(rng_state)
  c = random.choices(population, weights=weights, cum_weights=cum_weights, k=k)
  return c, _swap_rng_state(orig_rng_state)
