# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A LossFn computes the loss on a batch of data for one index."""
import abc
from typing import Tuple, Union

from absl import logging
import chex
from enn import base as enn_base
from enn import networks
from enn import utils
from enn_acme import base as agent_base
import haiku as hk
import jax
import jax.numpy as jnp
import reverb


class SingleIndexLossFn(abc.ABC):
  """A SingleIndexLossFn defines how to process one batch of data, one index."""

  @abc.abstractmethod
  def __call__(
      self,
      apply: networks.ApplyNoState,
      params: hk.Params,
      state: agent_base.LearnerState,
      batch: reverb.ReplaySample,
      index: enn_base.Index,
  ) -> Tuple[chex.Array, agent_base.LossMetrics]:
    """Compute the loss on a single batch of data, for one index."""


def average_single_index_loss(single_loss: SingleIndexLossFn,
                              num_index_samples: int = 1) -> agent_base.LossFn:
  """Average a single index loss over multiple index samples."""

  def loss_fn(enn: networks.EnnNoState,
              params: hk.Params,
              state: agent_base.LearnerState,
              batch: reverb.ReplaySample,
              key: chex.PRNGKey) -> chex.Array:
    batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_samples)
    batched_loss = jax.vmap(single_loss, in_axes=[None, None, None, None, 0])
    loss, metrics = batched_loss(
        enn.apply, params, state, batch, batched_indexer(key))
    return jnp.mean(loss), jax.tree_map(jnp.mean, metrics)

  return loss_fn


def parse_loss_fn(
    loss_fn: Union[agent_base.LossFn, SingleIndexLossFn]) -> agent_base.LossFn:
  if isinstance(loss_fn, SingleIndexLossFn):
    logging.warn(
        'WARNING: coercing single_index_loss to LossFn with 1 random sample.')
    loss_fn = average_single_index_loss(loss_fn, num_index_samples=1)
  return loss_fn

