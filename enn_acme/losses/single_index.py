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
from typing import Tuple

import chex
from enn import base as enn_base
from enn import networks as enn_networks
from enn import utils
from enn_acme import base as agent_base
import haiku as hk
import jax
import jax.numpy as jnp
import reverb
import typing_extensions as te


# Simple alises for generic modules
_ENN = enn_base.EpistemicNetwork[agent_base.Input, agent_base.Output]


class SingleLossFn(te.Protocol[agent_base.Input, agent_base.Output]):
  """A SingleLossFn defines how to process one batch of data, one index."""

  def __call__(
      self,
      apply: enn_base.ApplyFn[agent_base.Input, agent_base.Output],
      params: hk.Params,
      state: agent_base.LearnerState,
      batch: reverb.ReplaySample,
      index: enn_base.Index,
  ) -> Tuple[chex.Array, agent_base.LossMetrics]:
    """Compute the loss on a single batch of data, for one index."""


def average_single_index_loss(
    single_loss: SingleLossFn[agent_base.Input, agent_base.Output],
    num_index_samples: int = 1
) -> agent_base.LossFn[agent_base.Input, agent_base.Output]:
  """Average a single index loss over multiple index samples."""

  def loss_fn(enn: _ENN[agent_base.Input, agent_base.Output],
              params: hk.Params, state: agent_base.LearnerState,
              batch: reverb.ReplaySample, key: chex.PRNGKey) -> chex.Array:
    batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_samples)
    batched_loss = jax.vmap(single_loss, in_axes=[None, None, None, None, 0])
    loss, metrics = batched_loss(
        enn.apply, params, state, batch, batched_indexer(key))
    return jnp.mean(loss), jax.tree_util.tree_map(jnp.mean, metrics)

  return loss_fn


# Loss modules specialized to work only with Array inputs.
LossFnArray = agent_base.LossFn[chex.Array, enn_networks.Output]
SingleLossFnArray = SingleLossFn[chex.Array, enn_networks.Output]
