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
"""Q learning variants for loss function definition."""
import dataclasses
from typing import Tuple

from acme import types
import chex
from enn import base as enn_base
from enn import losses
from enn import networks
from enn_acme import base as agent_base
from enn_acme.losses import single_index
import haiku as hk
import jax
import jax.numpy as jnp
import reverb
import rlax


@dataclasses.dataclass
class ClippedQlearning(single_index.SingleLossFnArray):
  """Clipped Q learning."""
  discount: float
  max_abs_reward: float = 1

  def __call__(
      self,
      apply: networks.ApplyArray,
      params: hk.Params,
      state: agent_base.LearnerState,
      batch: reverb.ReplaySample,
      index: enn_base.Index,
  ) -> Tuple[chex.Array, agent_base.LossMetrics]:
    """Evaluate loss for one batch, for one single index."""
    transitions: types.Transition = batch.data
    dummy_network_state = {}
    net_out_tm1, unused_state = apply(params, dummy_network_state,
                                      transitions.observation, index)
    q_tm1 = networks.parse_net_output(net_out_tm1)
    net_out_t, unused_state = apply(state.target_params, dummy_network_state,
                                    transitions.next_observation, index)
    q_t = networks.parse_net_output(net_out_t)

    d_t = (transitions.discount * self.discount).astype(jnp.float32)
    r_t = jnp.clip(transitions.reward, -self.max_abs_reward,
                   self.max_abs_reward).astype(jnp.float32)
    td_errors = jax.vmap(rlax.q_learning)(
        q_tm1, transitions.action, r_t, d_t, q_t)

    return jnp.mean(jnp.square(td_errors)), {}


@dataclasses.dataclass
class Categorical2HotQlearning(single_index.SingleLossFnArray):
  """Q learning applied with cross-entropy loss to two-hot targets."""
  discount: float

  def __call__(
      self,
      apply: networks.ApplyArray,
      params: hk.Params,
      state: agent_base.LearnerState,
      batch: reverb.ReplaySample,
      index: enn_base.Index,
  ) -> Tuple[chex.Array, agent_base.LossMetrics]:
    """Evaluate loss for one batch, for one single index."""
    transitions: types.Transition = batch.data
    batch_idx = jnp.arange(transitions.observation.shape[0])

    dummy_network_state = {}
    net_out_tm1, unused_state = apply(params, dummy_network_state,
                                      transitions.observation, index)
    net_out_t, unused_state = apply(state.target_params, dummy_network_state,
                                    transitions.next_observation, index)

    # Check that network output has the right type
    assert isinstance(net_out_tm1, networks.CatOutputWithPrior)
    assert isinstance(net_out_t, networks.CatOutputWithPrior)

    # Form target values in real space
    d_t = (transitions.discount * self.discount).astype(jnp.float32)
    v_t = jnp.max(net_out_t.preds, axis=-1)
    target_val = transitions.reward + d_t * v_t - net_out_tm1.prior[
        batch_idx, transitions.action]

    # Convert values to 2-hot target probabilities
    logits = net_out_tm1.train[batch_idx, transitions.action, :]
    target_probs = jax.vmap(losses.transform_to_2hot, in_axes=[0, None])(
        target_val, net_out_tm1.extra['atoms'])
    xent_loss = -jnp.sum(target_probs * jax.nn.log_softmax(logits), axis=-1)

    return jnp.mean(xent_loss), {}
