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
"""A model-based ENN agent."""

import dataclasses
import typing as tp

import chex
from enn import base
from enn.networks import base as networks_base
import haiku as hk
import jax
import jax.numpy as jnp


# Helpful aliases for network definition
_Enn = base.EpistemicNetwork[base.Input, base.Output]  # Generic ENN
OutputWithPrior = networks_base.OutputWithPrior
Input = chex.Array  # Input frame is an array
Hidden = chex.Array  # Hidden state is an array
HiddenA = chex.Array  # Hidden state including action representation


class ActionEmbedding(tp.Protocol):

  def __call__(self, hidden: Hidden, action: int) -> HiddenA:
    """Takes in a batch of hiddens and adds action to representation."""


class PredictionOutput(tp.NamedTuple):
  """Output of the prediction network."""
  reward: OutputWithPrior
  value: OutputWithPrior
  policy: OutputWithPrior  # Logits
  extra: tp.Dict[str, chex.Array] = {}


@dataclasses.dataclass
class AgentModel:
  """Agent's model of the environment."""
  representation: _Enn[Input, OutputWithPrior]  # Maps o_t, index --> h_t
  action_embedding: ActionEmbedding  # Maps h_t, a_t --> h_plus_a
  dynamics: _Enn[HiddenA, OutputWithPrior]  # Maps h_plus_a, index --> h_tp1
  prediction: _Enn[Hidden, PredictionOutput]  # Maps h_tp1, index --> (r, v, p)


class DepthOneSearchOutput(tp.NamedTuple):
  """The output of depth one search over actions."""
  root: PredictionOutput  # Output at the root node
  children: tp.Sequence[PredictionOutput]  # Outputs for each action


ApplyOneStep = base.ApplyFn[Input, DepthOneSearchOutput]
EnnOneStep = _Enn[Input, DepthOneSearchOutput]


def make_enn_onestep(model: AgentModel, num_actions: int) -> EnnOneStep:
  """Combine pieces of an AgentModel for feed-forward learning.

  WARNING: We assume that none of the hk.State is used. We use _ for the unused
  states returned by the apply and init functions. We assume that a single
  params contains *all* the params for all the network pieces. We also assume
  integer actions starting from 0.

  Args:
    model: an AgentModel that defines the network architectures.
    num_actions: number of actions to compute Q-values for (assumes 0-start).

  Returns:
    ENN that forwards input -> DepthOneSearchOutput as if it was feed-forward.
  """

  def apply(
      params: hk.Params,
      dummy_state: hk.State,
      inputs: Input,
      index: base.Index,
  ) -> tp.Tuple[DepthOneSearchOutput, hk.State]:

    # Form the hidden units from the representation network.
    h_t, _ = model.representation.apply(params, dummy_state, inputs, index)

    # Reward and value prediction at the root node
    root, _ = model.prediction.apply(params, dummy_state, h_t.preds, index)

    children = []
    for action in range(num_actions):
      # Incorporate action into representation
      h_plus_a = model.action_embedding(h_t.preds, action)

      # Forward the dynamics portion to make the next hidden state.
      h_tp1, _ = model.dynamics.apply(params, dummy_state, h_plus_a, index)

      # Forward the network predictions
      output, _ = model.prediction.apply(
          params, dummy_state, h_tp1.preds, index)

      children.append(output)

    return DepthOneSearchOutput(root, children), {}

  def init(
      key: chex.PRNGKey,
      inputs: Input,
      index: base.Index,
  ) -> tp.Tuple[hk.Params, hk.State]:
    # Splitting random keys
    repr_key, dynamic_key, pred_key = jax.random.split(key, 3)

    # Dummy network state
    dummy_state = {}

    # Representation network
    repr_params, _ = model.representation.init(repr_key, inputs, index)

    # Dynamics network
    h_t, _ = model.representation.apply(repr_params, dummy_state, inputs, index)
    # Assumes integer actions and selects action=0
    h_plus_a = model.action_embedding(h_t.preds, 0)
    dynamic_params, _ = model.dynamics.init(dynamic_key, h_plus_a, index)

    # Prediction network
    h_tp1, _ = model.dynamics.apply(
        dynamic_params, dummy_state, h_plus_a, index)
    pred_params, _ = model.prediction.init(pred_key, h_tp1.preds, index)

    return {**repr_params, **dynamic_params, **pred_params}, {}

  return EnnOneStep(apply, init, model.prediction.indexer)


def stack_action_rewards(net_out: DepthOneSearchOutput) -> chex.Array:
  """Returns a jnp array of reward predictions of shape (batch, actions)."""
  stacked_out = jax.tree_util.tree_map(
      lambda *args: jnp.stack([*args], axis=1), *net_out.children)
  rewards = jnp.squeeze(stacked_out.reward.preds, axis=-1)
  batch_size = net_out.root.reward.preds.shape[0]
  num_actions = len(net_out.children)
  chex.assert_shape(rewards, [batch_size, num_actions])
  return rewards


def stack_action_values(net_out: DepthOneSearchOutput) -> chex.Array:
  """Returns a jnp array of value predictions of shape (batch, actions)."""
  stacked_out = jax.tree_util.tree_map(
      lambda *args: jnp.stack([*args], axis=1), *net_out.children)
  values = jnp.squeeze(stacked_out.value.preds, axis=-1)
  batch_size = net_out.root.value.preds.shape[0]
  num_actions = len(net_out.children)
  chex.assert_shape(values, [batch_size, num_actions])
  return values
