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
"""MLP-based models."""

import typing as tp

import chex
from enn import base as enn_base
from enn_acme.experiments.model_based import base
import haiku as hk
import jax
import jax.numpy as jnp


def make_outer_one_hot_embedding(num_actions: int) -> base.ActionEmbedding:
  """Make an outer product action embedding function.

  Args:
    num_actions: Number of actions.
  Returns:
    An action embedding function that performs outer product on a batch of
    hidden representations with a one-hot vector made from an integer action.
  """

  def single_embed(single_hidden: chex.Array, action: int) -> chex.Array:
    one_hot = jax.nn.one_hot(action, num_actions)
    return jnp.outer(single_hidden, one_hot).ravel()

  return jax.vmap(single_embed, in_axes=[0, None])


def make_single_model(
    action_embedding: base.ActionEmbedding,
    hidden_dim: int,
    num_actions: int,
    repr_hiddens: tp.Sequence[int] = (),
    dynamics_hiddens: tp.Sequence[int] = (),
    pred_hiddens: tp.Sequence[int] = (),
    layernorm: bool = False,
    return_hidden_state: bool = False,
) -> base.AgentModel:
  """Creates a simple MLP-based AgentModel."""

  # Write the networks without any ENN nomenclature
  def repr_net(inputs: base.Input) -> base.OutputWithPrior:
    output_sizes = list(repr_hiddens) + [hidden_dim]
    net = hk.nets.MLP(output_sizes, name='representation')
    flat_input = hk.Flatten()(inputs)
    out = net(flat_input)
    if layernorm:
      out = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(out)
    return base.OutputWithPrior(out, prior=jnp.zeros_like(out))

  def dynamics_net(inputs: base.HiddenA) -> base.OutputWithPrior:
    output_sizes = list(dynamics_hiddens) + [hidden_dim]
    net = hk.nets.MLP(output_sizes, name='dynamics')
    out = net(inputs)
    if layernorm:
      out = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(out)
    return base.OutputWithPrior(out, prior=jnp.zeros_like(out))

  def pred_net(inputs: base.Hidden) -> base.PredictionOutput:
    hiddens = list(pred_hiddens)
    r_net = hk.nets.MLP(hiddens + [1], name='prediction_r')
    v_net = hk.nets.MLP(hiddens + [1], name='prediction_v')
    p_net = hk.nets.MLP(hiddens + [num_actions], name='prediction_p')
    reward = r_net(inputs)
    value = v_net(inputs)
    policy = p_net(inputs)
    extra = {'hidden_state': inputs} if return_hidden_state else {}
    make_output = lambda x: base.OutputWithPrior(x, prior=jnp.zeros_like(x))
    return base.PredictionOutput(
        reward=make_output(reward),
        value=make_output(value),
        policy=make_output(policy),
        extra=extra,
    )

  return base.AgentModel(
      representation=_wrap_net_fn_as_enn(repr_net),
      action_embedding=action_embedding,
      dynamics=_wrap_net_fn_as_enn(dynamics_net),
      prediction=_wrap_net_fn_as_enn(pred_net),
  )


_Input = tp.TypeVar('_Input')
_Output = tp.TypeVar('_Output')


# TODO(author4): Integrate with enn networks utils.
def _wrap_net_fn_as_enn(
    net_fn: tp.Callable[[_Input], _Output],  # Pre-transformed
) -> enn_base.EpistemicNetwork[_Input, _Output]:
  """Wrap pre-transformed functions as ENNs with dummy index."""
  transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))

  def apply(params: hk.Params,
            state: hk.State,
            inputs: _Input,
            index: enn_base.Index) -> tp.Tuple[_Output, hk.State]:
    del index
    return transformed.apply(params, state, inputs)

  return enn_base.EpistemicNetwork[_Input, _Output](
      apply=apply,
      init=lambda k, x, z: transformed.init(k, x),
      indexer=lambda k: k,
  )
