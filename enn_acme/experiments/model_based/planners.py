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
"""Planners for the model-based agent."""

from acme.jax import utils
import chex
import distrax
import dm_env
from enn import base as enn_base
from enn_acme import base as agent_base
from enn_acme.experiments.model_based import base
import haiku as hk
import jax
import jax.numpy as jnp


class ThompsonQPlanner(
    agent_base.EnnPlanner[base.Input, base.DepthOneSearchOutput]):
  """A planner that acts greedily according to sampled q-values."""

  def __init__(
      self,
      enn: base.EnnOneStep,
      seed: int = 0,
      epsilon: float = 0.,
      discount: float = 0.99,
  ):
    self.enn = enn
    self.rng = hk.PRNGSequence(seed)
    self.index = enn.indexer(next(self.rng))

    def sample_index(key: chex.PRNGKey) -> enn_base.Index:
      return self.enn.indexer(key)
    self._sample_index = jax.jit(sample_index)

    def select_greedy(
        params: hk.Params,
        observation: chex.Array,
        index: enn_base.Index,
        key: chex.PRNGKey,
    ) -> agent_base.Action:
      observation = utils.add_batch_dim(observation)
      net_out, _ = self.enn.apply(params, {}, observation, index)
      rewards = base.stack_action_rewards(net_out)
      values = base.stack_action_values(net_out)
      q_values = jnp.squeeze(rewards + discount * values, axis=0)
      chex.assert_rank(q_values, 1)
      return distrax.EpsilonGreedy(q_values, epsilon).sample(seed=key)

    self._select_greedy = jax.jit(select_greedy)

  def select_action(
      self, params: hk.Params, observation: chex.Array) -> agent_base.Action:
    """Selects an action given params and observation."""
    return self._select_greedy(params, observation, self.index, next(self.rng))

  def observe_first(self, timestep: dm_env.TimeStep):
    """Resample an epistemic index at the start of the episode."""
    self.index = self._sample_index(next(self.rng))


class ThompsonPolicyPlanner(
    agent_base.EnnPlanner[base.Input, base.DepthOneSearchOutput]):
  """A planner that acts according to the policy head."""

  def __init__(
      self,
      enn: base.EnnOneStep,
      seed: int = 0,
      epsilon: float = 0.,
  ):
    self.enn = enn
    self.rng = hk.PRNGSequence(seed)
    self.index = enn.indexer(next(self.rng))

    def sample_index(key: chex.PRNGKey) -> enn_base.Index:
      return self.enn.indexer(key)
    self._sample_index = jax.jit(sample_index)

    def sample_action(
        params: hk.Params,
        observation: chex.Array,
        index: enn_base.Index,
        key: chex.PRNGKey,
    ) -> agent_base.Action:
      observation = utils.add_batch_dim(observation)
      net_out, _ = self.enn.apply(params, {}, observation, index)
      logits = jnp.squeeze(net_out.root.policy.preds)
      chex.assert_rank(logits, 1)
      policy_key, egreedy_key = jax.random.split(key)
      policy_action_sample = distrax.Categorical(logits=logits).sample(
          seed=policy_key)
      one_hot = jax.nn.one_hot(policy_action_sample, logits.shape[0])
      return distrax.EpsilonGreedy(one_hot, epsilon).sample(seed=egreedy_key)

    self._sample_action = jax.jit(sample_action)

  def select_action(
      self, params: hk.Params, observation: chex.Array) -> agent_base.Action:
    """Selects an action given params and observation."""
    return self._sample_action(params, observation, self.index, next(self.rng))

  def observe_first(self, timestep: dm_env.TimeStep):
    """Resample an epistemic index at the start of the episode."""
    self.index = self._sample_index(next(self.rng))
