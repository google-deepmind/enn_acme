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
"""An EnnPlanner that selects actions based on Sample-based IDS."""

from typing import Optional, Sequence

from acme import specs
from acme.jax import utils
import chex
from enn import base_legacy as enn_base
from enn import networks
from enn import utils as enn_utils
from enn_acme import base as agent_base
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import typing_extensions


class InformationCalculator(typing_extensions.Protocol):

  def __call__(self,
               params: hk.Params,
               observation: enn_base.Array,
               key: enn_base.RngKey) -> enn_base.Array:
    """Estimates information gain for each action."""


class RegretCalculator(typing_extensions.Protocol):

  def __call__(self,
               params: hk.Params,
               observation: enn_base.Array,
               key: enn_base.RngKey) -> enn_base.Array:
    """Estimates regret for each action."""


class InformationRatioOptimizer(typing_extensions.Protocol):

  def __call__(self,
               regret: enn_base.Array,
               information: enn_base.Array,
               key: enn_base.RngKey) -> enn_base.Array:
    """Returns the probability distribution that minimizes information ratio."""


class IdsPlanner(agent_base.EnnPlanner):
  """A planner that performs IDS based on enn outputs."""

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               environment_spec: specs.EnvironmentSpec,
               information_calculator: InformationCalculator,
               regret_calculator: RegretCalculator,
               info_ratio_optimizer: InformationRatioOptimizer,
               seed: int = 0):
    self.enn = enn
    self.num_action = environment_spec.actions.num_values
    self.information_calculator = information_calculator
    self.regret_calculator = regret_calculator
    self.info_ratio_optimizer = info_ratio_optimizer
    self.rng = hk.PRNGSequence(seed)

  def select_action(self,
                    params: hk.Params,
                    observation: enn_base.Array) -> agent_base.Action:
    """Selects an action given params and observation."""
    regret = self.regret_calculator(params, observation, next(self.rng))
    information = self.information_calculator(params, observation,
                                              next(self.rng))
    probs = self.info_ratio_optimizer(regret, information, next(self.rng))
    action = jax.random.choice(next(self.rng), self.num_action, p=probs)
    return utils.to_numpy_squeeze(action)


class DiscreteInformatioRatioOptimizer(InformationRatioOptimizer):
  """Search over pairs of actions to minimize the information ratio."""

  def __init__(self,
               probability_discretize: int = 101):
    super().__init__()

    def optimize(regret: enn_base.Array,
                 information: enn_base.Array,
                 key: enn_base.RngKey) -> enn_base.Array:
      """Returns probability distribution that minimizes information ratio."""

      num_action = len(regret)
      chex.assert_shape(regret, (num_action,))
      chex.assert_shape(information, (num_action,))

      # make discrete probability candidates over pairs of actions. Note that
      # this computation is only done during tracing, i.e.,
      # probability_canditates will not be recomputed unless the shape of regret
      # or information change.
      probability_candidates = []
      for a1 in range(num_action):
        for a2 in range(a1 + 1, num_action):
          for p in np.linspace(0, 1, probability_discretize):
            candidate_probability = np.zeros(num_action)
            candidate_probability[a1] = 1 - p
            candidate_probability[a2] = p
            probability_candidates.append(candidate_probability)
      probability_candidates = jnp.array(probability_candidates)
      num_probability_candidates = len(probability_candidates)

      expected_regret_sq = jnp.dot(probability_candidates, regret) ** 2
      chex.assert_shape(expected_regret_sq, (num_probability_candidates,))

      expected_information = jnp.dot(probability_candidates, information)
      chex.assert_shape(expected_information, (num_probability_candidates,))

      information_ratio = expected_regret_sq / expected_information
      chex.assert_shape(information_ratio, (num_probability_candidates,))

      # compute argmin and break ties randomly
      index = jnp.argmin(information_ratio + jax.random.uniform(
          key, (num_probability_candidates,), maxval=1e-9))
      return probability_candidates[index]
    self._optimize = jax.jit(optimize)

  def __call__(self,
               regret: enn_base.Array,
               information: enn_base.Array,
               key: enn_base.RngKey) -> enn_base.Array:
    return self._optimize(regret, information, key)


class RegretWithPessimism(RegretCalculator):
  """Sample based average regret with pessimism."""

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               num_sample: int = 100,
               pessimism: float = 0.,):
    super().__init__()
    forward = jax.jit(make_batched_forward(enn=enn, batch_size=num_sample))

    def sample_based_regret(
        params: hk.Params,
        observation: enn_base.Array,
        key: enn_base.RngKey) -> enn_base.Array:
      """Estimates regret for each action."""
      batched_out = forward(params, observation, key)
      # TODO(author4): Sort out the need for squeeze/batch more clearly.
      batched_q = jnp.squeeze(networks.parse_net_output(batched_out))
      assert (batched_q.ndim == 2) and (batched_q.shape[0] == num_sample)
      sample_regret = jnp.max(batched_q, axis=1, keepdims=True) - batched_q
      return jnp.mean(sample_regret, axis=0) + pessimism
    self._sample_based_regret = jax.jit(sample_based_regret)

  def __call__(self,
               params: hk.Params,
               observation: enn_base.Array,
               key: enn_base.RngKey) -> enn_base.Array:
    return self._sample_based_regret(params, observation, key)


class VarianceGVF(InformationCalculator):
  """Computes the variance of GVFs."""

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               num_sample: int = 100,
               ridge_factor: float = 1e-6,
               exclude_keys: Optional[Sequence[str]] = None,
               jit: bool = False):
    super().__init__()
    forward = make_batched_forward(enn=enn, batch_size=num_sample)
    self._forward = jax.jit(forward)
    self.num_sample = num_sample
    self.ridge_factor = ridge_factor
    self.exclude_keys = exclude_keys or []

    def compute_variance(params: hk.Params,
                         observation: enn_base.Array,
                         key: enn_base.RngKey) -> enn_base.Array:
      batched_out = self._forward(params, observation, key)
      # TODO(author2): Forces network to fit the OutputWithPrior format.
      assert isinstance(batched_out, enn_base.OutputWithPrior)

      # TODO(author4): Sort out the need for squeeze/batch more clearly.
      batched_q = jnp.squeeze(networks.parse_net_output(batched_out))
      assert (batched_q.ndim == 2) and (batched_q.shape[0] == self.num_sample)
      total_variance = jnp.var(batched_q, axis=0)

      # GVF predictions should live in the .extra component.
      for gvf_key, batched_gvf in batched_out.extra.items():
        if gvf_key not in self.exclude_keys:
          # TODO(author4): Sort out a standard way of structuring gvf outputs.
          batched_gvf = jnp.squeeze(batched_gvf)
          assert (batched_gvf.ndim == 2) or (batched_gvf.ndim == 3)
          assert batched_gvf.shape[0] == self.num_sample
          key_variance = jnp.var(batched_gvf, axis=0)
          if key_variance.ndim == 2:  # (A, gvf_dim)
            key_variance = jnp.sum(key_variance, axis=1)
          assert key_variance.shape == total_variance.shape
          total_variance += key_variance

      total_variance += self.ridge_factor
      return total_variance

    # TODO(author4): Check/test whether we can jit this function in general.
    if jit:
      self._compute_variance = jax.jit(compute_variance)
    else:
      self._compute_variance = compute_variance

  def __call__(self,
               params: hk.Params,
               observation: enn_base.Array,
               key: enn_base.RngKey) -> enn_base.Array:
    """Estimates information gain for each action."""
    return self._compute_variance(params, observation, key)


class VarianceOptimalAction(InformationCalculator):
  """Computes the variance of conditional expectation of Q conditioned on A*."""

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               num_sample: int = 100,
               ridge_factor: float = 1e-6,):
    super().__init__()
    forward = make_batched_forward(enn=enn, batch_size=num_sample)
    self._forward = jax.jit(forward)
    self.num_sample = num_sample
    self.ridge_factor = ridge_factor

  def __call__(self,
               params: hk.Params,
               observation: enn_base.Array,
               key: enn_base.RngKey) -> enn_base.Array:
    """Estimates information gain for each action."""
    # TODO(author4): Note this cannot be jax.jit in current form.
    # TODO(author4): This implementation does not allow for GVF yet!
    # TODO(author4): Sort out the need for squeeze/batch more clearly.

    batched_out = self._forward(params, observation, key)
    batched_q = np.squeeze(networks.parse_net_output(batched_out))
    assert (batched_q.ndim == 2) and (batched_q.shape[0] == self.num_sample)

    return compute_var_cond_mean(batched_q) + self.ridge_factor


def compute_var_cond_mean(q_samples: enn_base.Array) -> enn_base.Array:
  """Computes the variance of conditional means given a set of q samples."""

  num_action = q_samples.shape[1]
  # Currently use pandas to get a clear implementation.
  # qdf is the dataframe version of q_samples with num_sample rows and
  # num_action columns labeled by intergers 0, 1, ..., num_action - 1.
  qdf = pd.DataFrame(np.asarray(q_samples), columns=range(num_action))
  qdf_mean = qdf.mean()  # series with length num_action
  # Add an optimal action column.
  qdf['optimal_action'] = qdf.apply(lambda x: x.argmax(), axis=1)
  # Estimated probability of each action being optimal.
  # Series of len optimal action.
  opt_action_prob = qdf.optimal_action.value_counts(normalize=True, sort=False)
  # conditional means of shape: (num potentially actions, num_action)
  qdf_cond_mean = qdf.groupby('optimal_action').mean()
  # Variance of the conditional means. Series of len num_action.
  qdf_var_cond_mean = (
      qdf_cond_mean.apply(lambda x: x - qdf_mean, axis=1)**2
      ).apply(lambda x: np.sum(x * opt_action_prob), axis=0)

  return qdf_var_cond_mean.sort_index().to_numpy()


def make_batched_forward(enn: enn_base.EpistemicNetwork, batch_size: int):
  def forward(params: hk.Params,
              observation: enn_base.Array,
              key: enn_base.RngKey) -> enn_base.Output:
    """Fast/efficient implementation of batched forward in Jax."""
    batched_indexer = enn_utils.make_batch_indexer(enn.indexer, batch_size)
    batched_forward = jax.vmap(enn.apply, in_axes=[None, None, 0])
    observation = utils.add_batch_dim(observation)
    return batched_forward(params, observation, batched_indexer(key))
  return forward


def make_default_variance_ids_planner(
    enn: enn_base.EpistemicNetwork,
    environment_spec: specs.EnvironmentSpec,
    seed: int = 0,
    jit: bool = False) -> IdsPlanner:
  return IdsPlanner(
      enn=enn,
      environment_spec=environment_spec,
      information_calculator=VarianceGVF(enn=enn, jit=jit),
      regret_calculator=RegretWithPessimism(enn=enn),
      info_ratio_optimizer=DiscreteInformatioRatioOptimizer(),
      seed=seed)
