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
"""An EnnPlanner that selects actions based on Thompson sampling."""
from acme.jax import utils
import dm_env
from enn import base as enn_base
from enn import utils as enn_utils
from enn_acme import base as agent_base
import haiku as hk
import jax
import rlax


class ThompsonQPlanner(agent_base.EnnPlanner):
  """A planner that performs Thompson sampling planning based on Q values."""

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               seed: int = 0,
               epsilon: float = 0.):
    self.enn = enn
    self.rng = hk.PRNGSequence(seed)
    self.index = enn.indexer(next(self.rng))

    def sample_index(key: enn_base.RngKey) -> enn_base.Index:
      return self.enn.indexer(key)
    self._sample_index = jax.jit(sample_index)

    def batched_egreedy(params: hk.Params,
                        observation: enn_base.Array,
                        index: enn_base.Index,
                        key: enn_base.RngKey) -> agent_base.Action:
      observation = utils.add_batch_dim(observation)
      net_out = self.enn.apply(params, observation, index)
      action_values = enn_utils.parse_net_output(net_out)
      return rlax.epsilon_greedy(epsilon).sample(key, action_values)
    self._batched_egreedy = jax.jit(batched_egreedy)

  def select_action(self,
                    params: hk.Params,
                    observation: enn_base.Array) -> agent_base.Action:
    """Selects an action given params and observation."""
    action = self._batched_egreedy(
        params, observation, self.index, next(self.rng))
    return utils.to_numpy_squeeze(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    """Resample an epistemic index at the start of the episode."""
    self.index = self._sample_index(next(self.rng))
