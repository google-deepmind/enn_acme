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
"""A random planner used for testing."""

from acme import specs
from acme.jax import utils
from enn import base_legacy as enn_base
from enn_acme import base as agent_base
import haiku as hk
import jax


class RandomPlanner(agent_base.EnnPlanner):
  """A planner selects actions randomly."""

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               environment_spec: specs.EnvironmentSpec,
               seed: int = 0):
    self.enn = enn
    self.num_actions = environment_spec.actions.num_values
    self.rng = hk.PRNGSequence(seed)

  def select_action(self,
                    params: hk.Params,
                    observation: enn_base.Array) -> agent_base.Action:
    """Selects an action given params and observation."""
    action = jax.random.choice(next(self.rng), self.num_actions)
    return utils.to_numpy_squeeze(action)
