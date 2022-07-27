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
"""Base classes for ENN agent design."""
import abc
import dataclasses
import typing as tp

from acme import types
import chex
import dm_env
from enn import base as enn_base
import haiku as hk
import optax
import reverb
import typing_extensions as te

Action = int
Input = tp.TypeVar('Input')  # Inputs to neural network
Output = tp.TypeVar('Output')  # Outputs of neural network
LossMetrics = enn_base.LossMetrics


@dataclasses.dataclass
class EnnPlanner(abc.ABC, tp.Generic[Input, Output]):
  """A planner can select actions based on an ENN knowledge representation.

  The EnnPlanner is one of the core units we often change in algorithm design.
  Examples might include Epsilon-Greedy, Thompson Sampling, IDS and more...
  We implement some examples in `planners.py`.

  The EnnPlanner is then used by an acme.Actor to select actions. However, we do
  not implement this as an acme.Actor since we want to hide the variable_client
  and choice of replay "adder" from action selection. For example of how this is
  used to make an acme.Actor see PlannerActor in agents/acting.py.
  """
  enn: enn_base.EpistemicNetwork[Input, Output]
  seed: tp.Optional[int] = 0

  @abc.abstractmethod
  def select_action(
      self, params: hk.Params, observation: Input) -> Action:
    """Selects an action given params and observation."""

  def observe_first(self, timestep: dm_env.TimeStep):
    """Optional: make a first observation from the environment."""

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    """Optional: make an observation of timestep data from the environment."""


class LearnerState(tp.NamedTuple):
  """Complete state of learner used for checkpointing / modifying loss."""
  params: hk.Params
  target_params: hk.Params
  opt_state: optax.OptState
  learner_steps: int
  extra: tp.Optional[tp.Dict[str, tp.Any]] = None


class LossFn(te.Protocol[Input, Output]):
  """A LossFn defines how to process one batch of data, for one random key.

  te.Protocol means that any functions with this signature are also valid
  LossFn.
  """

  def __call__(self,
               enn: enn_base.EpistemicNetwork[Input, Output],
               params: hk.Params,
               state: LearnerState,
               batch: reverb.ReplaySample,
               key: chex.PRNGKey) -> tp.Tuple[chex.Array, LossMetrics]:
    """Compute the loss on a single batch of data, for one random key."""
