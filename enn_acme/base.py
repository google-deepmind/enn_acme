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
from typing import Any, Dict, NamedTuple, Optional, Tuple

from acme import types
import dm_env
from enn import base as enn_base
import haiku as hk
import optax
import reverb
import typing_extensions

Action = int
LossMetrics = Dict[str, enn_base.Array]


@dataclasses.dataclass
class EnnPlanner(abc.ABC):
  """A planner can select actions based on an ENN knowledge representation.

  The EnnPlanner is one of the core units we often change in algorithm design.
  Examples might include Epsilon-Greedy, Thompson Sampling, IDS and more...
  We implement some examples in `planners.py`.

  The EnnPlanner is then used by an acme.Actor to select actions. However, we do
  not implement this as an acme.Actor since we want to hide the variable_client
  and choice of replay "adder" from action selection. For example of how this is
  used to make an acme.Actor see PlannerActor in agents/acting.py.
  """
  enn: enn_base.EpistemicNetwork
  seed: Optional[int] = 0

  @abc.abstractmethod
  def select_action(
      self, params: hk.Params, observation: enn_base.Array) -> Action:
    """Selects an action given params and observation."""

  def observe_first(self, timestep: dm_env.TimeStep):
    """Optional: make a first observation from the environment."""

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    """Optional: make an observation of timestep data from the environment."""


class LearnerState(NamedTuple):
  """Complete state of learner used for checkpointing / modifying loss."""
  params: hk.Params
  target_params: hk.Params
  opt_state: optax.OptState
  learner_steps: int
  extra: Optional[Dict[str, Any]] = None


class LossFn(typing_extensions.Protocol):
  """A LossFn defines how to process one batch of data, for one random key.

  typing_extensions.Protocol means that any functions with this signature
  are also valid LossFn.
  """

  def __call__(self,
               enn: enn_base.EpistemicNetwork,
               params: hk.Params,
               state: LearnerState,
               batch: reverb.ReplaySample,
               key: enn_base.RngKey) -> Tuple[enn_base.Array, LossMetrics]:
    """Compute the loss on a single batch of data, for one random key."""
