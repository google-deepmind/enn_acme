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
"""Actor handles interaction with the environment.

The Actor is a concept from the Acme library and is mostly a thin wrapper around
the planner + infrastructure to interface with the learner and replay.
"""
import dataclasses
from typing import Optional

import acme
from acme import adders
from acme import types
from acme.jax import variable_utils
import dm_env

from enn_acme import base as agent_base


@dataclasses.dataclass
class PlannerActor(acme.Actor):
  """An actor based on acme library wrapped around an EnnPlanner.

  The Actor is essentially a thin wrapper around the planner + infrastructure to
  interface with the learner and replay. For many research questions you will
  not need to edit this class.
  """
  planner: agent_base.EnnPlanner  # How to select actions from knowledge
  variable_client: variable_utils.VariableClient  # Communicate variables/params
  adder: Optional[adders.Adder] = None  # Interface with replay

  def select_action(self, observation: types.NestedArray) -> agent_base.Action:
    return self.planner.select_action(
        params=self.variable_client.params,
        observation=observation,
    )

  def observe_first(self, timestep: dm_env.TimeStep):
    self.planner.observe_first(timestep)
    if self.adder:
      self.adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    self.planner.observe(action, next_timestep)
    if self.adder:
      self.adder.add(action, next_timestep)

  def update(self):
    self.variable_client.update()

