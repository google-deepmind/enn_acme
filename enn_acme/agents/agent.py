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
"""A single process agent definition.

Combines an actor, learner and replay server with some logic to handle the
ratio between learning and acting.
"""
import dataclasses
import typing as tp

from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent as agent_lib
from acme.agents import replay
from acme.jax import variable_utils
from acme.utils import loggers
from enn import base as enn_base
from enn_acme import base as agent_base
from enn_acme.agents import acting
from enn_acme.agents import learning
import optax


# Simple alises for generic modules
_ENN = enn_base.EpistemicNetwork[agent_base.Input, agent_base.Output]
_LossFn = agent_base.LossFn[agent_base.Input, agent_base.Output]
_Planner = agent_base.EnnPlanner[agent_base.Input, agent_base.Output]


@dataclasses.dataclass
class AgentConfig:
  """Configuration options for single-process agent."""
  seed: int = 0

  # N-step adder options
  n_step: int = 1  # Number of transitions in each sample
  adder_discount: float = 1.  # Only used in N-step learning

  # Learner options
  optimizer: optax.GradientTransformation = optax.adam(1e-3)
  target_update_period: int = 4
  learner_logger: tp.Optional[loggers.Logger] = None

  # Replay options
  batch_size: int = 128
  min_replay_size: int = 128
  max_replay_size: int = 10_000
  samples_per_insert: int = 128
  prefetch_size: int = 4
  replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE


class EnnAgent(agent_lib.Agent, tp.Generic[agent_base.Input,
                                           agent_base.Output]):
  """A single-process Acme agent based around an ENN."""

  def __init__(self,
               enn: _ENN[agent_base.Input, agent_base.Output],
               loss_fn: _LossFn[agent_base.Input, agent_base.Output],
               planner: _Planner[agent_base.Input, agent_base.Output],
               config: AgentConfig,
               environment_spec: specs.EnvironmentSpec,
               input_spec: tp.Optional[specs.Array] = None):
    # Data is handled via the reverb replay.
    reverb_replay = replay.make_reverb_prioritized_nstep_replay(
        environment_spec=environment_spec,
        batch_size=config.batch_size,
        max_replay_size=config.max_replay_size,
        min_replay_size=1,
        n_step=config.n_step,
        discount=config.adder_discount,
        replay_table_name=config.replay_table_name,
        prefetch_size=config.prefetch_size,
    )
    self._server = reverb_replay.server

    # Learner updates ENN knowledge representation.
    input_spec = input_spec or environment_spec.observations
    learner = learning.SgdLearner[agent_base.Input, agent_base.Output](
        input_spec=input_spec,
        enn=enn,
        loss_fn=loss_fn,
        optimizer=config.optimizer,
        data_iterator=reverb_replay.data_iterator,
        target_update_period=config.target_update_period,
        seed=config.seed,
        logger=config.learner_logger,
    )

    # Select actions according to the actor
    actor = acting.PlannerActor[agent_base.Input, agent_base.Output](
        planner=planner,
        variable_client=variable_utils.VariableClient(learner, ''),
        adder=reverb_replay.adder,
    )

    # Wrap actor and learner as single-process agent.
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(config.min_replay_size, config.batch_size),
        observations_per_step=config.batch_size / config.samples_per_insert,
    )
