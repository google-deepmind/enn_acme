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
"""Distributed ENN Agent."""

import dataclasses
import typing as tp

import acme
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.jax import savers
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
from enn import base as enn_base
from enn_acme import base as agent_base
from enn_acme.agents import acting
from enn_acme.agents import agent
from enn_acme.agents import learning
import launchpad as lp
import reverb
import typing_extensions as te


# Helpful Types
class _EnnFactory(te.Protocol[agent_base.Input, agent_base.Output]):
  """Defines an Enn based on environment specs."""

  def __call__(
      self,
      env_specs: specs.EnvironmentSpec
  ) -> enn_base.EpistemicNetwork[agent_base.Input, agent_base.Output]:
    """Defines an Enn based on environment specs."""


class _PlannerFactory(te.Protocol[agent_base.Input, agent_base.Output]):
  """Defines an Enn Planner from an Enn and a seed."""

  def __call__(
      self,
      enn: enn_base.EpistemicNetwork[agent_base.Input, agent_base.Output],
      seed: int,
  ) -> agent_base.EnnPlanner[agent_base.Input, agent_base.Output]:
    """Defines an Enn Planner from an Enn and a seed."""


@dataclasses.dataclass
class DistributedEnnAgent(tp.Generic[agent_base.Input, agent_base.Output]):
  """Distributed Enn agent."""
  # Constructors for key agent components.
  environment_factory: tp.Callable[[bool], dm_env.Environment]
  enn_factory: _EnnFactory[agent_base.Input, agent_base.Output]
  loss_fn: agent_base.LossFn[agent_base.Input, agent_base.Output]
  planner_factory: _PlannerFactory[agent_base.Input, agent_base.Output]

  # Agent configuration.
  config: agent.AgentConfig
  environment_spec: specs.EnvironmentSpec
  input_spec: tp.Optional[specs.Array] = None

  # Distributed configuration.
  num_actors: int = 1
  num_caches: int = 1
  variable_update_period: int = 1000
  log_to_bigtable: bool = False

  name: str = 'distributed_agent'

  # Placeholder for launchpad program.
  _program: tp.Optional[lp.Program] = None

  def replay(self):
    """The replay storage."""
    if self.config.samples_per_insert:
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self.config.min_replay_size,
          samples_per_insert=self.config.samples_per_insert,
          error_buffer=self.config.batch_size,
      )
    else:
      limiter = reverb.rate_limiters.MinSize(self.config.min_replay_size)
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self.config.max_replay_size,
        rate_limiter=limiter,
        signature=adders.NStepTransitionAdder.signature(
            environment_spec=self.environment_spec),
    )
    return [replay_table]

  def counter(self):
    """Creates the master counter process."""
    counter = counting.Counter()
    return savers.CheckpointingRunner(
        counter, time_delta_minutes=1, subdirectory='counter')

  def learner(self, replay: reverb.Client, counter: counting.Counter):
    """The Learning part of the agent."""

    # The dataset object to learn from.
    dataset = datasets.make_reverb_dataset(
        server_address=replay.server_address,
        batch_size=self.config.batch_size,
        prefetch_size=self.config.prefetch_size,
    )

    logger = loggers.make_default_logger('learner', time_delta=10.)
    counter = counting.Counter(counter, 'learner')

    # Return the learning agent.
    input_spec = self.input_spec or self.environment_spec.observations
    learner = learning.SgdLearner[agent_base.Input, agent_base.Output](
        input_spec=input_spec,
        enn=self.enn_factory(self.environment_spec),
        loss_fn=self.loss_fn,
        optimizer=self.config.optimizer,
        data_iterator=dataset.as_numpy_iterator(),
        target_update_period=self.config.target_update_period,
        seed=self.config.seed,
        counter=counter,
        logger=logger,
    )

    return savers.CheckpointingRunner(
        learner, time_delta_minutes=60, subdirectory='learner')

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
      *,
      actor_id: int,
  ) -> acme.EnvironmentLoop:
    """The actor process."""

    environment = self.environment_factory(False)
    enn = self.enn_factory(self.environment_spec)
    planner = self.planner_factory(enn, self.config.seed + actor_id)

    # Component to add things into replay.
    adder = adders.NStepTransitionAdder(
        priority_fns={adders.DEFAULT_PRIORITY_TABLE: lambda x: 1.},
        client=replay,
        n_step=self.config.n_step,
        discount=self.config.adder_discount,
    )
    variable_client = variable_utils.VariableClient(
        variable_source, '', update_period=self.variable_update_period)
    actor = acting.PlannerActor[agent_base.Input, agent_base.Output](
        planner, variable_client, adder)

    # Create the loop to connect environment and agent.
    counter = counting.Counter(counter, 'actor')
    logger = loggers.make_default_logger('actor', save_data=False)

    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def evaluator(
      self,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ):
    """The evaluation process."""
    environment = self.environment_factory(True)
    enn = self.enn_factory(self.environment_spec)
    planner = self.planner_factory(enn, self.config.seed + 666)

    variable_client = variable_utils.VariableClient(
        variable_source, '', update_period=self.variable_update_period)
    actor = acting.PlannerActor(planner, variable_client, adder=None)

    # Create the run loop and return it.
    logger = loggers.make_default_logger('evaluator')
    counter = counting.Counter(counter, 'evaluator')
    return acme.EnvironmentLoop(
        environment, actor, counter=counter, logger=logger)

  def _build(self, name: str) -> lp.Program:
    """Builds the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      replay = program.add_node(lp.ReverbNode(self.replay))

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))

    with program.group('learner'):
      learner = program.add_node(lp.CourierNode(self.learner, replay, counter))

    with program.group('evaluator'):
      program.add_node(lp.CourierNode(self.evaluator, learner, counter))

    with program.group('cacher'):
      # Create a set of learner caches.
      sources = []
      for _ in range(self.num_caches):
        cacher = program.add_node(
            # TODO(author2): Remove CacherNode as it is only for internal use
            lp.CacherNode(
                learner, refresh_interval_ms=2000, stale_after_ms=4000))
        sources.append(cacher)

    with program.group('actor'):
      # Add actors which pull round-robin from our variable sources.
      for actor_id in range(self.num_actors):
        source = sources[actor_id % len(sources)]
        node = lp.CourierNode(
            self.actor,
            replay,
            source,
            counter,
            actor_id=actor_id)
        program.add_node(node)

    return program

  @property
  def program(self) -> lp.Program:
    if self._program is None:
      self._program = self._build(name=self.name)
    return self._program
