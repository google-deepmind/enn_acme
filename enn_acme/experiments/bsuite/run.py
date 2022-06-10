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
"""Run a JAX agent on a bsuite experiment."""
import typing

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from acme.jax import utils
from bsuite import bsuite
from enn import base_legacy as enn_base
from enn_acme import agents as enn_agents
from enn_acme import losses
from enn_acme import planners
from enn_acme.experiments.bsuite import prior_scale
import jax.numpy as jnp
from neural_testbed import agents as testbed_agents
from neural_testbed import base as testbed_base
from neural_testbed.agents.factories import epinet
from neural_testbed.agents.factories.sweeps import testbed as factories
import numpy as np


# Which bsuite environment to run
_BSUITE_ID = flags.DEFINE_string('bsuite_id', 'catch/0',
                                 'Which bsuite environment to run.')
_RESULTS_DIR = flags.DEFINE_string('results_dir', '/tmp/',
                                   'Where to save csv files.')
_OVERWRITE = flags.DEFINE_bool('overwrite', True,
                               'Whether to overwrite results.')

# ENN training
_AGENT_ID = flags.DEFINE_string('agent_id', 'epinet',
                                'Which benchmark agent to run.')
_SEED = flags.DEFINE_integer('seed', 0, 'Seed for agent.')
_INDEX_DIM = flags.DEFINE_integer('index_dim', 2, '')
_HIDDEN_SIZE = flags.DEFINE_integer('hidden_size', 50, '')
_NUM_EPISODES = flags.DEFINE_integer('num_episodes', None,
                                     'Number of episodes to run.')
_NUM_INDEX_SAMPLES = flags.DEFINE_integer(
    'num_index_samples', 20, 'Number of index samples per batch of learning.')
_EPSILON = flags.DEFINE_float('epsilon', 0., 'Epsilon greedy.')
_PRIOR_SCALE = flags.DEFINE_float('prior_scale', 1.,
                                  'Scale for prior function.')

FLAGS = flags.FLAGS


def _spec_to_prior(spec: specs.EnvironmentSpec) -> testbed_base.PriorKnowledge:
  """Converts environment spec to neural testbed prior knowledge."""
  dummy_input = utils.zeros_like(spec.observations)
  return testbed_base.PriorKnowledge(
      input_dim=np.prod(dummy_input.shape),  # Unused
      num_train=100,  # Unused
      tau=10,  # Unused
      num_classes=spec.actions.num_values,
      temperature=1,
  )


def _wrap_with_flatten(
    enn: enn_base.EpistemicNetwork) -> enn_base.EpistemicNetwork:
  """Wraps an ENN with a flattening layer."""
  flatten = lambda x: jnp.reshape(x, [x.shape[0], -1])
  return enn_base.EpistemicNetwork(
      apply=lambda p, x, z: enn.apply(p, flatten(x), z),
      init=lambda k, x, z: enn.init(k, flatten(x), z),
      indexer=enn.indexer,
  )


def make_enn(agent: str,
             spec: specs.EnvironmentSpec) -> enn_base.EpistemicNetwork:
  """Parse the ENN from the agent name and prior."""
  # Parse testbed "prior" information from environment
  prior = _spec_to_prior(spec)

  # Obtain the std observed of a random linear function of observations,
  # calculated by letting a random agent observe 100 timesteps.
  problem_std = prior_scale.problem_std(_BSUITE_ID.value)

  if agent == 'epinet':
    # Overriding epinet config
    config = epinet.EpinetConfig(
        index_dim=_INDEX_DIM.value,
        prior_scale=_PRIOR_SCALE.value / problem_std,
        hidden_sizes=[_HIDDEN_SIZE.value, _HIDDEN_SIZE.value],
        epi_hiddens=[50],
        add_hiddens=(),
        seed=_SEED.value,
    )
    testbed_agent = epinet.make_agent(config)
    enn = testbed_agent.config.enn_ctor(prior)

  else:
    # Use the default agent settings
    paper_agent = factories.get_paper_agent(agent)
    config = paper_agent.default
    config.seed = _SEED.value

    # Rescale problem std if available
    if hasattr(config, 'prior_scale'):
      config.prior_scale /= problem_std

    testbed_agent = paper_agent.ctor(config)
    assert isinstance(testbed_agent, testbed_agents.VanillaEnnAgent)
    testbed_agent = typing.cast(testbed_agents.VanillaEnnAgent, testbed_agent)
    enn = testbed_agent.config.enn_ctor(prior)

  return _wrap_with_flatten(enn)


def main(_):
  """Runs a DQN agent on a given bsuite environment, logging to CSV."""
  environment = bsuite.load_and_record_to_csv(
      _BSUITE_ID.value, _RESULTS_DIR.value, overwrite=_OVERWRITE.value)
  environment = wrappers.SinglePrecisionWrapper(environment)
  spec = specs.make_environment_spec(environment)

  enn = make_enn(_AGENT_ID.value, spec)
  config = enn_agents.AgentConfig()  # Can override options from FLAGS here.
  single_loss = losses.ClippedQlearning(discount=0.99)
  agent = enn_agents.EnnAgent(
      environment_spec=spec,
      enn=enn,
      loss_fn=losses.average_single_index_loss(single_loss,
                                               _NUM_INDEX_SAMPLES.value),
      planner=planners.ThompsonQPlanner(
          enn, _SEED.value, epsilon=_EPSILON.value),
      config=config,
  )

  num_episodes = _NUM_EPISODES.value or environment.bsuite_num_episodes  # pytype: disable=attribute-error
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=num_episodes)


if __name__ == '__main__':
  app.run(main)
