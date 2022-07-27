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
"""Tests for enn agent."""

from absl.testing import absltest
import acme
from acme import specs
from acme.jax import utils
from acme.testing import fakes
import chex
from enn import networks
from enn_acme import losses
from enn_acme import planners
from enn_acme.agents import agent as enn_agent
import numpy as np


class EnnTest(absltest.TestCase):

  def test_enn_agent(self):
    seed = 0

    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_shape=(10, 5),
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    enn = networks.MLPEnsembleMatchedPrior(
        output_sizes=[10, 10, spec.actions.num_values],
        dummy_input=utils.add_batch_dim(utils.zeros_like(spec.observations)),
        num_ensemble=2,
        prior_scale=1.,
    )
    test_config = enn_agent.AgentConfig()
    test_config.min_observations = test_config.batch_size = 10
    single_loss = losses.ClippedQlearning(discount=0.99)
    agent = enn_agent.EnnAgent[chex.Array, networks.Output](
        environment_spec=spec,
        enn=enn,
        loss_fn=losses.average_single_index_loss(single_loss, 1),
        planner=planners.ThompsonQPlanner(enn, seed),
        config=test_config,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=20)


if __name__ == '__main__':
  absltest.main()
