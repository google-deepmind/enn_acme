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

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from bsuite import bsuite
from enn_acme import agents
from enn_acme import base as agent_base
from enn_acme import losses as agent_losses
from enn_acme.experiments.model_based import base
from enn_acme.experiments.model_based import losses
from enn_acme.experiments.model_based import networks
from enn_acme.experiments.model_based import planners
from enn_acme.losses import utils as loss_utils


# Which bsuite environment to run
flags.DEFINE_string('bsuite_id', 'catch/0', 'Which bsuite environment to run.')
flags.DEFINE_string('results_dir', '/tmp/', 'Where to save csv files.')
flags.DEFINE_bool('overwrite', True, 'Whether to overwrite results.')

# Agent flags
flags.DEFINE_integer('seed', 0, 'Seed for experiment.')
flags.DEFINE_integer('hidden_dim', 32, 'Latent hidden.')
flags.DEFINE_integer('num_episodes', None, 'Number of episodes to run.')
flags.DEFINE_float('epsilon', 0., 'Epsilon greedy.')
flags.DEFINE_bool('use_policy_head', False,
                  'Whether to use the policy head to select actions.')


FLAGS = flags.FLAGS

# Shorthands for the ENN input and output types
_Input = base.Input
_Output = base.DepthOneSearchOutput


def _make_loss() -> agent_base.LossFn[_Input, _Output]:
  """Make the loss function for the model-based agent."""
  reward_loss = losses.RewardLoss()
  value_loss = losses.ValueLoss(discount=0.99)
  policy_loss = losses.PolicyLoss(temperature=0.001, discount=0.99)
  cfg_class = loss_utils.CombineLossConfig[_Input, _Output]
  ave_fn = agent_losses.average_single_index_loss
  loss_configs = [
      cfg_class(ave_fn(reward_loss, num_index_samples=1), 'reward'),
      cfg_class(ave_fn(value_loss, num_index_samples=1), 'value'),
      cfg_class(ave_fn(policy_loss, num_index_samples=1), 'policy'),
  ]
  return loss_utils.combine_losses(loss_configs)


def main(_):
  """Runs a model-based agent on a given bsuite environment, logging to CSV."""
  # Load environment
  environment = bsuite.load_and_record_to_csv(
      FLAGS.bsuite_id, results_dir=FLAGS.results_dir, overwrite=FLAGS.overwrite)
  environment = wrappers.SinglePrecisionWrapper(environment)
  spec = specs.make_environment_spec(environment)
  num_actions = spec.actions.num_values

  # Define the model and ENN
  model = networks.make_single_model(
      action_embedding=networks.make_outer_one_hot_embedding(num_actions),
      hidden_dim=FLAGS.hidden_dim,
      num_actions=num_actions,
      repr_hiddens=[20],
      dynamics_hiddens=[],
      pred_hiddens=[20],
  )
  enn = base.make_enn_onestep(model, num_actions)

  # Planner
  if FLAGS.use_policy_head:
    planner = planners.ThompsonPolicyPlanner(enn, FLAGS.seed, FLAGS.epsilon)
  else:
    planner = planners.ThompsonQPlanner(
        enn, FLAGS.seed, FLAGS.epsilon, discount=0.99)

  # Define the agent
  config = agents.AgentConfig(seed=FLAGS.seed)
  agent = agents.EnnAgent[_Input, _Output](
      environment_spec=spec,
      enn=enn,
      loss_fn=_make_loss(),
      planner=planner,
      config=config,
  )

  num_episodes = FLAGS.num_episodes or environment.bsuite_num_episodes  # pytype: disable=attribute-error
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=num_episodes)


if __name__ == '__main__':
  app.run(main)
