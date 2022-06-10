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
"""EnnLearner is a learner compatible with Acme.

The learner takes batches of data and learns on them via `step()`. The core
logic is implemented via the loss: agent_base.LossFn.
"""
import functools
from typing import Iterator, List, Optional, Tuple

import acme
from acme import specs
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from enn import base_legacy as enn_base
from enn_acme import base as agent_base
import haiku as hk
import jax
import optax
import reverb


class SgdLearner(acme.Learner, acme.Saveable):
  """A Learner for acme library based around SGD on batches."""

  def __init__(
      self,
      input_spec: specs.Array,
      enn: enn_base.EpistemicNetwork,
      loss_fn: agent_base.LossFn,
      optimizer: optax.GradientTransformation,
      data_iterator: Iterator[reverb.ReplaySample],
      target_update_period: int,
      seed: int = 0,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):
    """Initialize the Enn Learner."""
    self.enn = enn

    # Internalize the loss_fn
    self._loss = jax.jit(functools.partial(loss_fn, self.enn))

    # SGD performs the loss, optimizer update and periodic target net update.
    def sgd_step(
        state: agent_base.LearnerState,
        batch: reverb.ReplaySample,
        key: enn_base.RngKey,
    ) -> Tuple[agent_base.LearnerState, agent_base.LossMetrics]:
      # Implements one SGD step of the loss and updates the learner state
      (loss, metrics), grads = jax.value_and_grad(
          self._loss, has_aux=True)(state.params, state, batch, key)
      metrics.update({'total_loss': loss})

      # Apply the optimizer updates
      updates, new_opt_state = optimizer.update(grads, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      # Periodically update target networks.
      steps = state.learner_steps + 1
      target_params = optax.periodic_update(new_params, state.target_params,
                                            steps, target_update_period)
      new_learner_state = agent_base.LearnerState(new_params, target_params,
                                                  new_opt_state, steps)
      return new_learner_state, metrics

    self._sgd_step = jax.jit(sgd_step)

    # Internalise agent components
    self._data_iterator = utils.prefetch(data_iterator)
    self._rng = hk.PRNGSequence(seed)
    self._target_update_period = target_update_period
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Initialize the network parameters
    dummy_index = self.enn.indexer(next(self._rng))
    dummy_input = utils.add_batch_dim(
        jax.tree_map(lambda x: x.generate_value(), input_spec))
    initial_params = self.enn.init(next(self._rng), dummy_input, dummy_index)
    self._state = agent_base.LearnerState(
        params=initial_params,
        target_params=initial_params,
        opt_state=optimizer.init(initial_params),
        learner_steps=0,
    )

  def step(self):
    """Take one SGD step on the learner."""
    self._state, loss_metrics = self._sgd_step(self._state,
                                               next(self._data_iterator),
                                               next(self._rng))

    # Update our counts and record it.
    result = self._counter.increment(steps=1)
    result.update(loss_metrics)
    self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[hk.Params]:
    return [self._state.params]

  def save(self) -> agent_base.LearnerState:
    return self._state

  def restore(self, state: agent_base.LearnerState):
    self._state = state
