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
"""Helpful functions relating to losses."""
from typing import Callable, Optional, Tuple

from enn import base as enn_base
from enn import losses as enn_losses
from enn_acme import base as agent_base
import haiku as hk
import reverb


def add_l2_weight_decay(
    loss_fn: agent_base.LossFn,
    scale_fn: Callable[[int], float],  # Maps learner_steps --> l2 decay
    predicate: Optional[enn_losses.PredicateFn] = None,
) -> agent_base.LossFn:
  """Adds l2 weight decay to a given loss function."""
  def new_loss(
      enn: enn_base.EpistemicNetwork,
      params: hk.Params,
      state: agent_base.LearnerState,
      batch: reverb.ReplaySample,
      key: enn_base.RngKey,
  ) -> Tuple[enn_base.Array, agent_base.LossMetrics]:
    loss, metrics = loss_fn(enn, params, state, batch, key)
    l2_penalty = enn_losses.l2_weights_with_predicate(params, predicate)
    decay = l2_penalty * scale_fn(state.learner_steps)
    total_loss = loss + decay
    metrics['decay'] = decay
    metrics['raw_loss'] = loss
    return total_loss, metrics
  return new_loss
