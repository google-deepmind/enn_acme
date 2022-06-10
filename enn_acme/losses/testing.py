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
"""A LossFn computes the loss on a batch of data for one index."""
from typing import Tuple

from enn import base_legacy as enn_base
from enn_acme import base as agent_base
import haiku as hk
import jax.numpy as jnp
import reverb


def dummy_loss(
    enn: enn_base.EpistemicNetwork,
    params: hk.Params,
    state: agent_base.LearnerState,
    batch: reverb.ReplaySample,
    key: enn_base.RngKey,
) -> Tuple[enn_base.Array, agent_base.LossMetrics]:
  """A dummy loss function that always returns loss=1."""
  del enn, params, state, batch, key
  return (jnp.ones(1), {})
