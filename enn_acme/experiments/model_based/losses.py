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
"""Value learning loss."""
import dataclasses
import typing as tp

from acme import types
import chex
from enn import base as enn_base
from enn_acme import base as agent_base
from enn_acme import losses
from enn_acme.experiments.model_based import base
import haiku as hk
import jax
import jax.numpy as jnp
import reverb


class RewardLoss(losses.SingleLossFn[base.Input, base.DepthOneSearchOutput]):
  """Learns the reward on the observed transition."""

  def __call__(
      self,
      apply: base.ApplyOneStep,
      params: hk.Params,
      state: agent_base.LearnerState,
      batch: reverb.ReplaySample,
      index: enn_base.Index,
  ) -> tp.Tuple[chex.Array, agent_base.LossMetrics]:
    """Learns the reward on the observed transition."""
    dummy_network_state = {}
    transitions: types.Transition = batch.data
    batch_size = transitions.observation.shape[0]

    net_out, _ = apply(
        params, dummy_network_state, transitions.observation, index)
    output_tm1: base.DepthOneSearchOutput = net_out
    r_hat_tm1 = base.stack_action_rewards(output_tm1)  # (batch, actions)

    def compute_reward_err(
        reward: float, r_preds: chex.Array, action: int) -> float:
      return reward - r_preds[action]

    reward_err = jax.vmap(compute_reward_err)(
        transitions.reward, r_hat_tm1, transitions.action)
    chex.assert_shape(reward_err, [batch_size])
    r_loss = jnp.mean(reward_err ** 2)

    return r_loss, {}


@dataclasses.dataclass
class ValueLoss(losses.SingleLossFn[base.Input, base.DepthOneSearchOutput]):
  """Model-based value loss + one-step value equivalence."""
  discount: float = 0.99

  def __call__(
      self,
      apply: base.ApplyOneStep,
      params: hk.Params,
      state: agent_base.LearnerState,
      batch: reverb.ReplaySample,
      index: enn_base.Index,
  ) -> tp.Tuple[chex.Array, agent_base.LossMetrics]:
    """Learns the value on the observed transition."""
    dummy_network_state = {}
    transitions: types.Transition = batch.data
    batch_size = transitions.observation.shape[0]

    net_out_t, _ = apply(
        params, dummy_network_state, transitions.observation, index)
    output_t: base.DepthOneSearchOutput = net_out_t

    net_out_t_target, _ = apply(state.target_params, dummy_network_state,
                                transitions.observation, index)
    output_t_target: base.DepthOneSearchOutput = net_out_t_target

    net_out_tp1_target, _ = apply(state.target_params, dummy_network_state,
                                  transitions.next_observation, index)
    output_tp1_target: base.DepthOneSearchOutput = net_out_tp1_target

    # Naming: {variable}_{observation time}_{unrolled step}

    # Predictions starting at o_t
    v_t_0 = jnp.squeeze(output_t.root.value.preds, axis=-1)  # (B,)
    v_t_1 = base.stack_action_values(output_t)  # (B, A)
    r_t_1_target = base.stack_action_rewards(output_t_target)  # (B, A)
    v_t_1_target = base.stack_action_values(output_t_target)  # (B, A)

    # Predictions starting at o_tp1
    r_tp1_1_target = base.stack_action_rewards(output_tp1_target)  # (B, A)
    v_tp1_1_target = base.stack_action_values(output_tp1_target)  # (B, A)

    # Parsing discount and actions
    non_terminal_tp1 = transitions.discount.astype(jnp.float32)  # (B,)
    a_t = transitions.action  # (B,)

    # First unrolled step k=0
    def compute_td_err_0(
        v_t_0: chex.Array,
        r_t_1: chex.Array,
        v_t_1: chex.Array,
    ) -> float:
      q_t = r_t_1 + self.discount * v_t_1
      return jnp.max(q_t) - v_t_0

    td_errors = jax.vmap(compute_td_err_0)(v_t_0, r_t_1_target, v_t_1_target)
    chex.assert_shape(td_errors, [batch_size])
    v_loss_0 = jnp.mean(td_errors ** 2)

    # Second unrolled step k=1
    def compute_td_err_1(
        v_t_1: chex.Array,
        r_tp1_1: chex.Array,
        v_tp1_1: chex.Array,
        non_terminal: float,
        action: int,
    ) -> float:
      q_tp1 = (r_tp1_1 + self.discount * v_tp1_1) * non_terminal
      return jnp.max(q_tp1) - v_t_1[action]

    td_errors = jax.vmap(compute_td_err_1)(
        v_t_1, r_tp1_1_target, v_tp1_1_target, non_terminal_tp1, a_t)
    chex.assert_shape(td_errors, [batch_size])
    v_loss_1 = jnp.mean(td_errors ** 2)

    return v_loss_0 + v_loss_1, {}


@dataclasses.dataclass
class PolicyLoss(losses.SingleLossFn[base.Input, base.DepthOneSearchOutput]):
  """Learns the greedy policy with respect to the q-values."""
  temperature: float = 0.01
  discount: float = 0.99

  def __call__(
      self,
      apply: base.ApplyOneStep,
      params: hk.Params,
      state: agent_base.LearnerState,
      batch: reverb.ReplaySample,
      index: enn_base.Index,
  ) -> tp.Tuple[chex.Array, agent_base.LossMetrics]:
    """Learns the reward on the observed transition."""
    dummy_network_state = {}
    transitions: types.Transition = batch.data
    batch_size = transitions.observation.shape[0]

    net_out_tm1, _ = apply(
        params, dummy_network_state, transitions.observation, index)
    output_tm1: base.DepthOneSearchOutput = net_out_tm1
    net_out_t, _ = apply(
        params, dummy_network_state, transitions.next_observation, index)
    output_t: base.DepthOneSearchOutput = net_out_t

    # Reward, value, and policy (logits) predictions. All of these have shape
    # (batch, actions).
    r_tm1 = base.stack_action_rewards(output_tm1)
    v_tm1 = base.stack_action_values(output_tm1)
    p_tm1 = output_tm1.root.policy.preds
    r_t = base.stack_action_rewards(output_t)
    v_t = base.stack_action_values(output_t)
    p_t = output_t.root.policy.preds

    def compute_policy_loss(
        logits: chex.Array,
        reward: chex.Array,
        value: chex.Array,
    ) -> float:
      q_values = reward + self.discount * value
      label = jax.nn.softmax(q_values / self.temperature)
      return -jnp.sum(jax.lax.stop_gradient(label) * jax.nn.log_softmax(logits))

    p_loss_tm1 = jax.vmap(compute_policy_loss)(p_tm1, r_tm1, v_tm1)
    p_loss_t = jax.vmap(compute_policy_loss)(p_t, r_t, v_t)
    chex.assert_shape(p_loss_tm1, [batch_size])
    chex.assert_shape(p_loss_t, [batch_size])
    p_loss = jnp.mean(p_loss_tm1 + p_loss_t)

    return p_loss, {}
