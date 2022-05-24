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
"""Tests for planners.ids_sample."""

from absl.testing import absltest
from absl.testing import parameterized
from enn_acme.planners import ids_sample
import numpy as np
import pandas as pd


class ComputeVarCondMeanTest(parameterized.TestCase):

  def test_normal_case(self):
    batched_q = np.array([
        [0., -1.],
        [2., 1.],
        [-2., 0],
        [1., 2.],
    ])
    expected_var_cond_mean = np.array([0.5625, 0.25])
    var_cond_mean = ids_sample.compute_var_cond_mean(batched_q)
    self.assertTrue(
        (np.absolute(expected_var_cond_mean - var_cond_mean) < 1e-5).all(),
        msg=(f'expected var of cond means to be {expected_var_cond_mean}, '
             f'observed {var_cond_mean}')
        )

  def test_class_imbalance(self):
    batched_q = np.array([
        [0., -1.],
        [2., 1.],
        [-2., 0],
    ])
    expected_var_cond_mean = np.array([2., 0.])
    var_cond_mean = ids_sample.compute_var_cond_mean(batched_q)
    self.assertTrue(
        (np.absolute(expected_var_cond_mean - var_cond_mean) < 1e-5).all(),
        msg=(f'expected var of cond means to be {expected_var_cond_mean}, '
             f'observed {var_cond_mean}')
        )

  def test_one_class(self):
    batched_q = np.array([
        [0., -1.],
        [2., 1.],
    ])
    expected_var_cond_mean = np.array([0., 0.])
    var_cond_mean = ids_sample.compute_var_cond_mean(batched_q)
    self.assertTrue(
        (np.absolute(expected_var_cond_mean - var_cond_mean) < 1e-5).all(),
        msg=(f'expected var of cond means to be {expected_var_cond_mean}, '
             f'observed {var_cond_mean}')
        )

  @parameterized.parameters(range(4))
  def test_random(self, seed):
    rng = np.random.default_rng(seed)
    batched_q = rng.normal(size=(16, 4))

    num_action = batched_q.shape[1]
    num_sample = batched_q.shape[0]
    q_mean = np.mean(batched_q, axis=0)
    # Currently use pandas to get a clear implementation.
    df = pd.DataFrame(np.asarray(batched_q), columns=range(num_action))
    df['optimal_action'] = df.apply(lambda x: x.argmax(), axis=1)
    total_probability = 0
    total_variance = 0
    for unused_optimal_action, sub_df in df.groupby('optimal_action'):
      conditional_probability = len(sub_df) / num_sample
      conditional_mean = np.mean(sub_df[range(num_action)].values, axis=0)
      conditional_variance = np.square(conditional_mean - q_mean)
      total_probability += conditional_probability
      total_variance += conditional_probability * conditional_variance
    self.assertAlmostEqual(total_probability, 1.0)

    var_cond_mean = ids_sample.compute_var_cond_mean(batched_q)
    self.assertTrue(
        (np.absolute(total_variance - var_cond_mean) < 1e-5).all(),
        msg=(f'expected var of cond means to be {total_variance}, '
             f'observed {var_cond_mean}')
        )


if __name__ == '__main__':
  absltest.main()
