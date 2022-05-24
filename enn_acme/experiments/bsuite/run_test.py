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
"""Tests for experiments.bsuite.run."""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from bsuite import sweep
from enn_acme.experiments.bsuite import run
import jax


FLAGS = flags.FLAGS

# Parse absl flags
jax.config.parse_flags_with_absl()


class RunTest(parameterized.TestCase):

  @parameterized.parameters([[bsuite_id] for bsuite_id in sweep.TESTING])
  def test_each_bsuite_env(self, bsuite_id: str):
    with flagsaver.flagsaver(bsuite_id=bsuite_id,
                             num_episodes=5,
                             index_dim=1,):
      run.main(None)


if __name__ == '__main__':
  absltest.main()
