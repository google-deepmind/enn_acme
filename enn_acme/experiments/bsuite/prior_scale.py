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
"""Handles prior rescaling on bsuite.

In our mathematical algorithm we rescale the sampled prior_fn based on the
observations of a random agent over the first 100 timesteps in bsuite. Since
our agent actually doesn't update its parameters over the first 128 steps it
is therefore mathematically equivalent to first run a uniformly random agent,
and use this value to rescale the prior at initialization.

We view this purely as a convenience in coding/implementation, and should have
absolutely no effect on the results.

For these problem_std, we sample a random linear: observation -> 1, and log
the observed standard deviation of that value.
"""

import os
import tempfile

import pyarrow.parquet as pq
import requests


def problem_std(bsuite_id: str) -> float:
  """Obtains std of a random linear function of input for that bsuite_id."""
  url = 'https://storage.googleapis.com/dm-enn/prior_scaling.parquet'

  with tempfile.TemporaryDirectory() as tmpdir:
    response = requests.get(url, verify=False)

    # Make a temporary file for downloading
    bsuite_name = bsuite_id.replace('/', '-')
    filepath = os.path.join(tmpdir, f'/tmp/prior_scale_{bsuite_name}.parquet')
    open(filepath, 'wb').write(response.content)

    # Read data from temporary file
    with open(filepath, 'rb') as f:
      table = pq.read_table(f)

  df = table.to_pandas()
  return df.loc[df.bsuite_id == bsuite_id, 'std'].iloc[0]

