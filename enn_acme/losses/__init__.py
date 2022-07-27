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
"""Exposing the public methods of the losses."""


# Single Index
from enn_acme.losses.single_index import average_single_index_loss
from enn_acme.losses.single_index import LossFnArray
from enn_acme.losses.single_index import SingleLossFn
from enn_acme.losses.single_index import SingleLossFnArray

# Testing
from enn_acme.losses.testing import dummy_loss

# Additional useful functions
from enn_acme.losses.utils import add_l2_weight_decay

# Value Learning
from enn_acme.losses.value_learning import Categorical2HotQlearning
from enn_acme.losses.value_learning import ClippedQlearning
