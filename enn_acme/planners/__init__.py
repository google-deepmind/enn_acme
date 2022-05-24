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
"""Exposing the public methods of the planners."""

# IDS
from enn_acme.planners.ids_sample import compute_var_cond_mean
from enn_acme.planners.ids_sample import DiscreteInformatioRatioOptimizer
from enn_acme.planners.ids_sample import IdsPlanner
from enn_acme.planners.ids_sample import InformationCalculator
from enn_acme.planners.ids_sample import InformationRatioOptimizer
from enn_acme.planners.ids_sample import make_default_variance_ids_planner
from enn_acme.planners.ids_sample import RegretCalculator
from enn_acme.planners.ids_sample import RegretWithPessimism
from enn_acme.planners.ids_sample import VarianceGVF
from enn_acme.planners.ids_sample import VarianceOptimalAction

# Random
from enn_acme.planners.random import RandomPlanner

# Thompson sampling
from enn_acme.planners.thompson import ThompsonQPlanner
