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
"""Install script for setuptools."""

import imp  # pylint: disable=deprecated-module

import setuptools

# Additional requirements for testing.
testing_require = [
    'gym',
    'gym[atari]',
    'mock',
    'pytest-xdist',
    'pytype',
]

setuptools.setup(
    name='enn_acme',
    description=(
        'ENN_ACME. '
        'An interface for designing RL agents in Acme using the ENN library.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/deepmind/enn_acme',
    author='DeepMind',
    author_email='enn_acme-eng+os@google.com',
    license='Apache License, Version 2.0',
    # TODO(author1): Use LINT.IfChange(version) instead of imp.
    version=imp.load_source('_metadata', 'enn_acme/_metadata.py').__version__,
    keywords='probabilistic-inference python machine-learning',
    packages=setuptools.find_packages(),
    install_requires=[
        'absl-py',
        'bsuite',
        'chex',
        'dm-acme @ git+https://git@github.com/deepmind/acme',
        'dm-env',
        'dm-haiku',
        'dm-launchpad==0.5.2',
        'dm-reverb==0.7.2',
        'dm-sonnet',
        'enn @ git+https://git@github.com/deepmind/enn',
        'jax',
        'jaxlib',
        'neural_testbed @ git+https://git@github.com/deepmind/neural_testbed',
        'numpy',
        'optax',
        'pandas',
        'pyarrow',
        'rlax',
        'requests',
        'tensorflow==2.8.0',
        'typing-extensions',
    ],
    extras_require={
        'testing': testing_require,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
