# ENN Agent in Acme

`enn_acme` defines an interface for designing RL agents that combines:

1. `enn`: https://github.com/deepmind/enn
2. `acme`: https://github.com/deepmind/acme

Before diving into `enn_acme`, you should first read the tutorials for both of these underlying libraries.
`enn_acme` is really a thin convenience layer designed to expose certain "key concepts" in agent design.

## Key concepts

We outline the key high-level interfaces for our code in base.py:

-   `enn`: epistemic neural network = knowledge representation.
-   `EnnPlanner`: selects actions given (params, observation) for ENN.
-   `LossFn`: tells the agent how to evaluate loss of a given batch of data.


## Getting started

The best place to get started is in our [colab tutorial].

Here, you can find a quick rundown of our main types in the code, and find an example of running a version of bootstrapped DQN on `bsuite` environments.

Once you have had a look at this colab, you might want to have a look at our example `experiments/`.

### Installation

We have tested `enn_acme` on Python 3.7. To install the dependencies:

1.  **Optional**: We recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies, so as not to clobber your system installation:

    ```bash
    python3 -m venv enn_acme
    source enn_acme/bin/activate
    pip install --upgrade pip setuptools
    ```

2.  Install `enn_acme` directly from [github](https://github.com/deepmind/enn_acme):

    ```bash
    pip install git+https://github.com/deepmind/enn_acme
    ```

More examples can be found in the [colab tutorial].

4. **Optional**: run the tests by executing `./test.sh` from `enn_acme` root directory.

## Citing

If you use `enn_acme` in your work, please cite the [enn paper] and [acme report]:

```bibtex
@article{osband2021epistemic,
  title={Epistemic neural networks},
  author={Osband, Ian and Wen, Zheng and Asghari, Mohammad and Ibrahimi, Morteza and Lu, Xiuyuan and Van Roy, Benjamin},
  journal={arXiv preprint arXiv:2107.08924},
  year={2022}
}
```

```bibtex
@article{hoffman2020acme,
    title={Acme: A Research Framework for Distributed Reinforcement Learning},
    author={Matt Hoffman and Bobak Shahriari and John Aslanides and Gabriel
        Barth-Maron and Feryal Behbahani and Tamara Norman and Abbas Abdolmaleki
        and Albin Cassirer and Fan Yang and Kate Baumli and Sarah Henderson and
        Alex Novikov and Sergio Gómez Colmenarejo and Serkan Cabi and Caglar
        Gulcehre and Tom Le Paine and Andrew Cowie and Ziyu Wang and Bilal Piot
        and Nando de Freitas},
    year={2020},
    journal={arXiv preprint arXiv:2006.00979},
    url={https://arxiv.org/abs/2006.00979},
}
```

[colab tutorial]: https://colab.research.google.com/github/deepmind/enn_acme/blob/master/enn_acme/tutorial.ipynb
[enn paper]: https://arxiv.org/abs/2107.08924
[acme report]: https://arxiv.org/abs/2006.00979
