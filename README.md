**Status:** Archive (code is provided as-is, no updates expected)

# Phasic Policy Gradient

#### [[Paper]](https://arxiv.org/abs/2009.04416)

Our approach to the ProcGen FruitBot Final Project uses Phasic Policy Gradient with additions of reward normalization and action penalization. The explanation for PPG can be found at (https://arxiv.org/abs/2009.04416).

Supported platforms:

- macOS 10.14 (Mojave)
- Ubuntu 16.04

Supported Pythons:

- 3.7 64-bit

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html if you don't have it, or install the dependencies from [`environment.yml`](environment.yml) manually.

```
git clone https://github.com/openai/phasic-policy-gradient.git
conda env update --name phasic-policy-gradient --file phasic-policy-gradient/environment.yml
conda activate phasic-policy-gradient
pip install -e phasic-policy-gradient
```

## Training and Testing

To train the environment using Fruitbot, use the following command.
```
python -m phasic_policy_gradient.train --rnorm [False, True] --acpenalization [False, True]
```

To test the model on more difficult levels:
```
python -m phasic_policy_gradient.test --model_path path/to/model.jd
```
For either testing or training, to modify the levels you run on, change start_level and num_levels in line 9 of envs.py
