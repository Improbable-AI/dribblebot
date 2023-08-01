# Go1 Soccer (Dribblebot) Starter Kit

# Table of contents
1. [Overview](#overview)
2. [System Requirements](#requirements)
3. [Training a Model](#simulation)
    1. [Installation](#installation)
    2. [Environment and Model Configuration](#configuration)
    3. [Training, Logging and Evaluation](#training)
4. [Deploying a Model (Coming Soon)](#realworld)

## Overview <a name="overview"></a>

This repository provides an implementation of the paper:


<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://gmargo11.github.io/dribblebot/" target="_blank">
      <b> DribbleBot: Dynamic Legged Manipulation in the Wild </b>
      </a>
      <br>
      <a href="https://yandongji.github.io/" target="_blank">Yandong Ji*</a>, <a href="https://gmargo11.github.io/" target="_blank">Gabriel B. Margolis*</a> and <a href="https://people.csail.mit.edu/pulkitag" target="_blank">Pulkit Agrawal</a>
      <br>
      <em>International Conference on Robotics and Automation (ICRA)</em>, 2023
      <br>
      <a href="https://arxiv.org/pdf/2304.01159.pdf">paper</a> /
      <a href="">bibtex</a> /
      <a href="https://gmargo11.github.io/dribblebot/" target="_blank">project page</a>
    <br>
</td>

<br>

This training code, environment and documentation build on [Walk these Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://github.com/Improbable-AI/walk-these-ways) by Gabriel Margolis and Pulkit Agrawal, Improbable AI Lab, MIT (Paper: https://arxiv.org/pdf/2212.03238.pdf) and the Isaac Gym simulator from 
NVIDIA (Paper: https://arxiv.org/abs/2108.10470). All redistributed code retains its
original [license](LICENSES/legged_gym/LICENSE).

Our initial release provides the following features:
* Train reinforcement learning policies for the Go1 robot using PPO, IsaacGym, Domain Randomization to dribble a soccer ball in simulation following a random ball velocity command in global frame.
* Evaluate a pre-trained soccer policy in simulation.

## System Requirements <a name="requirements"></a>

**Simulated Training and Evaluation**: Isaac Gym requires an NVIDIA GPU. To train in the default configuration, we recommend a GPU with at least 10GB of VRAM. The code can run on a smaller GPU if you decrease the number of parallel environments (`Cfg.env.num_envs`). However, training will be slower with fewer environments.

## Training a Model <a name="simulation"></a>

### Installation using Conda<a name="installation"></a>

#### Create a new conda environment with Python (3.8 suggested)
```bash
conda create -n dribblebot python==3.8
conda activate dribblebot
```
#### Install pytorch 1.10 with cuda-11.3:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

#### Install Isaac Gym

1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
2. unzip the file via:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```

3. now install the python package
    ```bash
    cd isaacgym/python && pip install -e .
    ```
4. Verify the installation by try running an example

    ```bash
    python examples/1080_balls_of_solitude.py
    ```
5. For troubleshooting check docs `isaacgym/docs/index.html`

#### Install the `dribblebot` package

In this repository, run `pip install -e .`

### Evaluate the pre-trained policy

If everything is installed correctly, you should be able to run the evaluation script with:

```bash
python scripts/play_dribbling_pretrained.py
```

You should see a robot manipulate a yellow soccer following random global velocity commands.

### Environment and Model Configuration <a name="configuration"></a>


**CODE STRUCTURE** The main environment for simulating a legged robot is
in [legged_robot.py](dribblebot/envs/base/legged_robot.py). The default configuration parameters including reward
weightings are defined in [legged_robot_config.py::Cfg](dribblebot/envs/base/legged_robot_config.py).

There are three scripts in the [scripts](scripts/) directory:

```bash
scripts
├── __init__.py
├── play_dribbling_custom.py
├── play_dribbling_pretrained.py
└── train_dribbling.py
```

### Training, Logging and evaluation <a name="training"></a>

To train the Go1 controller from [Dribblebot](https://gmargo11.github.io/dribblebot/), run: 

```bash
python scripts/train_dribbling.py
```

After initializing the simulator, the script will print out a list of metrics every ten training iterations.

Training with the default configuration requires about 12GB of GPU memory. If you have less memory available, you can 
still train by reducing the number of parallel environments used in simulation (the default is `Cfg.env.num_envs = 1000`).

To visualize training progress, first set up weights and bias (wandb):

#### Set Up Weights and Bias (wandb):

Weights and Biases is the service that will provide you a dashboard where you can see the progress log of your training runs, including statistics and videos.

First, follow the instructions here to create you wandb account: https://docs.wandb.ai/quickstart

Make sure to perform the `wandb.login()` step from your local computer.

Finally, use a web browser to go to the wandb IP (defaults to `localhost:3001`) 

To evaluate a pretrained trained policy, run `play_dribbling_pretrained.py`. We provie a pretrained agent checkpoint in the [./runs/dribbling](runs/dribbling) directory.

## Deploying a Model (Coming Soon) <a name="realworld"></a>

We are working a modular version of the vision processing code so DribbleBot can be easily deployed on Go1. It will be added in a future release.