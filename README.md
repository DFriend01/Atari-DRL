# Atari-DRL

## Table of Contents

* [Project Description](#project-description)
* [Technologies](#technologies)
* [Setup](#setup)
* [Acknowledgements](#acknowledgements)

## Project Description

This project is an implementation of a deep reinforcement learning model that uses a general neural network architecture to train an agent to play [Atari 2600 games](https://en.wikipedia.org/wiki/List_of_Atari_2600_games). 

<img src="https://miro.medium.com/max/640/1*jXpSVhjWRxgzDDKKVAQR8A.gif" style="display:block; margin-left:auto; margin-right:auto; width:40%;">

<p style="margin-top:1cm">The model uses a Deep Q-Network (DQN) that consists of two convolutional layers and two fully connected layers. The model uses the a Double DQN and Prioritized Experience Replay for training. If you are interested in training the model on an Atari game, please refer to the [Setup](#setup) section to get started.</p>

If you want to create your own DRL model, I recommend that you check out the [Acknowledgements](#acknowledgements) section for some resources to get you pointed in the right direction.

## Technologies

* [Python 3.9.1](https://www.python.org/)
* [PyTorch 1.9.0](https://pytorch.org/)

## Setup

#### Prerequisites

1) [`Python`](https://www.python.org/) must be installed on your device.
2) Make sure you have [`git`](https://git-scm.com/), [`zlib`](https://zlib.net/), and [`cmake`](https://cmake.org/) installed.

Clone this repository and change to the project directory to get started.

Go to the `roms` directory in the project folder and extract the files from the `Roms.rar` file to get the Atari ROMs. The `.rar` file was obtained from [here](https://github.com/openai/atari-py).

#### Windows Setup

```
$ cd Atari-DRL
$ python -m venv venv
$ venv\Scripts\activate
$ pip install ipykernel
$ ipython kernel install --user --name=atari
$ pip install -r requirements.txt
$ python -m atari_py.import_roms roms/
$ jupyter-notebook
```

#### MacOS/Linux Setup

```
$ cd Atari-DRL
$ python -m venv venv
$ source venv/bin/activate
$ pip install ipykernel
$ ipython kernel install --user --name=atari
$ pip install -r requirements.txt
$ python -m atari_py.import_roms roms/
$ jupyter-notebook
```

Open the `sample_code.ipynb` file to get started. Make sure to switch the kernel to `atari` by going to `Kernel > Change kernel > atari` on the jupyter notebook menu bar.

## Acknowledgements

#### Tutorials and Source Code

* [DEEPLIZARD](https://deeplizard.com/learn/playlist/PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv): This website gives a good introduction to Q-learning and Deep Reinforcement Learning. This implementation of a DRL agent was a heavily adapted version of the tutorial offered by this website.

* [jcwleo](https://github.com/rlcode/per): My implementation of prioritized experience replay was based off this source code.

#### Academic Papers

* [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning): This paper was the first to present a deep learning model that successfully trained a DRL agent to play Atari games. It provides a good starting point for implementing a DRL model and offers some data preprocessing techniques.

* [Deep Reinforcement Learning with Double Q-learning](https://deepmind.com/research/publications/deep-reinforcement-learning-double-q-learning): This paper demonstrates the use of a Double Deep Q-Network and provides a mathematical formula for getting the reward subject to the Double Q-learning algorithm.

* [Prioritized Experience Replay](https://deepmind.com/research/publications/prioritized-experience-replay): This paper discusses the concept of using error to prioritize which replay experiences are sampled, creating the notion of prioritized experience replay. This paper provides some methods for implementation.