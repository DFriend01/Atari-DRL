import os
import gym
import pdb
import random
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sumtree import SumTree
from collections import namedtuple, OrderedDict
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))
DO_NOTHING = 0
MAX_COLOR_VALUE = 255
SAVED_DATA = 'saved_data'
SAVED_MODELS = 'saved_models'
SAVED_REWARDS = 'saved_rewards'

class DQN(nn.Module):
    
    """
    This class represents a Deep Q-Network with two convolutional layers and 
    two fully connected layers.
    
    Attributes
    ----------
    conv_layers : nn.Sequential
        The convolutional layers used in the DQN.
        
    fc_layers : nn.Sequential
        The fully connected layers used in the DQN.
    """
    
    def __init__(self, action_space, frame_height, frame_width, k_frames):
        """
        Initializes the DQN layers and their weights.
        
        Parameters
        ----------
        action_space : int
            The number of possible actions that the agent can take in its environment.
            action_space > 0
            
        frame_height : int
            The height of the frames that are used as inputs to the DQN.
            frame_height > 0
            
        frame_width : int
            The width of the frames that are used as inputs to the DQN.
            frame_width > 0
            
        k_frames : int
            The number of frames that are stacked in the input.
            k_frames > 1
        """
        assert action_space > 0 and isinstance(action_space, int), 'action_space should be a positive integer'
        assert frame_height > 0 and isinstance(frame_height, int), 'frame_height should be a positive integer'
        assert frame_width > 0 and isinstance(frame_width, int), 'frame_width should be a positive integer'
        assert k_frames > 1 and isinstance(k_frames, int), 'k_frames should be an integer greater than 1'
        
        super().__init__()
        conv1 = nn.Conv2d(in_channels=k_frames, out_channels=32, kernel_size=8, stride=4)
        conv1_height = ((frame_height + 2*conv1.padding[0] - conv1.dilation[0]*(conv1.kernel_size[0] - 1) - 1)\
                        // conv1.stride[0]) + 1
        conv1_width = ((frame_width + 2*conv1.padding[1] - conv1.dilation[1]*(conv1.kernel_size[1] - 1) - 1)\
                        // conv1.stride[1]) + 1
        
        conv2 = nn.Conv2d(in_channels=conv1.out_channels, out_channels=64, kernel_size=4, stride=2)
        conv2_height = ((conv1_height + 2*conv2.padding[0] - conv2.dilation[0]*(conv2.kernel_size[0] - 1) - 1)\
                        // conv2.stride[0]) + 1
        conv2_width = ((conv1_width + 2*conv2.padding[1] - conv2.dilation[1]*(conv2.kernel_size[1] - 1) - 1)\
                        // conv2.stride[1]) + 1
        
        fc1 = nn.Linear(in_features=conv2.out_channels * conv2_height * conv2_width, out_features=512)
        fc2 = nn.Linear(in_features=fc1.out_features, out_features=action_space)
        softmax = nn.Softmax(dim=1)
        
        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv1),
            ('relu1', nn.ReLU()),
            ('conv2', conv2),
            ('relu2', nn.ReLU())
        ]))
        
        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', fc1),
            ('relu3', nn.ReLU()),
            ('fc2', fc2),
            ('output', softmax)
        ]))
        self.fc_layers.apply(self.__init_weights)
        nn.utils.clip_grad_norm_(self.parameters(), 1)
        
    def __init_weights(self, layer):
        """
        Randomizes the weights of a fully connected layer. This method is used recursively
        with the "apply" method provided by the pytorch library.
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, t) -> torch.tensor:
        """
        Sends an input tensor through the hidden layers of the DQN.
        
        Parameters
        ----------
        t : torch.tensor
            An input tensor representing the current state. The input tensor must have
            the following shape: (batch_size, k_frames, frame_height, frame_width) where
            each frame is represented as a grayscale image.
            
        Returns
        -------
        torch.tensor
            A tensor holding the Q values for each possible action in the action space.
            The output shape is (batch_size, action_space).
        """
        t = t.float()
        t = self.conv_layers(t)
        t = t.flatten(start_dim=1)
        t = self.fc_layers(t)
        return t

class ReplayMemory:
    """
    This class represents replay memory that samples uniformly.

    Attributes
    ----------
    capacity : int
        The maximum number of experiences that can be stored in memory.
        capacity > 0

    memory : list
        A list that contains the most recent experiences. It cannot contain
        more experiences than the capacity attribute permits.

    data_ptr : int
        The index of the element in the memory list that will be overwitten on
        the next push to memory. This value is incremented for each push to memory.
        If this value reaches the value for capacity, data_ptr will be set to zero.
        0 <= data_ptr < capacity

    experience_count : int
        The number of experiences currently stored in memory.
        0 <= experience_count <= capacity
    """
    def __init__(self, capacity):
        """
        Initializes the replay memory.
        
        Parameters
        ----------
        capacity : int
            The maximum number of experiences that can be stored in memory.
            capacity > 0
        """
        assert capacity > 0 and isinstance(capacity, int), 'capacity should be a positive integer'

        self.capacity = capacity
        self.memory = self.capacity * [0]
        self.data_ptr = 0
        self.experience_count = 0

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The current number of experiences stored in memory.
        """
        return self.experience_count

    def push(self, experience):
        """
        Adds an experience to replay memory. If the replay memory is full, then the least recently
        added experience is overwritten.
        
        Parameters
        ----------
        experience : namedtuple
            The experience to be added to replay memory.
        """
        self.memory[self.data_ptr] = experience
        self.data_ptr += 1
        if self.experience_count < self.capacity:
            self.experience_count += 1
        if self.data_ptr >= self.capacity:
            self.data_ptr = 0

    def sample(self, batch_size) -> tuple:
        """
        Samples a random batch of experiences. The class method can_provide_sample
        should be called before this function.
        
        Parameters
        ----------
        batch_size : int
            The number of experiences to sample at once.
        
        Returns
        -------
        tuple
            A tuple containing:
                1) The states for the given sampled experiences
                2) The actions taken in each of the sampled states
                3) The rewards observed from taking the action in each state
                4) The next states that are observed
        """
        batch = random.sample(self.memory[:self.experience_count], batch_size)
        experiences = Experience(*zip(*batch))
        states = torch.cat(experiences.state)
        actions = torch.cat(experiences.action)
        rewards = torch.cat(experiences.reward)
        next_states = torch.cat(experiences.next_state)
        return states, actions, rewards, next_states

    def can_provide_sample(self, batch_size) -> bool:
        """
        Determines if a sample of a specific size can be drawn from memory.
        
        Parameters
        ----------
        batch_size : int
            The number of experiences that are requested for sampling.
            
        Returns
        -------
        bool
            Returns true if a sample can be drawn, and false otherwise.
        """
        return self.__len__() >= batch_size


class PrioritizedReplayMemory:
    """
    This class is an implementation of Prioritized Experience Replay. It uses a Sum Tree
    to store the experiences observed by the agent.
    
    Attributes
    ----------
    capacity : int
        The maximum number of experiences that can be stored in memory.
        
    alpha : float
        This is a parameter that quantifies the affect of error on the probability
        distribution that is used to sample experiences. If alpha is 0, the distribution
        is uniform and error does not affect which experiences get sampled. If alpha is 1,
        then error completely determines the probability distribtion of sampling.
        0 <= alpha <= 1
        
    beta : float
        This is a parameter that controls how much prioritization should be applied for a
        given sample. This value is annealed towards 1 as more samples are drawn from memory.
        0 <= beta <= 1
        
    beta_increment : float
        This value linearly anneals beta towards 1. Each time a sample is drawn from memory,
        beta increases by this value.
        beta_increment > 0
        
    eps : float
        A small values added to the error of an experience to ensure that it has a non-zero
        probability that it will be drawn.
        eps > 0
        
    tree : SumTree
        A SumTree object that contains the experiences and is responsible for sampling.
        
    current_size : int
        Represents the number of experiences in memory. Cannot exceed the capacity.
        0 <= current_size <= capacity
    """
    
    def __init__(self, capacity, alpha, beta, beta_increment, eps):
        """
        Initializes the replay memory.
        
        Parameters
        ----------
        capacity : int
            The maximum number of experiences that can be stored in memory.
            capacity > 0
        
        alpha : float
            This is a parameter that quantifies the affect of error on the probability
            distribution that is used to sample experiences. If alpha is 0, the distribution
            is uniform and error does not affect which experiences get sampled. If alpha is 1,
            then error completely determines the probability distribtion of sampling.
            0 <= alpha <= 1

        beta : float
            This is a parameter that controls how much prioritization should be applied for a
            given sample. This value is annealed towards 1 as more samples are drawn from memory.
            0 <= beta <= 1

        beta_increment : float
            This value linearly anneals beta towards 1. Each time a sample is drawn from memory,
            beta increases by this value.
            beta_increment > 0

        eps : float
            A small values added to the error of an experience to ensure that it has a non-zero
            probability that it will be drawn.
            eps > 0
        """
        assert capacity > 0 and isinstance(capacity, int), 'capacity should be a positive integer'
        assert 0. <= alpha <= 1. and isinstance(alpha, float), 'alpha must be a float in the interval [0,1]'
        assert 0. <= beta <= 1. and isinstance(beta, float), 'beta must be a float in the interval [0,1]'
        assert beta_increment > 0. and isinstance(beta_increment, float), 'beta_increment must be a positive float'
        assert eps > 0. and isinstance(eps, float), 'eps must be a positive float'
        
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.tree = SumTree(capacity)
        self.current_size = 0

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            Returns the number of experiences currently stored in replay memory.
        """
        return len(self.tree)
    
    def push(self, experience, policy_net, target_net, done):
        """
        Adds an experience to replay memory. If the replay memory is full, then the least recently
        added experience is overwritten.
        
        Parameters
        ----------
        experience : namedtuple
            The experience to be added to replay memory.
            
        policy_net : DQN
            The policy network.
            
        target_net : DQN
            The target network.
            
        done : bool
            True if this experience is the last experience of an episode,
            and false otherwise.
        """
        current = QValues.get_QValues(policy_net, experience.state, experience.action).item()
        
        with torch.no_grad():
            if done:
                target = experience.reward.item()
            else:
                _, next_action = QValues.get_next(target_net, experience.state)
                next_q = QValues.get_QValues(target_net, experience.next_state, next_action)
                target = next_q.item() + experience.reward.item()
            
        error = np.abs(current - target)
            
        priority = (error + self.eps) ** self.alpha
        self.tree.add(priority, experience)
        
    def sample(self, batch_size) -> tuple:
        """
        Samples a random batch of experiences. The class method can_provide_sample
        should be called before this function.
        
        Parameters
        ----------
        batch_size : int
            The number of experiences to sample at once.
        
        Returns
        -------
        tuple
            A tuple containing:
                1) The states for the given sampled experiences
                2) The actions taken in each of the sampled states
                3) The rewards observed from taking the action in each state
                4) The next states that are observed
                5) The indeces of where each experience is stored in the sum tree
                6) The importance sampling weights for each experience
        """
        batch = []
        idxs = []
        segment_len = self.tree.get_total_priority() / batch_size
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment])
        
        for i in range(batch_size):
            lower = segment_len * i
            upper = segment_len * (i + 1)
            s = np.random.uniform(lower, upper)
            
            (idx, priority, experience) = self.tree.get(s)
            priorities.append(priority)
            batch.append(experience)
            idxs.append(idx)
            
        sampling_probabilities = priorities / self.tree.get_total_priority()
        is_weight = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        experiences = Experience(*zip(*batch))
        states = torch.cat(experiences.state)
        actions = torch.cat(experiences.action)
        rewards = torch.cat(experiences.reward)
        next_states = torch.cat(experiences.next_state)
        return states, actions, rewards, next_states, idxs, is_weight
    
    def can_provide_sample(self, batch_size) -> bool:
        """
        Determines if a sample of a specific size can be drawn from memory.
        
        Parameters
        ----------
        batch_size : int
            The number of experiences that are requested for sampling.
            
        Returns
        -------
        bool
            Returns true if a sample can be drawn, and false otherwise.
        """
        return self.__len__() >= batch_size
    
    def update(self, idx, error):
        """
        Updates the sum tree when a new experience is added.
        
        Parameters
        ----------
        idx : int
            The index of where the new experience is written in the sum tree.
            
        error : float
            The error associated with a given experience.
        """
        priority = (np.abs(error) + self.eps) ** self.alpha
        self.tree.update(idx, priority)

class Agent:
    """
    This class represents the agent. The agent is responsible for making decisions in its
    environment following an epsilon greedy policy.
    
    Attributes
    ----------
    actions_taken : int
        The total number of actions that the agent has taken.
        
    action_space : int
        The total number of possible actions that the agent could take in its environment.
        action_space > 0
        
    eps_start : float
        The starting value for the exploration rate. Usually is 1.0.
        0 <= eps_start <= 1.0
        
    eps_end : float
        The ending value for the exploration rate.
        0 <= eps_end <= eps_start
        
    eps_decay : float
        The value for which the exploration rate is linearly annealed towards eps_end.
        eps_decay >= 0
    """
    def __init__(self, action_space, eps_start, eps_end, eps_decay):
        """
        Initializes the agent class.
        
        Parameters
        ----------
        action_space : int
            The total number of possible actions that the agent could take in its environment.
            action_space > 0
        
        eps_start : float
            The starting value for epsilon in the epsilon greedy policy. Usually is 1.0.
            0 <= eps_start <= 1.0

        eps_end : float
            The ending value for epsilon in the epsilon greedy policy.
            0 <= eps_end <= eps_start
        
        eps_decay : float
            The value for which epsilon is linearlly annealed towards eps_end.
            eps_decay >= 0
        """
        assert action_space > 0 and isinstance(action_space, int), 'action_space must be a positive integer'
        assert 0. <= eps_start <= 1. and isinstance(eps_start, float), 'eps_start must be a float on the interval [0,1]'
        assert 0. <= eps_end <= eps_start and isinstance(eps_end, float), 'eps_end must be a float on the interval [0, eps_start]'
        assert eps_decay >= 0. and isinstance(eps_decay, float), 'eps_decay must be a non-negative float'
        
        self.actions_taken = 0
        self.action_space = action_space
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
    def __get_exploration_rate(self) -> float:
        """
        Calculates the exploration rate that follows linear decay.
            
        Returns
        -------
        float
            The exploration rate.
        """
        return np.max([self.eps_end, self.eps_start - self.actions_taken * self.eps_decay])
    
    def select_action(self, state, policy_net) -> torch.tensor:
        """
        Selects the next action that the agent should take based on the epsilon
        greedy policy.
        
        Parameters
        ----------
        state : torch.tensor
            A tensor containing a stack of grayscale frames representing the current
            state. The input shape should be (k_frames, scaled_height, scaled_width).
            
        policy_net : DQN
            The policy network that is used to exploit the current environment.
            
        Returns
        -------
        torch.tensor
            Returns a tensor with a single integer that contains the action that the
            agent should take.
        """
        rate = self.__get_exploration_rate()
        self.actions_taken += 1
        
        if rate <= np.random.uniform(0,1):
            with torch.no_grad():
                q_values = policy_net(state).squeeze().detach().numpy()
            action = np.argmax(q_values)
        else:
            action = np.random.randint(self.action_space)
            
        return torch.tensor([action], dtype=torch.int64).to(device)

class AtariEnv:
    """
    This class manages the operations that are performed on the Atari environment, provides
    information about the Atari environment, and performs preprocessing on the frames provided
    by the Atari environment for training.
    
    Attributes
    ----------
    env : gym.envs.atari.atari_env
        The environment representing the Atari game.
        
    scale_height : int
        The height that the frames are scaled to during preprocessing.
        scale_height > 0
        
    scale_width : int
        The width that the frames are scaled to during preprocessing.
        scale_width > 0
        
    k_frames : int
        The number of frames that are stack together to form one state.
        k_frames > 1
        
    enable_rendering : bool
        If true, the Atari environment will be rendered during training.
        
    state : list
        A list of arrays that represent each frame for the current state.
        
    prev_frame : np.array
        The previous frame that the agent experienced in the Atari environment.
        
    xform : torchvision.transforms.Compose
        A torchvision object that performs preprocessing on a frame. It first converts the frame
        to a PIL image, converts it to a grayscale image, and finally resizes the frame to a specified
        height and width.
        
    done : bool
        If true, the current state is a terminating state for the current episode.
        
    info : list
        A list that contains information about the Atari environment.
    """
    def __init__(self, gym_env, scale_height, scale_width, k_frames, enable_rendering):
        """
        Initializes the AtariEnv object.
        
        Parameters
        ----------
        gym_env : str
            A string that represents the Atari environment. Go to the following link to see the Atari games
            and their strings: https://gym.openai.com/envs/#atari
            
        scale_height : int
            The height that the frames are scaled to during preprocessing.
            scale_height > 0
        
        scale_width : int
            The width that the frames are scaled to during preprocessing.
            scale_width > 0

        k_frames : int
            The number of frames that are stack together to form one state.
            k_frames > 1

        enable_rendering : bool
            If true, the Atari environment will be rendered during training.
        """
        assert scale_height > 0 and isinstance(scale_height, int), 'scale_height should be a positive integer'
        assert scale_width > 0 and isinstance(scale_width, int), 'scale_width should be a positive integer'
        assert k_frames > 1 and isinstance(k_frames, int), 'k_frames should be an integer greater than 1'
        
        self.env = gym.make(gym_env).unwrapped
        self.scale_height = scale_height
        self.scale_width = scale_width
        self.k_frames = k_frames
        self.enable_rendering = enable_rendering
        self.state = []
        self.prev_frame = np.zeros((self.scale_height, self.scale_width), dtype=np.float32)
        self.xform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.scale_height, self.scale_width)),
            T.Grayscale()
        ])
        self.reset()
        
    def reset(self):
        """
        Resets the Atari environment and the current state.
        """
        self.done = False
        self.info = {}
        self.env.reset()
        self.state = []
        
    def close(self):
        """
        Closes the Atari environment.
        """
        self.env.close()
        
    def render(self, mode='human'):
        """
        Renders the Atari environment.
        
        Parameters
        ----------
        mode : str
            Renders the Atari environment according to a specified mode.
            
            From the Gym Documentation:
            - human: render to the current display or terminal and
              return nothing. Usually for human consumption.
            - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
              representing RGB values for an x-by-y pixel image, suitable
              for turning into a video.
            - ansi: Return a string (str) or StringIO.StringIO containing a
              terminal-style text representation. The text can include newlines
              and ANSI escape sequences (e.g. for colors).
              
        Returns
        -------
        None, np.array, str
            The return value varies depending on the specifed mode. See the above.
        """
        return self.env.render(mode)
    
    def in_initial_state(self) -> bool:
        """
        Returns
        -------
        bool
            Returns true if the Atari is in an initial state, and false otherwise.
        """
        return self.state == []
    
    def get_unscaled_height(self) -> int:
        """
        Returns
        -------
        int
            Returns the height of the unscaled frame from the Atari environment.
        """
        return self.render('rgb_array').shape[0]
    
    def get_unscaled_width(self) -> int:
        """
        Returns
        -------
        int
            Returns the width of the unscaled frame from the Atari environment.
        """
        return self.render('rgb_array').shape[1]
    
    def get_action_space(self) -> int:
        """
        Returns
        -------
        int
            Returns the total number of possible actions that the agent can take in the
            Atari environment.
        """
        return self.env.action_space.n
    
    def get_state(self) -> torch.tensor:
        """
        Returns
        -------
        torch.tensor
            Returns a tensor representing the current state of the Atari environment. It
            contains stacked grayscaled frames to represent one state.
        """
        if self.in_initial_state() or self.done:
            return torch.zeros(self.k_frames, self.scale_height, self.scale_width)
        return torch.tensor(self.state).to(device)
    
    def get_info(self) -> list:
        """
        Returns
        -------
        list
            Returns information about the current state of the Atari environment.
        """
        return self.info
    
    def execute_action(self, action) -> torch.tensor:
        """
        Executes a specified action on the last frame (the k_frame-th frame of the stack).
        The step method in atari gym environments automatically implement frame skipping, 
        so it is not explicitely coded here. Each frame in the state is the difference
        between the current frame and the previous frame.
        
        Parameters
        ----------
        action : torch.tensor
            A tensor with a single integer value that represents the action that the agent
            will execute.
            
        Returns
        -------
        torch.tensor
            Returns a tensor with a single element that represents the reward observed by
            the agent as a result of executing the specified action. 
        """
        reward = 0
        if self.in_initial_state():
            for _ in range(self.k_frames - 1):
                self.__step(DO_NOTHING)
        reward = self.__step(action.item())
        if self.done:
            self.state = []
            self.prev_frame = np.zeros((self.scale_height, self.scale_width), dtype=np.float32)
        return torch.tensor([reward / self.k_frames], device=device)
    
    def __step(self, action) -> torch.tensor:
        """
        Executes a specified action for a few frames and renders the current frame.
        
        Parameters
        ----------
        action : int
            The action to be taken.
            
        Returns
        -------
        torch.tensor
            Returns a tensor that contains a single element representing the reward observed
            by the agent as a result of taking the specified action.
        """
        next_frame, reward, self.done, self.info = self.env.step(action)
        if self.enable_rendering:
            next_frame = self.render('rgb_array')
        next_frame = self.__process_frame(next_frame)
        
        if len(self.state) == self.k_frames:
            del self.state[0]    
        self.state.append(next_frame - self.prev_frame)
        self.prev_frame = next_frame
        return reward
    
    def __process_frame(self, frame) -> np.array:
        """
        Processes a frame to make it appropriate for training. 
        The frame is converted to a grayscale image, its dimensions are scaled
        and the intensity values of the grayscale image are normalized.
        
        Parameters
        ----------
        frame : np.array
            The current frame to be processed.
            
        Returns
        -------
        np.array
            Returns an array representing the modified frame.
        """
        xframe = np.array(self.xform(frame), dtype=np.float32) / MAX_COLOR_VALUE
        return xframe

class QValues:
    """
    This class holds static methods that are used to calculate Q values.
    
    """
    
    @staticmethod
    def get_QValues(policy_net, states, actions) -> torch.tensor:
        """
        Gets the Q values associated with each state-action pair. This method supports mini batching.
        For example, if the mini batch size was 32, then the "states" tensor contains 32 states, and
        the "actions" tensor contains 32 actions. In other words, the size of the first dimension of
        "states" and "actions" should be the same.
        
        Parameters
        ----------
        policy_net : DQN
            The policy network.
            
        states : torch.tensor
            A tensor containing multiple states to be processed.
            
        actions : torch.tensor
            A tensor containing multiple actions associated with each state.
            
        Returns
        -------
        torch.tensor
            A tensor containing the Q value for each state-action pair.
        """
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states) -> torch.tensor:
        """
        Gets the maximum Q values and actions associated with the next states that are observed.
        
        Parameters
        ----------
        target_net : DQN
            The target network.
            
        next_states : torch.tensor
            A tensor containing the next states that are observed.
            
        Returns
        -------
        tuple
            A tuple containing a tensor with the maximum Q values and action associated with
            the maximum Q values.
        """
        max_values, max_actions = torch.max(target_net(next_states), dim=1)
        return max_values, max_actions

class AtariAI:
    
    @staticmethod
    def plot(values):
        """
        Plots the mean reward for each episode.

        Parameters
        ----------
        values : list
            A list containing the mean reward for each episode, where the index of each element
            in the list corresponds to the episode that the mean reward belongs to.
        """
        plt.clf()
        plt.title('Mean Reward vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.plot(values)
        plt.show()
        if is_ipython: display.clear_output(wait=True)

    @staticmethod
    def save_rewards(filename, rewards):
        """
        Saves the total rewards for each episode on disk.

        Parameters
        ----------
        filename : str
            The name of the file.

        rewards : list
            A list containing the mean rewards for each episode.
        """
        if not os.path.exists(os.path.join(SAVED_DATA, SAVED_REWARDS)):
            os.makedirs(os.path.join(SAVED_DATA, SAVED_REWARDS))

        reward_data = pd.DataFrame(rewards, columns=['Reward'])
        reward_data.to_csv(os.path.join(SAVED_DATA, SAVED_REWARDS, filename + '.csv'), index=False)
            
    @staticmethod
    def save_state_dict(gym_env, target_net, scaled_height, scaled_width, k_frames):
        """
        Saves the target network's state dictionary and input shape on disk.
        The model is saved to the 'saved_models' directory.

        Parameters
        ----------
        gym_env : str
            The name of the Atari game environment that the model is trained in.

        target_net : DQN
            The target network containing the state dictionary to be saved.

        scaled_height : int
            The height of the frames that are used as inputs to the DQN.
            frame_height > 0
            
        scaled_width : int
            The width of the frames that are used as inputs to the DQN.
            frame_width > 0
            
        k_frames : int
            The number of frames that are stacked in the input.
            k_frames > 1
        """
        if not os.path.exists(os.path.join(SAVED_DATA, SAVED_MODELS)):
            os.makedirs(os.path.join(SAVED_DATA, SAVED_MODELS))
            
        dqn_data = pd.DataFrame([[scaled_height, scaled_width, k_frames]], 
                                columns=['scaled_height', 'scaled_width', 'k_frames'])
        dqn_data.to_csv(os.path.join(SAVED_DATA, SAVED_MODELS, gym_env + '.csv'), index=False)
        torch.save(target_net.state_dict(), os.path.join(SAVED_DATA, SAVED_MODELS, gym_env + '.pth'))

    def train_DQN(gym_env,
              scaled_height=84,
              scaled_width=84,
              k_frames=4,
              memory_size=10000,
              greedy_start=1.,
              greedy_end=0.01,
              greedy_decay=1e-5,
              num_episodes=100,
              max_timesteps=200,
              discount=0.99,
              batch_size=32,
              target_update=10,
              optim_lr=2.5e-4,
              optim_eps=1e-2,
              render=False,
              plot_reward=False,
              save_rewards=False,
              save_model=False
    ):
        """
        This function trains an agent to play an Atari game using a DQN and returns
        the target network.
        """
        atarienv = AtariEnv(gym_env, scaled_height, scaled_width, k_frames, render)
        agent = Agent(atarienv.get_action_space(), greedy_start, greedy_end, greedy_decay)
        memory = ReplayMemory(memory_size)
        
        policy_net = DQN(atarienv.get_action_space(), scaled_height, scaled_width, k_frames)
        target_net = DQN(atarienv.get_action_space(), scaled_height, scaled_width, k_frames)
        target_net.load_state_dict(policy_net.state_dict())

        policy_net.train()
        target_net.eval()
        
        optimizer = optim.Adam(params=policy_net.parameters(), lr=optim_lr, eps=optim_eps)
        loss_fcn = nn.MSELoss(reduction='none')
        
        episode_rewards = []
        for episode in range(num_episodes):
            #Initialize starting state
            total_reward = 0
            atarienv.reset()
            state = atarienv.get_state().unsqueeze(0)
            
            for timestep in range(max_timesteps):
                #Select an action to take
                action = agent.select_action(state, policy_net)
                
                #Execute the action, the observe the next state and reward
                reward = atarienv.execute_action(action)
                next_state = atarienv.get_state().unsqueeze(0)
                total_reward += reward.item()
                
                #Store the experience in replay memory
                memory.push(Experience(state, action, next_state, reward))
                state = next_state
                
                #Sample random batch from replay memory and learn
                if memory.can_provide_sample(batch_size):
                    states, actions, rewards, next_states = memory.sample(batch_size)
                    
                    current_q_values = QValues.get_QValues(policy_net, states, actions)
                    if (timestep == max_timesteps - 1) or (atarienv.done):
                        target_q_values = rewards
                    else:
                        next_q_values, _ = QValues.get_next(target_net, next_states)
                        target_q_values = (next_q_values * discount) + rewards.float()

                    #Calculate the loss (Mean Squared Error)
                    loss = loss_fcn(current_q_values.float(), target_q_values.unsqueeze(1).float()).float().mean()

                    #Perform gradient descent
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if timestep == max_timesteps - 1 or atarienv.done:
                    episode_rewards.append(total_reward)
                if atarienv.done:
                    break
        
            #Update the target network every "target_update" episodes.
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                target_net.eval()
                if plot_reward:
                    AtariAI.plot(episode_rewards)
                    
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        
        if plot_reward:
            AtariAI.plot(episode_rewards)
        if save_rewards:
            AtariAI.save_rewards(gym_env + '_DQN', episode_rewards)
        if save_model:
            AtariAI.save_state_dict(gym_env, target_net, scaled_height, scaled_width, k_frames)
            
        atarienv.close()
        return target_net
    
    @staticmethod
    def train_DDQN_PER(gym_env,
              scaled_height=84,
              scaled_width=84,
              k_frames=4,
              memory_size=10000,
              memory_alpha=0.7,
              memory_beta=0.4,
              memory_beta_increment=1e-5,
              memory_eps=1e-2,
              greedy_start=1.,
              greedy_end=0.01,
              greedy_decay=1e-5,
              num_episodes=100,
              max_timesteps=200,
              discount=0.99,
              batch_size=32,
              target_update=10,
              optim_lr=2.5e-4,
              optim_eps=1e-2,
              render=False,
              plot_reward=False,
              save_rewards=False,
              save_model=False
             ):

        """
        This function trains an agent to play an Atari game using a DDQN 
        prioritized experience replay. The target network is returned.
        """
        atarienv = AtariEnv(gym_env, scaled_height, scaled_width, k_frames, render)
        agent = Agent(atarienv.get_action_space(), greedy_start, greedy_end, greedy_decay)
        memory = PrioritizedReplayMemory(memory_size, memory_alpha, memory_beta, memory_beta_increment, memory_eps)
        
        policy_net = DQN(atarienv.get_action_space(), scaled_height, scaled_width, k_frames)
        target_net = DQN(atarienv.get_action_space(), scaled_height, scaled_width, k_frames)
        target_net.load_state_dict(policy_net.state_dict())

        policy_net.train()
        target_net.eval()
        
        optimizer = optim.Adam(params=policy_net.parameters(), lr=optim_lr, eps=optim_eps)
        loss_fcn = nn.MSELoss(reduction='none')
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            #Initialize starting state
            total_reward = 0
            atarienv.reset()
            state = atarienv.get_state().unsqueeze(0)
            
            for timestep in range(max_timesteps):
                #Select an action to take
                action = agent.select_action(state, policy_net)
                
                #Execute the action, the observe the next state and reward
                reward = atarienv.execute_action(action)
                next_state = atarienv.get_state().unsqueeze(0)
                total_reward += reward.item()
                
                #Store the experience in replay memory
                memory.push(Experience(state, action, next_state, reward), policy_net, target_net, atarienv.done)
                state = next_state
                
                #Sample random batch from replay memory and learn
                if memory.can_provide_sample(batch_size):
                    states, actions, rewards, next_states, idxs, is_weights = memory.sample(batch_size)
                    
                    current_q_values = QValues.get_QValues(policy_net, states, actions)
                    if (timestep == max_timesteps - 1) or (atarienv.done):
                        target_q_values = rewards
                    else:
                        _, next_actions = QValues.get_next(policy_net, next_states)
                        next_q_values = QValues.get_QValues(target_net, next_states, next_actions).squeeze(1)
                        target_q_values = (next_q_values * discount) + rewards.float()
                        
                    errors = list(torch.abs(current_q_values - target_q_values.unsqueeze(1)).detach().numpy().squeeze())
                    for idx, error in zip(idxs, errors):
                        memory.update(idx, error)

                    #Calculate the loss (Mean Squared Error)
                    loss = (torch.FloatTensor(is_weights).unsqueeze(1) * \
                            loss_fcn(current_q_values.float(), target_q_values.unsqueeze(1).float()).float()).mean()

                    #Perform gradient descent
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if timestep == max_timesteps - 1 or atarienv.done:
                    episode_rewards.append(total_reward)
                if atarienv.done:
                    break
        
            #Update the target network every "target_update" episodes.
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                target_net.eval()
                if plot_reward:
                    AtariAI.plot(episode_rewards)
                    
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        
        if plot_reward:
            AtariAI.plot(episode_rewards)
        if save_rewards:
            AtariAI.save_rewards(gym_env + '_DDQN_PER', episode_rewards)
        if save_model:
            AtariAI.save_state_dict(gym_env, target_net, scaled_height, scaled_width, k_frames)
            
        atarienv.close()
        return target_net
    
    @staticmethod
    def test(gym_env, num_episodes, max_timesteps):
        assert os.path.exists(os.path.join(SAVED_DATA, SAVED_MODELS, gym_env + '.pth')) and \
            os.path.exists(os.path.join(SAVED_DATA, SAVED_MODELS, gym_env + '.csv')), \
            'The state dictionary or csv file for ' + gym_env + ' is not saved on disk'
        
        frame_data = pd.read_csv(os.path.join(SAVED_DATA, SAVED_MODELS, gym_env + '.csv'))
        scaled_height = frame_data['scaled_height'].values.item()
        scaled_width = frame_data['scaled_width'].values.item()
        k_frames = frame_data['k_frames'].values.item()
        
        atarienv = AtariEnv(gym_env, scaled_height, scaled_width, k_frames, enable_rendering=True)
        
        policy_net = DQN(atarienv.get_action_space(), scaled_height, scaled_width, k_frames)
        policy_state_dict = torch.load(os.path.join(SAVED_DATA, SAVED_MODELS, gym_env + '.pth'))
        policy_net.load_state_dict(policy_state_dict)
        policy_net.eval()
        
        for _ in range(num_episodes):
            atarienv.reset()
            state = atarienv.get_state().unsqueeze(0)
            for _ in range(max_timesteps):
                #Select an action to take
                with torch.no_grad():
                    action = torch.tensor([np.argmax(policy_net(state).squeeze().detach().numpy())], 
                                          dtype=torch.int64).to(device)
                
                #Execute the action
                atarienv.execute_action(action)
                next_state = atarienv.get_state().unsqueeze(0)
                state = next_state
                
                if atarienv.done:
                    break
