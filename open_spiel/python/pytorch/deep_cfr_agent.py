# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN agent implemented in PyTorch. WORK IN PROGRESS"""

import collections
import math
import random
import sys
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F

from open_spiel.python import rl_agent
from open_spiel.python.utils.replay_buffer import ReplayBuffer

AdvantageMemory = collections.namedtuple(
    "AdvantageMemory", "info_state iteration advantage action")

StrategyMemory = collections.namedtuple(
    "StrategyMemory", "info_state iteration strategy_action_probs")


class SonnetLinear(nn.Module):
  """A Sonnet linear module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, in_size, out_size, activate_relu=True):
    """Creates a Sonnet linear layer.

    Args:
      in_size: (int) number of inputs
      out_size: (int) number of outputs
      activate_relu: (bool) whether to include a ReLU activation layer
    """
    super(SonnetLinear, self).__init__()
    self._activate_relu = activate_relu
    self._in_size = in_size
    self._out_size = out_size
    # stddev = 1.0 / math.sqrt(self._in_size)
    # mean = 0
    # lower = (-2 * stddev - mean) / stddev
    # upper = (2 * stddev - mean) / stddev
    # # Weight initialization inspired by Sonnet's Linear layer,
    # # which cites https://arxiv.org/abs/1502.03167v3
    # # pytorch default: initialized from
    # # uniform(-sqrt(1/in_features), sqrt(1/in_features))
    self._weight = None
    self._bias = None
    self.reset()

  def forward(self, tensor):
    y = F.linear(tensor, self._weight, self._bias)
    return F.relu(y) if self._activate_relu else y

  def reset(self):
    stddev = 1.0 / math.sqrt(self._in_size)
    mean = 0
    lower = (-2 * stddev - mean) / stddev
    upper = (2 * stddev - mean) / stddev
    # Weight initialization inspired by Sonnet's Linear layer,
    # which cites https://arxiv.org/abs/1502.03167v3
    # pytorch default: initialized from
    # uniform(-sqrt(1/in_features), sqrt(1/in_features))
    self._weight = nn.Parameter(
        torch.Tensor(
            stats.truncnorm.rvs(
                lower,
                upper,
                loc=mean,
                scale=stddev,
                size=[self._out_size, self._in_size])))
    self._bias = nn.Parameter(torch.zeros([self._out_size]))


class MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
    """

    super(MLP, self).__init__()
    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
      input_size = size
    # Output layer
    self._layers.append(
        SonnetLinear(
            in_size=input_size,
            out_size=output_size,
            activate_relu=activate_final))

    self.model = nn.ModuleList(self._layers)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x

  def reset(self):
    for layer in self._layers:
      layer.reset()


class ReservoirBuffer(object):
  """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.
  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

  def __init__(self, reservoir_buffer_capacity):
    self._reservoir_buffer_capacity = reservoir_buffer_capacity
    self._data = []
    self._add_calls = 0

  def add(self, element):
    """Potentially adds `element` to the reservoir buffer.

    Args:
      element: data to be added to the reservoir buffer.
    """
    if len(self._data) < self._reservoir_buffer_capacity:
      self._data.append(element)
    else:
      idx = np.random.randint(0, self._add_calls + 1)
      if idx < self._reservoir_buffer_capacity:
        self._data[idx] = element
    self._add_calls += 1

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.
    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)

class DeepCFR(rl_agent.AbstractAgent):
    """
    Deep CFR agent implementation in PyTorch.
    """

    def __init__(self,
                 player_id,
                 num_actions,
                 policy_network_layers=(256, 256),
                 advantage_network_layers=(128, 128),
                 num_iterations: int = 100, # CANDIDATE TO BE REMOVED
                 num_traversals: int = 20, # NOTE : CANDIDATE TO BE REMOVED
                 learning_rate: float = 1e-4,
                 batch_size_advantage=None,
                 batch_size_strategy=None,
                 memory_capacity: int = int(1e6),
                 policy_network_train_steps: int = 1,
                 advantage_network_train_steps: int = 1,
                 reinitialize_advantage_networks: bool = True
                 ):
        """Initialize the Deep CFR agent.
            policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
            advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
            num_iterations: (int) Number of training iterations.
            num_traversals: (int) Number of traversals per iteration.
            learning_rate: (float) Learning rate.
            batch_size_advantage: (int or None) Batch size to sample from advantage
                memories.
            batch_size_strategy: (int or None) Batch size to sample from strategy
                memories.
            memory_capacity: Number af samples that can be stored in memory.
            policy_network_train_steps: Number of policy network training steps (per
                iteration).
            advantage_network_train_steps: Number of advantage network training steps
                (per iteration).
            reinitialize_advantage_networks: Whether to re-initialize the advantage
                network before training on each iteration.
    """
    
        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()

        self._player_id = player_id
        self._num_actions = num_actions
        ...

    def step(self, time_step, is_evaluation=False):
        """Returns action probabilities and chosen action at `time_step`.

        Args:
            time_step: an instance of rl_environment.TimeStep.
            is_evaluation: bool indicating whether the step is an evaluation routine,
                as opposed to a normal training step.

        Returns:
            A `rl_agent.StepOutput` for the current `time_step` containing the action probs and chosen action.
        """
        if (not time_step.last()) and self._player_id == time_step.current_player:
            # info_state = time_step.observations["info_state"][self._player_id]
            # legal_actions = time_step.observations["legal_actions"][self._player_id]
            # action, probs = self._act(info_state, legal_actions, is_evaluation)
            # return rl_agent.StepOutput(action=action, probs=probs)
            ...
        else : # If it is not the agent's turn, return an empty StepOutput
          action = None
          probs = []

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
          self._step_counter += 1

          if self._step_counter % self._update_every == 0:
            self._update()
        
        ...
        return
    
    def save(self, data_path, optimizer_data_path = None):
        """Save the model to a file.

        Args:
            data_path: (str) Path to save the model.
            optimizer_data_path: (str) Path to save the optimizer state.
        """
        torch.save(self._policy_net.state_dict(), data_path)
        if optimizer_data_path is not None:
            torch.save(self._optimizer.state_dict(), optimizer_data_path)
        return
    
    def load(self, data_path, optimizer_data_path = None):
        """Load the model from a file.

        Args:
            data_path: (str) Path to load the model.
            optimizer_data_path: (str) Path to load the optimizer state.
        """
        self._policy_net.load_state_dict(torch.load(data_path))
        if optimizer_data_path is not None:
            self._optimizer.load_state_dict(torch.load(optimizer_data_path))
        return # questionable implementation
