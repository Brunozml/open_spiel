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

"""RL agents trained against fixed policy/bot as approximate responses.

This can be used to try to find exploits in policies or bots, as described in
Timbers et al. '20 (https://arxiv.org/abs/2004.09677), but only using RL
directly rather than RL+Search.
"""

import logging
from absl import app
from absl import flags
import numpy as np
# import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

import open_spiel.python.games
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import random_agent

from open_spiel.python.algorithms import dqn

FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_test",
                    "Directory to save/load the agent models.")
flags.DEFINE_integer("eval_episodes", 1000,
                     "How many episodes to run per eval.")

# Main algorithm parameters
flags.DEFINE_integer("seed", 0, "Seed to use for everything")
flags.DEFINE_string("game", "python_block_dominoes", "Game string")
flags.DEFINE_integer("num_players", 2, "Numebr of players")
flags.DEFINE_string("player_1", "first", " Player 1 in matchup (random | first | dqn)")
flags.DEFINE_string("player_2", "random", "Player 2 in matchup (random | first | dqn)")

# TODO: ENSURE THIS IS CORRECT
# def eval_head_to_head(env, agents, num_episodes):
#   """Evaluates `agents` against each other for `num_episodes`."""
#   num_players = len(agents)
#   rewards_agent1 = [] # TODO: change after test run to positions in game
#   rewards_agent2 = []

#   logging.info(f"env is turn based: {env.is_turn_based}")
#   for _ in range(num_episodes):
#     flip = _ % 2 == 1 # flip agents and reward assignment every other episode
#     agents = agents[::-1] if flip else agents
#     time_step = env.reset()
#     logging.info(f"Starting episode {_} ")
#     while not time_step.last():
#       player_id = time_step.observations["current_player"]
#       if env.is_turn_based:
#         agent_output = agents[player_id].step(
#             time_step, is_evaluation=True)
#         action_list = [agent_output.action]
#       else:
#         raise NotImplementedError("Only turn-based games are supported.")
#       time_step = env.step(action_list)
#       logging.info(f"Player: {player_id}, action: {action_list}")
    
#     logging.info(f"Episode rewards: {time_step.rewards}")
    
#     # Episode is over, step all agents with final state.
#     for agent in agents:
#       agent.step(time_step)
    
#     if flip:
#       rewards_agent1.append(time_step.rewards[1])
#       rewards_agent2.append(time_step.rewards[0])
#     else:
#       rewards_agent1.append(time_step.rewards[0])
#       rewards_agent2.append(time_step.rewards[1])
      
#   return rewards_agent1, rewards_agent2

def eval_against_random_bots(env, trained_agents, fixed_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(fixed_agents)
  all_episode_rewards = [[] for _ in range(num_players)]
  for player_pos in range(num_players):
    cur_agents = fixed_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      turn_num = 0
      while not time_step.last():
        turn_num += 1
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      all_episode_rewards[player_pos].append(episode_rewards)
  return all_episode_rewards

# TODO: complete this function
def load_agents(num_actions):
  """Load agents from a directory."""
  players = [FLAGS.player_1, FLAGS.player_2]
  agents = []
  for idx in range(len(players)):
    if players[idx] == "random":
      agents.append(random_agent.RandomAgent(player_id=idx, num_actions=num_actions))
    elif players[idx] == "first":
      agents.append(FirstActionAgent(idx, num_actions))
    # elif player == "dqn":
    #   agents.append(dqn.DQN(
    #       session=sess,
    #       player_id=idx,
    #       state_representation_size=info_state_size,
    #       num_actions=num_actions,
    #       discount_factor=0.99,
    #       epsilon_start=0.5,
    #       epsilon_end=0.1,
    #       hidden_layers_sizes=hidden_layers_sizes,
    #       replay_buffer_capacity=FLAGS.replay_buffer_capacity,
    #       batch_size=FLAGS.batch_size))
    else:
      raise RuntimeError("Unknown learner")
  
  return agents


class FirstActionAgent(rl_agent.AbstractAgent):
  """An example agent class."""

  def __init__(self, player_id, num_actions, name="first_action_agent"):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    # Pick the first legal action.
    cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
    action = cur_legal_actions[0]
    probs = np.zeros(self._num_actions)
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)


def main(_):
  np.random.seed(FLAGS.seed)
  # tf.random.set_random_seed(FLAGS.seed)

  num_players = FLAGS.num_players

  env = rl_environment.Environment(FLAGS.game, include_full_state=True)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  agents = load_agents(num_actions)
  print(agents)
  
  # Evaluate the agents
  # r1, r2 = eval_head_to_head(env, agents, FLAGS.eval_episodes)
  # r1, r2 = eval_against_fixed_bots(env, agents, FLAGS.eval_episodes)
  rewards = eval_against_random_bots(env, agents, agents, FLAGS.eval_episodes)
  print(rewards)
  # Plot the rewards
  plt.boxplot(rewards, vert=False)
  plt.title('Boxplot of Rewards')
  plt.xlabel('Rewards')
  plt.yticks(range(1, num_players + 1), ['Player ' + str(i) for i in range(1, num_players + 1)])
  plt.show()




if __name__ == "__main__":
  app.run(main)
