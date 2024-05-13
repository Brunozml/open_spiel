"""DQN agents trained on block dominoes by independent Q-learning using pytorch.

uses the `rl_environment.Environment` class to interact with the game.
"""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import matplotlib.pyplot as plt
import sys

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms import random_agent
import pyspiel

import open_spiel.python.games
import math
import os
import torch

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "python_block_dominoes", "Name of the game")
flags.DEFINE_string("checkpoint_dir", "/Users/brunozorrilla/Documents/GitHub/open_spiel/open_spiel/python/examples/agents/dqn_pytorch/q_network", # must be completed with .pt
                    "Directory to save/load the agent.") 
flags.DEFINE_integer(
    "save_every", int(1e1),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e3),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 100,
    "Episode frequency at which the DQN agents are evaluated.")
flags.DEFINE_boolean("interactive", False, "Whether to allow interactive play after training.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")

def eval_against_random_bots(env, trained_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    # random agents for evaluation
  num_actions = env.action_spec()["num_actions"]
  num_players = env.num_players
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
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
      sum_episode_rewards[player_pos] += episode_rewards
  
  return sum_episode_rewards / num_episodes


def main(_):
  num_players = 2
  game = pyspiel.load_game(FLAGS.game_name)
  env = rl_environment.Environment(game=game)
  num_actions = game.num_distinct_actions()
  info_state_size = game.information_state_tensor_size()

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  agents = [
    dqn.DQN(
        player_id=idx,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=hidden_layers_sizes,
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        batch_size=FLAGS.batch_size) for idx in range(num_players)
  ]
  # check if agents have been trained before

  for i in range(len(agents)):
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir + f'{i}.pt')
    if os.path.exists(checkpoint_path):
      print(f"Checkpoint {checkpoint_path} exists.")
      agents[i].load(checkpoint_path) # buggy
      # agents[i]._q_network  = torch.load(checkpoint_path)

    else:
      print(f"Checkpoint {checkpoint_path} does not exist.")

  for ep in range(FLAGS.num_train_episodes):
    if (ep + 1) % FLAGS.eval_every == 0:
      r_mean = eval_against_random_bots(env, agents, 1)
      logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)

    if (ep + 1) % FLAGS.save_every == 0:
        for i in range(len(agents)):
            agents[i].save(FLAGS.checkpoint_dir + f'{i}.pt')

    time_step = env.reset()
    while not time_step.last():
        player = time_step.observations["current_player"]
        agent_output = agents[player].step(time_step)
        time_step = env.step([agent_output.action])
        # print (f"Player: {player}, action: {agent_output.action}")

    # Episode is over, step all agents with final info state.
    for agent in agents:
        agent.step(time_step)

    # print("\n-=- Game over -=-\n")
    # print(f"Terminal state: {time_step}")
    # print(f"Returns: {time_step.rewards}")
  return


if __name__ == "__main__":
  app.run(main)
