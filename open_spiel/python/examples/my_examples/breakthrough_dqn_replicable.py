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

"""DQN agents trained on Breakthrough by independent Q-learning."""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_test",
                    "Directory to save/load the agent models.")
flags.DEFINE_integer(
    "save_every", int(1e3),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(50_000),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 100,
    "Episode frequency at which the DQN agents are evaluated.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.") # i.e. can store 100,000 integers in the replay buffer
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players): # player_pos is the position of the trained agent. we alternate between the two players.
    cur_agents = random_agents[:] # Copy the random agents.
    cur_agents[player_pos] = trained_agents[player_pos] # Replace with trained agent.
    for _ in range(num_episodes): 
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else: # In simultaneous move games, we need to pass all agent outputs to the environment.
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos] # Add the reward of the trained agent ONLY
      sum_episode_rewards[player_pos] += episode_rewards

  return sum_episode_rewards / num_episodes

def plot_rewards(rewards):
        def format_func(value, tick_number):
          if value > 0:
            # find number of multiples of 10
            exp = int(np.log10(value))
            return f'10^{exp}'
          else:
            return '0'
        fig = plt.figure()
        iterations = [i * FLAGS.eval_every for i in range(len(rewards))]
        plt.plot(iterations, [r[0] for r in rewards], color='blue', label='Starting agent')  # First reward in blue
        plt.plot(iterations, [r[1] for r in rewards], color='red', label='Agent 2')  # Second reward in red
        plt.xlabel('Training Iterations')
        plt.ylabel('Reward')
        plt.title(f'{FLAGS.game}: Reward vs Training Iterations')
        plt.legend()
        # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        plt.savefig(f'open_spiel/python/examples/saved_examples/{FLAGS.game}_reward_plot_dqn.png')
        plt.close(fig)  # Close the figure

def main(_):
  game = "kuhn_poker"
  num_players = 2
  # add a list to store rewards
  rewards = []


  # env_configs = {"columns": 5, "rows": 5}
  env = rl_environment.Environment(game) # , **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  with tf.Session() as sess:
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    # pylint: disable=g-complex-comprehension
    agents = [
        dqn.DQN(
            session=sess,
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            batch_size=FLAGS.batch_size) for idx in range(num_players)
    ]
    sess.run(tf.global_variables_initializer())

    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
        rewards.append(r_mean)
        logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)

        # plot the rewards
        plot_rewards(rewards)

      if (ep + 1) % FLAGS.save_every == 0:
        for agent in agents:
          agent.save(FLAGS.checkpoint_dir)

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = agents[player_id].step(time_step)
          action_list = [agent_output.action]
        else:
          agents_output = [agent.step(time_step) for agent in agents]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)

if __name__ == "__main__":
  app.run(main)
