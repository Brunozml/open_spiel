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

"""Python Deep CFR example."""

from typing import Union
import numpy as np
import pandas as pd
import collections
import time
from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel
import open_spiel.python.games

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 200, "Number of iterations")
flags.DEFINE_integer("num_traversals", 10, "Number of traversals/games")
flags.DEFINE_string("game_name", "python_block_dominoes", "Name of the game")

flags.DEFINE_integer("eval_every", 10,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 50, "Episode frequency at which the agents are saved.")
flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir",  "/tmp/deepcfr_dominoes", # "open_spiel/python/examples/agents/nfsp",
                    "Directory to save/load the agent.") #
flags.DEFINE_string("results_dir", "open_spiel/python/examples/block_dominoes/results/train/", 
                    "Directory to save the data.")

def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  game_name = FLAGS.game_name
  game = pyspiel.load_game(FLAGS.game_name)
  df = pd.DataFrame({})
  alg_name = "deep_cfr_tf"

  with tf.Session() as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=(16,),
        advantage_network_layers=(16,),
        num_iterations=FLAGS.num_iterations,
        num_traversals=FLAGS.num_traversals,
        learning_rate=1e-3,
        batch_size_advantage=128,
        batch_size_strategy=1024,
        memory_capacity=1e7,
        policy_network_train_steps=400,
        advantage_network_train_steps=20,
        reinitialize_advantage_networks=False)
    sess.run(tf.global_variables_initializer())

    bots = [
        {'type': 'random', 'bot': None},
        {'type': 'cfr', 'bot': deep_cfr_solver}
    ]

    # if FLAGS.use_checkpoints:
    #   if deep_cfr_solver.has_checkpoint(FLAGS.checkpoint_dir):
    #     deep_cfr_solver.restore(FLAGS.checkpoint_dir)
    #     logging.info("Loaded checkpoint from '%s'", FLAGS.checkpoint_dir)
    #   else:
    #     logging.info("No checkpoint found in '%s'", FLAGS.checkpoint_dir)

    # Modified deep-cfr solution logic for online policy evaluation
    advantage_losses = collections.defaultdict(list)
    for ep in range(deep_cfr_solver._num_iterations):
      start_time = time.time()
      #solve(deep_cfr_solver) #iterate over the training loop

      for p in range(deep_cfr_solver._num_players):
        for _ in range(deep_cfr_solver._num_traversals):
          deep_cfr_solver._traverse_game_tree(
            deep_cfr_solver._root_node, p)
        if deep_cfr_solver._reinitialize_advantage_networks:
            # Re-initialize advantage network for player and train from scratch.
          deep_cfr_solver.reinitialize_advantage_network(p)
        advantage_losses[p].append(
          deep_cfr_solver._learn_advantage_network(p))

      # Train policy network.
      policy_loss = deep_cfr_solver._learn_strategy_network()
      deep_cfr_solver._iteration += 1


      ep_time = time.time() - start_time
      # logging.info("Iteration %s took %s seconds", ep, ep_time)
      if ep % FLAGS.eval_every == 0: # evaluation rounds
        r_mean = eval_against_random_bots(game, bots, 5000)
        logging.info("[%s] Mean episode rewards %s", ep, r_mean)
        df = pd.concat([df, pd.DataFrame(log_info(ep, r_mean, ep_time))], ignore_index=True)
        df.to_csv(FLAGS.results_dir + f"{alg_name}_{game_name}_{FLAGS.num_iterations}.csv")

      if FLAGS.use_checkpoints and ep % FLAGS.save_every == 0:
        deep_cfr_solver.save(FLAGS.checkpoint_dir)
        logging.info("Saved checkpoint to '%s'", FLAGS.checkpoint_dir)

    average_policy = policy.tabular_policy_from_callable(
        game, deep_cfr_solver.action_probabilities)

    expl = exploitability.exploitability(game, average_policy)
    logging.info("Deep CFR in '%s' - expl: %s", FLAGS.game_name, expl)

    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    print("Computed player 0 value: {}".format(average_policy_values[0]))
    print("Expected player 0 value: {}".format(-1 / 18))
    print("Computed player 1 value: {}".format(average_policy_values[1]))
    print("Expected player 1 value: {}".format(1 / 18))


# auxilary functions
# plot it and save it
def log_info(ep, r_mean, ep_time) -> dict[str, list[Union[float, str]]]:
  return {
    "Iteration": [ep+1],
    "Rewards": [r_mean],
    "Time": [ep_time]
  }

def solve(self):
    """Modified deep-cfr solution logic for online policy evaluation"""
    advantage_losses = collections.defaultdict(list)
    for _ in range(self._num_iterations):
        for p in range(self._num_players):
            for _ in range(self._num_traversals):
                self._traverse_game_tree(self._root_node, p)
            if self._reinitialize_advantage_networks:
                # Re-initialize advantage network for player and train from scratch.
                self.reinitialize_advantage_network(p)
            advantage_losses[p].append(self._learn_advantage_network(p))
        self._iteration += 1
    # Train policy network.
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss


def eval_against_random_bots(game, bots, num_episodes):
    """Evaluates `trained` cfr agent against `random agents`"""
    num_players = len(bots)
    sum_episode_rewards = np.zeros(num_players)
    for episode in range(num_episodes):
        bots, cfr_player_index = alternate_starting_player(bots, episode)
        # print(episode, bots, cfr_player_index)
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                _apply_chance_node_action(state)
            else:
                # state = apply_player_action(state, bots)
                player = state.current_player()
                curr_bot = bots[player]  # Use the current player to select the bot
                action = _get_bot_action(state, curr_bot)
                state.apply_action(action)
                # if FLAGS.verbose:
                #     print(f"Player {curr_bot['type']:<10} [{player:<2}] chose action: {action:<5} ({state.action_to_string(action):<20})")
        sum_episode_rewards[cfr_player_index] += state.returns()[cfr_player_index]  # Use the current player to update the rewards
        # if FLAGS.verbose:
        #     _print_game_over_info(state)
    return sum_episode_rewards / num_episodes
 

def alternate_starting_player(bots, episode):
    """Alternate the starting player"""
    bots = bots[::-1]
    cfr_player_index = next(i for i, bot in enumerate(bots) if bot['type'] == 'cfr')
    return bots, cfr_player_index 

def _apply_chance_node_action(state):
    outcomes = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes)
    outcome = np.random.choice(action_list, p=prob_list)
    state.apply_action(outcome)

def _get_bot_action(state, curr_bot):
    """Get action based on bot type"""
    if curr_bot['type'] == 'random':
        return np.random.choice(state.legal_actions())
    elif curr_bot['type'] == 'cfr':
        action_probs = curr_bot['bot'].action_probabilities(state)
        # actions = list(action_probs.keys())
        # probabilities = list(action_probs.values())
        # action = np.random.choice(actions, p=probabilities)
        action = max(action_probs, key=action_probs.get)
        # if FLAGS.verbose:
        #     _print_bot_action_probabilities(curr_bot, state, action_probs)
        return action

def _print_bot_action_probabilities(curr_bot, state, action_probs):
    """Print bot action probabilities"""
    print(f"--- {curr_bot['type']} action probabilities ----")
    for action, probability in action_probs.items():
        print(f"    Action: {state.action_to_string(action)}, Probability: {probability}")
    print(f"--- ------------------- ----")

def _print_game_over_info(state):
    """Print game over information"""
    print("\n-=- Game over -=-\n")
    print(f"Terminal state:\n{state}")
    print(f"Returns: {state.returns()}")


if __name__ == "__main__":
  app.run(main)
