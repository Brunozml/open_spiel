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

from absl import app
from absl import flags
from absl import logging
import numpy as np
from datetime import datetime

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
import pyspiel
from open_spiel.python.pytorch import deep_cfr
from open_spiel.python.bots import human
import open_spiel.python.games

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 5_000, "Number of iterations")
flags.DEFINE_integer("num_traversals", 100, "Number of traversals/games")
_GAME_STRING = flags.DEFINE_string("game_name", "python_block_dominoes", "Name of the game")
_TABULAR = flags.DEFINE_bool("is_tabular", False, "game is tabular")
                             

def play_game(state: pyspiel.State,
              bots: dict):
  """Play the game via console."""

  while not state.is_terminal():
    print(f"State: \n{state}\n")
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      outcome = np.random.choice(action_list, p=prob_list)
      # print(f"Chance chose: {outcome} ({state.action_to_string(outcome)})")
      state.apply_action(outcome)
    else:
      player = state.current_player()
      curr_bot = bots[player]
      if curr_bot['type'] == 'human':
        action = curr_bot['bot'].step(state)
        print(f"Chose action: {action} ({state.action_to_string(action)})")
      elif curr_bot['type'] == 'cfr':
        action_probs = curr_bot['bot'].action_probabilities(state)
        for action, probability in action_probs.items():
          print(f"Action: {state.action_to_string(action)}, Probability: {probability}")
        action = max(action_probs, key=action_probs.get)
        print(f"Chose action: {action} ({state.action_to_string(action)})")
      
      state.apply_action(action)



  print("\n-=- Game over -=-\n")
  print(f"Terminal state:\n{state}")
  print(f"Returns: {state.returns()}")
  return

def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game_name)


  deep_cfr_solver = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=(64, 64),
      advantage_network_layers=(32, 32),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=1e-3,
      batch_size_advantage=None,
      batch_size_strategy=None,
      memory_capacity=int(1e8))

  # solving the game using deep cfr
  _, advantage_losses, policy_loss = deep_cfr_solver.solve()
  for player, losses in advantage_losses.items():
    logging.info("Advantage for player %d: %s", player,
                 losses[:2] + ["..."] + losses[-2:])
    logging.info("Advantage Buffer Size for player %s: '%s'", player,
                 len(deep_cfr_solver.advantage_buffers[player]))
  logging.info("Strategy Buffer Size: '%s'",
               len(deep_cfr_solver.strategy_buffer))
  logging.info("Final policy loss: '%s'", policy_loss)

  # find the NashConv (exploitability) of the game; applies only to tabular policies
  if _TABULAR.value:
    average_policy = policy.tabular_policy_from_callable(
        game, deep_cfr_solver.action_probabilities)
    pyspiel_policy = policy.python_policy_to_pyspiel_policy(average_policy)
    conv = pyspiel.nash_conv(game, pyspiel_policy)
    logging.info("Deep CFR in '%s' - NashConv: %s", FLAGS.game_name, conv)

    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    logging.info("Computed player 0 value: %.2f (expected: %.2f).",
                average_policy_values[0], -1 / 18)
    logging.info("Computed player 1 value: %.2f (expected: %.2f).",
                average_policy_values[1], 1 / 18)
  
  # save the trained policy
  now_str = datetime.now().strftime("%d-%m-%yh%H")
  deep_cfr_solver.save(f'open_spiel/python/examples/saved_examples/agents/CFRsolver{now_str}.pkl')

  bots = bots = [
    {'type': 'human', 'index': 0, 'bot': human.HumanBot()},
    {'type': 'cfr', 'index': 1, 'bot': deep_cfr_solver}
  ]

  while True:
      # play the game using the trained policy
      game = pyspiel.load_game(_GAME_STRING.value)
      state = game.new_initial_state()
      play_game(state, bots)
      user_input = input("Press any key to play again or 'q' to quit: ")
      if user_input.lower() == 'q':
          break



if __name__ == "__main__":
  app.run(main)
