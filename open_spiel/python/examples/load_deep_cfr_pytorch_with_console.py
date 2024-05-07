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

flags.DEFINE_integer("num_iterations", 1, "Number of iterations")
flags.DEFINE_integer("num_traversals", 1, "Number of traversals/games")
_GAME_STRING = flags.DEFINE_string("game_name", "python_block_dominoes", "Name of the game")
_CFR_SOLVER = flags.DEFINE_string("cfr_solver", "open_spiel/python/examples/saved_examples/agents/CFRsolver.pkl", "Path to the CFR solver")

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

  solver = deep_cfr.DeepCFRSolver.load(_CFR_SOLVER.value)
  bots = bots = [
    {'type': 'human', 'index': 0, 'bot': human.HumanBot()},
    {'type': 'cfr', 'index': 1, 'bot': solver}
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
