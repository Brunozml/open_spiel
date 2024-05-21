"""CFR agents trained on tiny version of block dominoes, using jax implementation."""

from absl import app
from absl import flags
from absl import logging

import open_spiel.python.games
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
import pyspiel
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "python_block_dominoes", "Name of the game.")
flags.DEFINE_integer("num_train_episodes", int(1000),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 100,
                     "Episode frequency at which the agents are evaluated.")


def main(_):
  game = pyspiel.load_game(FLAGS.game)

  cfr_solver = cfr.CFRSolver(game)

  for ep in range(FLAGS.num_train_episodes):
    cfr_solver.evaluate_and_update_policy()
    if (ep + 1) % FLAGS.eval_every == 0:
        expl = exploitability.exploitability(game, cfr_solver.average_policy())
        logging.info("[%s] Exploitability AVG %s", ep + 1, expl)

  average_policy = cfr_solver.average_policy()
  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
  logging.info("Computed player 0 value: {}".format(average_policy_values[0]))
  logging.info("Computed player 1 value: {}".format(average_policy_values[1]))
#   logging.info("Expected player 0 value: {}".format(-1 / 18))


if __name__ == "__main__":
  app.run(main)