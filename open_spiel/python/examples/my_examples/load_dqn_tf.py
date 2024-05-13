from absl import flags

import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python import rl_agent
from open_spiel.python.utils.replay_buffer import ReplayBuffer
from open_spiel.python.algorithms import dqn

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", "open_spiel/python/examples/agents/dqn",
                    "Directory to save/load the agent.") # "/tmp/nfsp_test",


# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")


def main(_):
  game = "python_block_dominoes"
  num_players = 2

  # env_configs = {"columns": 5, "rows": 5}
  env = rl_environment.Environment(game)  # , **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  with tf.Session() as sess:
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
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
    for agent in agents:
      agent.resotre(FLAGS.checkpoint_dir)
      

    sess.run(tf.global_variables_initializer())

