import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspiel
import open_spiel.python.games
import seaborn as sns

from learners import CFR, QLearner, MMD
from tree import Objective
from run import main

hyperparameters = {
    "kuhn_poker": {
        "annealing_temperature": {
            "temp_schedule": lambda i: 1 / np.sqrt(i),
            "lr_schedule": lambda i: 1 / np.sqrt(i),
            "mag_lr_schedule": lambda i: 0,
        },
        "moving_magnet": {
            "temp_schedule": lambda i: 1,
            "lr_schedule": lambda i: 0.1,
            "mag_lr_schedule": lambda i: 0.05,
        },
    },
    "leduc_poker": {
        "annealing_temperature": {
            "temp_schedule": lambda i: 5 / np.sqrt(i),
            "lr_schedule": lambda i: 1 / np.sqrt(i),
            "mag_lr_schedule": lambda i: 0,
        },
        "moving_magnet": {
            "temp_schedule": lambda i: 1,
            "lr_schedule": lambda i: 0.1,
            "mag_lr_schedule": lambda i: 0.05,
        },
    },
    "python_block_dominoes": {
        "annealing_temperature": {
            "temp_schedule": lambda i: 5 / np.sqrt(i),
            "lr_schedule": lambda i: 1 / np.sqrt(i),
            "mag_lr_schedule": lambda i: 0,
        },
        "moving_magnet": {
            "temp_schedule": lambda i: 1,
            "lr_schedule": lambda i: 0.1,
            "mag_lr_schedule": lambda i: 0.05,
        },
    },
}

game_choices = [
    "kuhn_poker",
    "leduc_poker",
    "python_block_dominoes",
]

approach_choices = ["annealing_temperature", "moving_magnet"]

alg = "mmd"
approach = "moving_magnet"
game_name = "python_block_dominoes"
game = pyspiel.load_game(game_name)

if alg == "cfr":
    learner = CFR(game, use_plus=False)
elif alg == "cfr_plus":
    learner = CFR(game, use_plus=True)
elif alg == "mmd":
    temp_schedule = hyperparameters[game_name][approach]["temp_schedule"]
    lr_schedule = hyperparameters[game_name][approach]["lr_schedule"]
    mag_lr_schedule = hyperparameters[game_name][approach]["mag_lr_schedule"]
    objective = Objective.standard
    learner = MMD(game,
                  temp_schedule,
                  lr_schedule,
                  mag_lr_schedule,
                  objective=objective)
elif alg == "qlearner": # currently not working
    learner = QLearner(game)

# alg = f"cfr_{args.variant}"
num_iterations = 100_000

fn = f"open_spiel/python/examples/saved_examples/{game_name}_{alg}_{num_iterations}_{approach}"
main(
    learner,
    num_iterations,
    fn,
)
df = pd.read_csv(fn + ".csv")
sns.lineplot(
    data=df,
    x="Iteration",
    y="Exploitability",
)
plt.yscale("log")
plt.xscale("log")
plt.savefig(fn + ".png")