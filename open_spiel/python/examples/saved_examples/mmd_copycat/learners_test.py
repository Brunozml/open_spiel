import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspiel
import open_spiel.python.games
import seaborn as sns

from learners import QLearner
from run import main


game = pyspiel.load_game("python_block_dominoes")

from open_spiel.python.policy import TabularPolicy
# 1. how does a normal tabular policy look like? 
tabular_policy = TabularPolicy(game)  # You need to create a game object
print(tabular_policy.states_per_player[0][0])


learner = QLearner(game)

num_iterations = 10

# learner.update()
for i in range(num_iterations):
    learner.update()
    if i % 10 == 0:
        print(learner.log_info())


