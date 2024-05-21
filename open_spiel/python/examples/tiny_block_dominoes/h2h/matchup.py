import argparse
import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import torch
import jax

import pyspiel
import open_spiel.python.games


game_name = "python_block_dominoes"
results_dir = "open_spiel/python/examples/saved_examples/tiny_block_dominoes/results/h2h/"
agents_dir = "open_spiel/python/examples/saved_examples/tiny_block_dominoes/agents/"
agent_choices = ["cfr", "cfrplus", "mmd_dilated", "random"]
num_episodes = 1_000
seeds = [0, 1, 2]

def make_player(dir: str, game: str, agent: str, seed: str):
    # specify process for cfr and cfrplus
    if agent in ["cfr", "cfrplus"]:
        with open (f"{dir}/{agent}_{game}_20000.pkl", 'rb') as f:
            learner = pickle.load(f)
        policy = learner.average_policy()
        return player_factory(policy)
    elif agent == "random":
        return random
    ...


def random(state: pyspiel.State) -> int:
    return np.random.choice(state.legal_actions())


def player_factory(policy):
    def player(state: pyspiel.State) -> int:
        action_probs = policy.action_probabilities(state)
        action_list = list(action_probs.keys())
        action = np.random.choice(action_list, p=list(action_probs.values()))
        return action
    return player


def matchup(game: pyspiel.Game,
            players: list[Callable[[list[float], list[int]], int]],
            num_episodes: int,
            fn: str,
) -> None:
    """Matchup players in a game for a number of episodes."""
    results = []
    for i in range(num_episodes):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
            else: 
                player = players[state.current_player()]
                action = player(state)
        
            state.apply_action(action)
        # record results
        results.append(state.returns())
        # save results
    np.save(fn + ".npy", results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent1",
        choices=agent_choices,
        required=True,
    )
    parser.add_argument(
        "--agent2",
        choices=agent_choices,
        required=True,
    )
    args = parser.parse_args()
    game = pyspiel.load_game(game_name)
    agents = [
        [make_player(agents_dir, game_name, agent, s) for s in seeds]
        for agent in [args.agent1, args.agent2]
    ]

    p1_outcomes = []
    moving = []
    for s1, a1 in enumerate(agents[0]):
        for s2, a2 in enumerate(agents[1]):
            # agent 1 moves first
            fn1 = results_dir + f"/{game_name}_{args.agent1}_{s1}_{args.agent2}_{s2}"
            matchup(game, [a1, a2], num_episodes, fn1)
            p1_outcomes += np.load(fn1 + ".npy")[:, 0].tolist()
            moving += (num_episodes // 2) * ["First Moving"]
            moving += (num_episodes // 2) * ["First Moving"]
            # agent 2 moves first
            fn2 = results_dir + f"/{game_name}_{args.agent2}_{s2}_{args.agent1}_{s1}"
            matchup(game, [a2, a1], num_episodes, fn2)
            p1_outcomes += np.load(fn2 + ".npy")[:, 1].tolist()
            moving += (num_episodes // 2) * ["Second Moving"]
            moving += (num_episodes // 2) * ["Second Moving"]

            print(len(p1_outcomes), len(moving))
    colname = f"{args.agent1} Return Against {args.agent2}"
    df = pd.DataFrame({colname: p1_outcomes, "Order": moving})
    # save
    df.to_csv(results_dir + f"{game_name}_{args.agent1}_{args.agent2}.csv")
    # plot
    sns.boxplot(data=df, x="Order", y = colname)
    plt.savefig(results_dir + f"/{game_name}_{args.agent1}_{args.agent2}.png")
    expected_return = round(df[colname].mean(), 2)
    std_err = round(df[colname].std() / np.sqrt(len(df[colname])), 2)
    print(f"Expected {colname}: {expected_return} +/- {std_err}")




if __name__ == "__main__":
    main()