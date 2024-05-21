"""Interface for learning in extensive-form games
"""

import pandas as pd

from learners import Learner

## utils
def is_power_of_2(n: int) -> bool:
    """Return whether `n` is power of 2"""
    return (n & (n - 1) == 0) and n != 0


def schedule(upper_lim: int) -> list[tuple[int, bool]]:
    """Return schedule of iterations and whether to save data"""
    ls: list[tuple[int, bool]] = []
    for i in range(upper_lim):
        ls.append((i, is_power_of_2(i) or i == upper_lim - 1))
    return ls

## 

def main(
    learner: Learner,
    num_iterations: int,
    fn: str,
) -> None:
    """Run `learner` for `num_iterations`

    Args:
        learner: Learner to run
        num_iterations: Number of iterations to run
        fn: Filename to which to save data
    """
    df = pd.DataFrame({})
    for _, should_save in schedule(num_iterations):
        learner.update()
        if should_save:
            df = pd.concat([df, pd.DataFrame(learner.log_info())], ignore_index=True)
            df.to_csv(fn + ".csv")