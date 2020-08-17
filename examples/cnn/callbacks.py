from rich.table import Table
from rich.console import Console
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from constants import persistence_file

mpl.use('Agg')
episodes = Table(show_header=True)
all_rewards = []
console = Console()
episodes.add_column("Episode")
episodes.add_column("Average Reward")
episodes.add_column("Episode Reward")
episodes.add_column("Epsilon")


def track_reward(dqn, _episode, episode_reward):
    all_rewards.append(episode_reward)


def append_row(dqn, episode, episode_reward):
    episodes.add_row(
        str(dqn.episodes_run),
        str(np.average(all_rewards)),
        str(episode_reward),
        str(dqn.epsilon))


def clear_console(_1, _2, _3):
    console.clear()


def print_table(_dqn, _episode, _episode_reward):
    console.print(episodes)


def plot(dqn, episode, episode_reward):
    sns.lineplot(x=range(len(all_rewards)),
                 y=all_rewards)

    plt.savefig("media/ms_pacman_rewards.png")


def save_model(dqn, _episode=None, _episode_reward=None):
    dqn.model.save(persistence_file)
    print("Saved model to", persistence_file)
