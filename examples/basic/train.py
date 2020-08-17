from sota_dqn import DQNTrainer, BasicReplayMemory
import gym
import os
import atexit

from rich.table import Table
from rich.console import Console
import tensorflow as tf
import numpy as np

from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Dense, Reshape, Flatten
import tensorflow.keras as keras

import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('Agg')


env = gym.make("CartPole-v1")
frame_buffer = 3
input_shape = env.observation_space.shape

persistence_file = "cartpole.model"


def get_model():
    if os.path.exists(persistence_file):
        print("Loading model from ", persistence_file)
        return tf.keras.models.load_model(persistence_file)

    inputs = []
    for i in range(frame_buffer):
        layer = Input(shape=input_shape)
        inputs.append(layer)

    reshape = Reshape((1,) + input_shape)
    reshaped = [reshape(i) for i in inputs]

    merged = Concatenate(axis=1)(
        reshaped) if frame_buffer != 1 else reshaped[0]

    d0 = Dense(24, activation="relu", name="dense0")(merged)
    d1 = Dense(48, activation="relu", name="dense1")(d0)
    d2 = Dense(24, activation="relu", name="dense2")(d1)

    flattened = Flatten()(d2)

    outputs = Dense(env.action_space.n, activation="relu",
                    name="output_dense")(flattened)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer="Adam",
        loss="mean_squared_error"
    )

    return model


model = get_model()
keras.utils.plot_model(model, "media/basic_model.png", show_shapes=True)
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

    plt.savefig("media/cartpole_rewards.png")


def save_model(dqn, _episode=None, _episode_reward=None):
    dqn.model.save(persistence_file)
    print("Saved model to", persistence_file)


atexit.register(lambda: model.save(persistence_file))


dqn = DQNTrainer(
    env=env,
    model=model,
    replay_batch_size=64,
    epochs_per_batch=1,
    frame_buffer_size=frame_buffer,
    input_shape=input_shape,
    memory=BasicReplayMemory(2000),
    episode_callbacks=[clear_console, track_reward,
                       plot, save_model, append_row, print_table]
)

dqn.train(episodes=40)
