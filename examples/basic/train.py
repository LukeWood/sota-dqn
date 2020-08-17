import numpy as np
import time
import cProfile
from sota_dqn import DQNTrainer, BasicReplayMemory
import gym
import os
import atexit

import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Dense, Reshape, Flatten
import tensorflow.keras as keras

from constants import persistence_file

from callbacks import clear_console, track_reward, plot, save_model, \
    append_row, print_table

env = gym.make("CartPole-v1")
frame_buffer = 3
input_shape = env.observation_space.shape


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

atexit.register(lambda: model.save(persistence_file))


t = 0
all_times = []


def step_start(dqn):
    global t
    t = time.time()*1000


def step_end(dqn, _episode, steps, reward):
    n = time.time()*1000
    all_times.append(n-t)


def print_avg_time(dqn, _1, _2):
    print("Avg step time", np.average(all_times))


dqn = DQNTrainer(
    env=env,
    model=model,
    replay_batch_size=12,
    epochs_per_batch=1,
    epsilon_decay=0.999,
    frame_buffer_size=frame_buffer,
    input_shape=input_shape,
    memory=BasicReplayMemory(2000),
    pre_step_callbacks=[step_start],
    post_step_callbacks=[step_end],
    post_episode_callbacks=[clear_console, track_reward,
                            plot, save_model, append_row, print_table, print_avg_time]
)

dqn.train(episodes=40, max_steps=10)
