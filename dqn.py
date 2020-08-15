import tensorflow as tf
import numpy as np

import os

import random
import gym

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import atexit


class ReplayMemory:
    def __init__(self, size=2000):
        self.size = size
        self.memory = []
        self.index = 0

    def save(self, frame):
        if len(self.memory) < self.size:
            self.memory.append(frame)
        else:
            self.memory[self.size % self.index] = frame
            self.index = self.index + 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self,
                 env=None,
                 model=None,
                 memory=None,
                 gamma=0.85,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.998,
                 tau=0.12,
                 replay_batch_size=16,
                 persistence_file=None
                 ):
        if not env:
            raise "env required"

        self.env = env
        self.model = model
        self.memory = memory

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.replay_batch_size = replay_batch_size

        self.training = True

        self.persistence_file = persistence_file
        if persistence_file:
            if os.path.exists(persistence_file):
                print("Loading model from", persistence_file)
                self.model = tf.keras.models.load_model(persistence_file)
            atexit.register(lambda: self.save_model(persistence_file))

        self.copy_to_target()

    def copy_to_target(self):
        self.target_model = tf.keras.models.clone_model(self.model)

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon_decay*self.epsilon, self.epsilon_min)

    def remember(self, state, action, reward, new_state, done):
        self.memory.save((state, action, reward, new_state, done))

    def pick_action(self, state):
        if self.training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def replay_train(self):
        if len(self.memory) < self.replay_batch_size:
            return

        samples = self.memory.sample(self.replay_batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_next = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_next*self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def save_model(self, f):
        print("Saving model to", f)
        self.model.save(f)


env = gym.make("CartPole-v1")

model = Sequential()
state_shape = env.observation_space.shape
model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
model.add(Dense(48, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(env.action_space.n))
model.compile(loss="mean_squared_error",
              optimizer="Adam")

dqn = DQN(env=env, model=model, memory=ReplayMemory(
    size=2000))

for trial in range(100):
    cur_state = env.reset().reshape(1, 4)

    for step in range(500):
        action = dqn.pick_action(cur_state)
        new_state, reward, done, diagnostics = env.step(action)
        new_state = new_state.reshape(1, 4)

        dqn.remember(cur_state, action, reward, new_state, done)
        dqn.replay_train()
        dqn.copy_to_target()
        dqn.decrement_epsilon()

        cur_state = new_state

        env.render()

        if done:
            break
