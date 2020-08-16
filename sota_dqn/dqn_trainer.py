import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import os
import atexit

from .dqn_base import DQNBase

import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')


class DQNTrainer(DQNBase):
    '''
            DQNTrainer trains a DQN Network.

                Parameters:
                    env (gym.Environment): openai gym environment
                    model_provider: function returning a keras model
                    memory: instance of BasicReplayMemory used in training
                    gamma: the gamma dqn hyperparameter
                    epsilon: probability the network takes a random action
                    epsilon_decay: the factor to multiply epsilon by
                    tau: tau hyperparameter
                    replay_batch_size: batch size for use in replay training
                    persistence_file: file to save the model to - required
                    save_every: automatically save the model every N iterations
    '''

    def __init__(self,
                 env=None,
                 model=None,
                 memory=None,

                 input_shape=None,

                 frame_buffer_size=1,

                 observation_preprocessors=[],

                 # Hyper parameters
                 gamma=0.85,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.998,
                 tau=0.12,
                 replay_batch_size=64,
                 epochs_per_batch=1,

                 # Persistence
                 persistence_file=None,
                 save_every=10,

                 reward_chart=None
                 ):
        super().__init__(
            observation_preprocessors=observation_preprocessors,
            input_shape=input_shape,
            frame_buffer_size=frame_buffer_size
        )
        if not env:
            raise "env required"

        self.env = env
        self.model = model
        self.memory = memory

        self.episodes_run = 0
        self.all_rewards = []
        self.reward_chart = reward_chart

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.replay_batch_size = replay_batch_size
        self.epochs_per_batch = epochs_per_batch

        self.training = True

        self.persistence_file = persistence_file
        self.save_every = save_every

        if persistence_file:
            if os.path.exists(persistence_file):
                self.model = tf.keras.models.load_model(persistence_file)

            atexit.register(lambda: self.save_model(persistence_file))

        # tf.keras.models.clone_model does not copy weights
        self.target_model = tf.keras.models.clone_model(self.model)
        self.copy_to_target()

    def copy_to_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon_decay*self.epsilon, self.epsilon_min)

    def remember(self, state, action, reward, new_state, done):
        self.memory.save((state, action, reward, new_state, done))

    def pick_action(self, state):
        if self.training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        res = self.model.predict(state)
        return np.argmax(res)

    def replay_train(self):
        if len(self.memory) < self.replay_batch_size:
            return

        samples = self.memory.sample(self.replay_batch_size)

        for i, sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_next = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_next*self.gamma
            self.model.fit(state,
                           target, epochs=self.epochs_per_batch, verbose=0)

    def save_model(self, f):
        print("Saving model to", f)
        self.model.save(f)

    def train(self, episodes=1, skip=0, max_steps=None, visualize=False):
        for trial in range(episodes):
            self.add_frame(self.env.reset())

            if trial % self.save_every == 0:
                if self.persistence_file is not None:
                    self.save_model(self.persistence_file)
                if self.reward_chart is not None:
                    sns.lineplot(x=range(len(self.all_rewards)),
                                 y=self.all_rewards)
                    plt.savefig(self.reward_chart)

            steps = 0
            done = False
            total_reward = 0
            while not done:
                steps = steps + 1

                cur_state = self.buffer_to_input()
                action = self.pick_action(cur_state)

                reward = 0
                for i in range(skip):
                    observation, nreward, done, diagnostics = self.env.step(
                        action)
                    self.add_frame(observation)
                    reward = reward + nreward
                    if done:
                        break

                if visualize:
                    self.env.render()

                observation, freward, done, diagnostics = self.env.step(action)
                reward = reward + freward
                self.add_frame(observation)

                new_state = self.buffer_to_input()

                self.remember(cur_state, action, reward, new_state, done)
                self.replay_train()
                self.copy_to_target()
                self.decrement_epsilon()

                total_reward = total_reward + reward
                if done:
                    break
                if max_steps is not None and steps > max_steps:
                    break

            self.all_rewards.append(total_reward)
            self.episodes_run = self.episodes_run + 1
            print("Episode", self.episodes_run, "Reward", total_reward)
