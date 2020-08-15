import tensorflow as tf
import numpy as np

import os
import atexit


class DQNTrainer:
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
                 model_provider=None,
                 memory=None,

                 # Hyper parameters
                 gamma=0.85,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.998,
                 tau=0.12,
                 replay_batch_size=16,

                 # Persistence
                 persistence_file=None,
                 save_every=10
                 ):
        if not env:
            raise "env required"

        self.env = env
        self.model = model_provider()
        self.target_model = model_provider()
        self.memory = memory

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.replay_batch_size = replay_batch_size

        self.training = True

        self.persistence_file = persistence_file
        self.save_every = save_every

        if persistence_file:
            if os.path.exists(persistence_file):
                self.model = tf.keras.models.load_model(persistence_file)
            atexit.register(lambda: self.save_model(persistence_file))

        self.copy_to_target()

    def copy_to_target(self):
        # unfortunately tf.keras.models.clone_model does not work after loading
        # the model from persistence_file
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

    def train(self, episodes=1):
        input_dims = [
            x if x is not None else 1 for x in self.model.input_shape]
        for trial in range(episodes):
            cur_state = self.env.reset().reshape(input_dims)

            if self.save_every and trial % self.save_every == 0:
                self.save_model(self.persistence_file)

            for step in range(500):
                action = self.pick_action(cur_state)
                new_state, reward, done, diagnostics = self.env.step(action)
                new_state = new_state.reshape(input_dims)

                self.remember(cur_state, action, reward, new_state, done)
                self.replay_train()
                self.copy_to_target()
                self.decrement_epsilon()

                cur_state = new_state

                if done:
                    break
