import tensorflow as tf
import numpy as np

from collections import deque
import os
import atexit

from .dqn_base import DQNBase


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
                 model_provider=None,
                 memory=None,

                 frame_buffer_size=1,

                 observation_preprocessors=[],

                 # Hyper parameters
                 gamma=0.85,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.998,
                 tau=0.12,
                 replay_batch_size=64,

                 # Persistence
                 persistence_file=None,
                 save_every=10
                 ):
        super().__init__(observation_preprocessors=observation_preprocessors)
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

        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.frame_buffer_size = frame_buffer_size

    def buffer_to_input(self):
        dims = self.model.input_shape[1:]
        result = np.zeros(dims)
        for i, frame in enumerate(self.frame_buffer):
            result[i] = frame
        return np.expand_dims(result, axis=0)

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

    def train(self, episodes=1, max_steps=None):
        for trial in range(episodes):
            self.frame_buffer.append(
                self.preprocess_observation(self.env.reset()))

            if self.save_every is not None and trial % self.save_every == 0:
                if self.persistence_file is not None:
                    self.save_model(self.persistence_file)

            steps = 0
            done = False
            while not done:
                steps = steps + 1
                cur_state = self.buffer_to_input()
                action = self.pick_action(cur_state)

                observation, reward, done, diagnostics = self.env.step(action)
                observation = self.preprocess_observation(observation)

                self.frame_buffer.append(observation)

                new_state = self.buffer_to_input()

                self.remember(cur_state, action, reward, new_state, done)
                self.replay_train()
                self.copy_to_target()
                self.decrement_epsilon()

                if done:
                    break
                if max_steps and steps > max_steps:
                    break
