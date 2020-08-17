import tensorflow as tf
import numpy as np

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
    '''

    def __init__(self,
                 env=None,
                 model=None,
                 memory=None,

                 input_shape=None,

                 frame_buffer_size=1,

                 observation_preprocessors=[],

                 pre_episode_callbacks=[],
                 post_episode_callbacks=[],

                 pre_step_callbacks=[],
                 post_step_callbacks=[],

                 # Hyper parameters
                 gamma=0.85,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.998,
                 tau=0.12,
                 replay_batch_size=64,
                 epochs_per_batch=1,
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

        self.steps = 0
        self.average_step_time = 0

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.replay_batch_size = replay_batch_size
        self.epochs_per_batch = epochs_per_batch

        self.pre_episode_callbacks = pre_episode_callbacks
        self.post_episode_callbacks = post_episode_callbacks

        self.pre_step_callbacks = pre_step_callbacks
        self.post_step_callbacks = post_step_callbacks

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
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        res = self.model.predict(state)
        return np.argmax(res)

    def replay_train(self):
        if len(self.memory) < self.replay_batch_size:
            return

        samples = self.memory.sample(self.replay_batch_size)

        states = []
        targets = []

        for i, sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_next = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_next*self.gamma
            states.append(state)
            targets.append(target)
        self.model.fit(states,
                       targets, epochs=self.epochs_per_batch, verbose=0)

    def train(self, episodes=1, skip=0, max_steps=None,
              print_every=5):
        for trial in range(episodes):
            self.add_frame(self.env.reset())

            steps = 1
            done = False
            total_reward = 0

            for callback in self.pre_episode_callbacks:
                callback(self, self.episodes_run)

            while not done:
                for callback in self.pre_step_callbacks:
                    callback(self)

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

                observation, freward, done, diagnostics = self.env.step(action)
                reward = reward + freward
                self.add_frame(observation)

                new_state = self.buffer_to_input()

                self.remember(cur_state, action, reward, new_state, done)
                self.replay_train()
                self.copy_to_target()
                self.decrement_epsilon()

                for callback in self.post_step_callbacks:
                    callback(self, self.episodes_run, steps, reward)

                total_reward = total_reward + reward
                if done:
                    break
                if max_steps is not None and steps > max_steps:
                    break

            for callback in self.post_episode_callbacks:
                callback(self, self.episodes_run, total_reward)

            self.episodes_run = self.episodes_run + 1
