import numpy as np
import time

from tensorflow.keras.models import load_model

from collections import deque

from .dqn_base import DQNBase


class DQNInference(DQNBase):
    def __init__(self,
                 env=None,
                 input_shape=None,
                 epsilon=0,
                 load_from=None,
                 observation_preprocessors=[],
                 frame_buffer_size=1,
                 warmup_actions=1
                 ):
        super().__init__(
            observation_preprocessors=observation_preprocessors,
            frame_buffer_size=frame_buffer_size,
            input_shape=input_shape
        )

        if not env:
            raise "env required"

        self.epsilon = epsilon
        self.env = env
        self.warmup_actions = warmup_actions

        self.model = load_model(load_from)

    def pick_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        print(self.model.predict(state))
        print(np.argmax(self.model.predict(state)))
        return np.argmax(self.model.predict(state))

    def play_round(self, render=True, sleep=50):
        self.add_frame(self.env.reset())
        done = False
        reward = 0
        random_actions_taken = 0
        while not done:
            state = self.buffer_to_input()
            if random_actions_taken < self.warmup_actions:
                action = self.env.action_space.sample()
                random_actions_taken = random_actions_taken + 1
            else:
                action = self.pick_action(state)
            observation, next_reward, done, _ = self.env.step(action)
            self.add_frame(observation)
            reward = reward + next_reward
            if render:
                self.env.render()
                if sleep:
                    time.sleep(sleep/1000)
        return reward
