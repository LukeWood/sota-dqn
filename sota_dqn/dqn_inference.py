import numpy as np
import time

from tensorflow.keras.models import load_model

from collections import deque


class DQNInference:
    def __init__(self,
                 env=None,
                 model=None,
                 epsilon=0,
                 load_from=None,
                 frame_buffer_size=1,
                 warmup_actions=1
                 ):
        if not env:
            raise "env required"

        self.epsilon = epsilon
        self.env = env
        self.warmup_actions = warmup_actions

        if load_from:
            self.model = load_model(load_from)
        else:
            self.model = model

        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.frame_buffer_size = frame_buffer_size

    def buffer_to_input(self):
        dims = (self.frame_buffer_size,) + self.env.observation_space.shape
        result = np.zeros(dims)
        for i, frame in enumerate(self.frame_buffer):
            result[i] = frame
        return np.expand_dims(result, axis=0)

    def pick_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def play_round(self, render=True, sleep=50):
        self.frame_buffer.append(self.env.reset())
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
            self.frame_buffer.append(observation)
            reward = reward + next_reward
            if render:
                self.env.render()
                if sleep:
                    time.sleep(sleep/1000)
        return reward
