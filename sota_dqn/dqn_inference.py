import numpy as np
import time

from tensorflow.keras.models import load_model


class DQNInference:
    def __init__(self,
                 env=None,
                 model=None,
                 epsilon=0,
                 load_from=None,
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

    def pick_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def play_round(self, render=True, sleep=50):
        input_dims = [
            x if x is not None else 1 for x in self.model.input_shape]
        state = self.env.reset().reshape(input_dims)
        done = False
        reward = 0
        random_actions_taken = 0
        while not done:
            if random_actions_taken < self.warmup_actions:
                action = self.env.action_space.sample()
                random_actions_taken = random_actions_taken + 1
            else:
                action = self.pick_action(state)
            state, next_reward, done, _ = self.env.step(action)
            state = state.reshape(input_dims)
            reward = reward + next_reward
            if render:
                self.env.render()
                if sleep:
                    time.sleep(sleep/1000)
        return reward
