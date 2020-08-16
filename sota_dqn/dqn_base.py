from functools import reduce
from collections import deque

import numpy as np


class DQNBase:
    def __init__(self,
                 input_shape=None,
                 frame_buffer_size=None,
                 observation_preprocessors=[]):
        self.input_shape = input_shape
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.observation_preprocessors = observation_preprocessors

    def preprocess_observation(self, observation):
        return reduce(lambda acc, preprocess: preprocess(acc),
                      self.observation_preprocessors,
                      observation)

    def buffer_to_input(self):
        result = []
        for i in range(self.frame_buffer_size):
            result.append(np.zeros(self.input_shape))
        for i, frame in enumerate(self.frame_buffer):
            result[i] = frame
        return [np.expand_dims(i, axis=0) for i in result]

    def add_frame(self, frame):
        self.frame_buffer.append(
            self.preprocess_observation(frame))
