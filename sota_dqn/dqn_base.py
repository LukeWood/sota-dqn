from functools import reduce


class DQNBase:
    def __init__(self,
                 observation_preprocessors=[]):
        self.observation_preprocessors = observation_preprocessors

    def preprocess_observation(self, observation):
        return reduce(lambda acc, preprocess: preprocess(acc),
                      self.observation_preprocessors,
                      observation)
