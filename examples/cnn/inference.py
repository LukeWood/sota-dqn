import gym

from constants import persistence_file
from sota_dqn import DQNInference
from preprocessing import grayscale
import tensorflow as tf

env = gym.make("MsPacman-v0")
input_shape = env.observation_space.shape[:-1]

model = tf.keras.models.load_model(persistence_file)

dqn = DQNInference(env=env,
                   model=model,
                   input_shape=input_shape,
                   observation_preprocessors=[grayscale],
                   frame_buffer_size=4,
                   warmup_actions=4)

dqn.play_round(sleep=50)
