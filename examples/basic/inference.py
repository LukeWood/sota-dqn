import gym
import tensorflow as tf
from sota_dqn import DQNInference

from constants import persistence_file

env = gym.make("CartPole-v1")

model = tf.keras.models.load_model(persistence_file)

dqn = DQNInference(env=env,
                   model=model,
                   input_shape=env.observation_space.shape,
                   frame_buffer_size=3, warmup_actions=4)

for _ in range(10):
    dqn.play_round(sleep=10)
