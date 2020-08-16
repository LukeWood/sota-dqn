import gym

from sota_dqn import DQNInference

from preprocessing import grayscale

env = gym.make("MsPacman-v0")
input_shape = env.observation_space.shape[:-1]

dqn = DQNInference(env=env, load_from="ms-pacman.model",
                   epsilon=0.01,
                   input_shape=input_shape,
                   observation_preprocessors=[grayscale],
                   frame_buffer_size=4, warmup_actions=4)

dqn.play_round(sleep=50)
