import gym

from sota_dqn import DQNInference

env = gym.make("MsPacman-v0")

dqn = DQNInference(env=env, load_from="ms-pacman.model",
                    epsilon=1,
                   input_shape=env.observation_space.shape,
                   frame_buffer_size=4, warmup_actions=4)

dqn.play_round(sleep=50)
