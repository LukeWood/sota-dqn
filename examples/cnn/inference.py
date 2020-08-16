import gym

from sota_dqn import DQNInference

env = gym.make("MsPacman-v0")

dqn = DQNInference(env=env, load_from="mrs-pacman.model",
                   frame_buffer_size=3, warmup_actions=16)

dqn.play_round(sleep=20)
