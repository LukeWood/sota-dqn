import gym
from gym.wrappers import Monitor

from sota_dqn import DQNInference

env = gym.make("CartPole-v1")
env = Monitor(env, './media/cartpole-video')
dqn = DQNInference(env=env, load_from="cartpole.model", warmup_actions=4)

dqn.play_round(sleep=1)
