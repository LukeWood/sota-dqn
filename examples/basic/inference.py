import gym

from sota_dqn import DQNInference

env = gym.make("CartPole-v1")

dqn = DQNInference(env=env, load_from="cartpole.model",
                   frame_buffer_size=3, warmup_actions=4)

for _ in range(10):
    dqn.play_round(sleep=1)
