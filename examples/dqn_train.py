import gym

from sota_dqn import DQNTrainer, BasicReplayMemory

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

env = gym.make("CartPole-v1")


def model_provider():
    model = Sequential()
    state_shape = env.observation_space.shape
    model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
    model.add(Dense(48, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(env.action_space.n))
    model.compile(loss="mean_squared_error",
                  optimizer="Adam")
    return model


dqn = DQNTrainer(env=env,
                 model_provider=model_provider,
                 memory=BasicReplayMemory(size=2000),
                 persistence_file="cartpole.model")

dqn.train(episodes=20)
