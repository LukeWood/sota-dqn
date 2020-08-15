import gym

from sota_dqn import DQNTrainer, BasicReplayMemory

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

env = gym.make("CartPole-v1")
frame_buffer = 3


def model_provider():
    model = Sequential()
    model.add(Dense(24, batch_size=1, input_shape=(frame_buffer,) +
                    env.observation_space.shape, activation="relu"))
    model.add(Dense(48, activation="relu"))
    model.add(Dense(24, activation="relu"))

    # Required for any frame stacking models
    model.add(Flatten())

    # Required for any DQN
    model.add(Dense(env.action_space.n))
    model.compile(loss="mean_squared_error",
                  optimizer="Adam")
    return model


dqn = DQNTrainer(env=env,
                 model_provider=model_provider,
                 memory=BasicReplayMemory(size=2000),
                 frame_buffer_size=frame_buffer,
                 persistence_file="cartpole.model")

dqn.train(episodes=100)
