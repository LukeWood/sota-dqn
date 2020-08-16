import gym

from sota_dqn import DQNTrainer, BasicReplayMemory

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Reshape

env = gym.make("MsPacman-v0")

# TODO(lukewood): get this to work with bigger batch sizes
frame_buffer = 3


def model_provider():
    input_shape = (frame_buffer,) + env.observation_space.shape
    model = Sequential()

    # We reshape the images to exist on the space (width, height, rgb*frames)
    reshape_space = env.observation_space.shape[:-1] + (3*frame_buffer,)

    model.add(Reshape(reshape_space, input_shape=input_shape))

    model.add(Conv2D(filters=16, kernel_size=(2, 2),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(24, activation="relu"))

    # Required for any frame stacking models

    # Required for any DQN
    model.add(Dense(env.action_space.n))
    model.compile(loss="mean_squared_error",
                  optimizer="Adam")
    return model


dqn = DQNTrainer(env=env,
                 model_provider=model_provider,
                 memory=BasicReplayMemory(size=2000),
                 frame_buffer_size=frame_buffer,
                 save_every=3,
                 persistence_file="mrs-pacman.model"
                 )

dqn.train(episodes=100)
