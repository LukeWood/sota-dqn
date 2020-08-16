import gym

from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, Flatten

import tensorflow.keras as keras

from sota_dqn import DQNTrainer, BasicReplayMemory

env = gym.make("MsPacman-v0")
frame_buffer = 4

input_shape = env.observation_space.shape
inputs = []
for i in range(frame_buffer):
    layer = Input(shape=input_shape)
    inputs.append(layer)

conv_layer = Conv2D(
        filters=16, kernel_size=(3, 3),
        padding='same', activation='relu'
)
inputs_convoluted = [conv_layer(inp) for inp in inputs]

pool_layer = MaxPooling2D(pool_size=(3, 3))
inputs_pooled = [pool_layer(i) for i in inputs_convoluted]

conv_layer2 = Conv2D(
        filters=16, kernel_size=(2, 2),
        padding='same', activation='relu'
)
inputs_convoluted = [conv_layer2(i) for i in inputs_pooled]

pool_layer2 = MaxPooling2D(pool_size=(2, 2))
inputs_pooled2 = [pool_layer2(i) for i in inputs_convoluted]

flatten_layer = Flatten()
flattened = [flatten_layer(i) for i in inputs_pooled2]
merged = Concatenate()(flattened)

d0 = Dense(48, activation='relu', name='dense0')(merged)
d1 = Dense(24, activation='relu', name='dense1')(d0)

outputs = \
    Dense(env.action_space.n, activation="relu", name="output_dense")(d1)

model = keras.Model(
    inputs=inputs,
    outputs=outputs
)

keras.utils.plot_model(model, "media/cnn_model.png", show_shapes=True)

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

dqn = DQNTrainer(
        env=env,
        model=model,
        replay_batch_size=1,
        input_shape=input_shape,
        memory=BasicReplayMemory(2000),
        frame_buffer_size=frame_buffer,
        persistence_file="ms-pacman.model",
        save_every=1,
        reward_chart="media/ms-pacman-rewards.png"
)

dqn.train(episodes=100)
