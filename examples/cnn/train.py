import os
import gym
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, Flatten, Reshape
from preprocessing import grayscale
from sota_dqn import DQNTrainer, BasicReplayMemory

from callbacks import clear_console, track_reward, plot, save_model, \
    append_row, print_table

from constants import persistence_file

env = gym.make("MsPacman-v0")
frame_buffer = 4

input_shape = env.observation_space.shape[:-1]


def get_model():
    if os.path.exists(persistence_file):
        print("Loading model from ", persistence_file)
        return tf.keras.models.load_model(persistence_file)

    inputs = []
    for i in range(frame_buffer):
        layer = Input(shape=input_shape)
        inputs.append(layer)

    reshape_layer = Reshape(input_shape + (1,))
    reshaped = [reshape_layer(i) for i in inputs]

    conv_layer = Conv2D(
        filters=16, kernel_size=(3, 3),
        padding='same', activation='relu'
    )
    inputs_convoluted = [conv_layer(inp) for inp in reshaped]

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
    merged = Concatenate()(flattened) if frame_buffer != 1 else flattened[0]

    d0 = Dense(48, activation='relu', name='dense0')(merged)
    d1 = Dense(24, activation='relu', name='dense1')(d0)
    d2 = Dense(24, activation='relu', name='dense2')(d1)

    outputs = \
        Dense(env.action_space.n, activation="relu", name="output_dense")(d2)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


model = get_model()

keras.utils.plot_model(model, "media/cnn_model.png", show_shapes=True)

dqn = DQNTrainer(
    env=env,
    model=model,
    observation_preprocessors=[grayscale],
    epsilon_decay=0.999,
    input_shape=input_shape,
    memory=BasicReplayMemory(2000),
    frame_buffer_size=frame_buffer,
    post_episode_callbacks=[clear_console, track_reward,
                            plot, save_model, append_row, print_table]
)

dqn.train(100, skip=5, max_steps=500)
