import gym

from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Dense, Reshape, Flatten
import tensorflow.keras as keras

from sota_dqn import DQNTrainer, BasicReplayMemory

env = gym.make("CartPole-v1")
frame_buffer = 3

input_shape = env.observation_space.shape

inputs = []
for i in range(frame_buffer):
    layer = Input(shape=input_shape)
    inputs.append(layer)


reshape = Reshape((1,) + input_shape)
reshaped = [reshape(i) for i in inputs]

merged = Concatenate(axis=1)(reshaped) if frame_buffer != 1 else reshaped[0]

d0 = Dense(24, activation="relu", name="dense0")(merged)
d1 = Dense(48, activation="relu", name="dense1")(d0)
d2 = Dense(24, activation="relu", name="dense2")(d1)

flattened = Flatten()(d2)

outputs = \
    Dense(env.action_space.n, activation="relu",
          name="output_dense")(flattened)

model = keras.Model(
    inputs=inputs,
    outputs=outputs
)

keras.utils.plot_model(model, "media/basic_model.png", show_shapes=True)

model.compile(
    optimizer="Adam",
    loss="mean_squared_error"
)

dqn = DQNTrainer(
    env=env,
    model=model,
    replay_batch_size=64,
    epochs_per_batch=1,
    input_shape=input_shape,
    memory=BasicReplayMemory(2000),
    frame_buffer_size=frame_buffer,
    persistence_file="cartpole.model",
    reward_chart="media/cartpole_rewards.png",
    save_every=1
)

dqn.train(episodes=40)
