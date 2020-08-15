import gym
from matplotlib import animation
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
env.reset()


def save_frames_as_gif(frames, outfile):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1]/72,
                        frames[0].shape[0]/72), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(outfile, writer='imagemagick', fps=60)


done = False
frames = []
for i in range(200):
    frames.append(env.render(mode="rgb_array"))

    action = env.action_space.sample()
    state, next_reward, done, _ = env.step(action)

save_frames_as_gif(frames, "./media/random-cartpole.gif")
