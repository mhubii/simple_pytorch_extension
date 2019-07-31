import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# get data
data = np.genfromtxt("path.csv")

fig, ax = plt.subplots()

# setup all plots
ax.plot(data[0, 0], data[0, 1], 'x', c='black', label='Spawn')                      # spawn of the agent

agent, = ax.plot(data[0,0], data[0,1], 'o', c='b', label='Agent') # agent
agent_line, = ax.plot(data[0,0], data[0,1], '-', c='b')           # small tail following the agent

# plot settings
ax.set_xlabel('x / a.u.')
ax.set_ylabel('y / a.u.')
ax.set_xlim(0, 32)
ax.set_ylim(0, 32)
ax.set_title("Moving Mean")
ax.legend()


# tail for the agent
tail = 0

def animate(i):
    agent.set_data(data[i,0], data[i,1])
    global tail
    agent_line.set_data(data[i-tail:i,0], data[i-tail:i,1])
    if (tail <= 50):
        tail += 1

    return agent, agent_line


ani = animation.FuncAnimation(fig, animate, blit=True, interval=50)
plt.show()
# uncomment these lines to save the plots as gif
#if (train_mode):
#    ani.save('img/epoch_{}.gif'.format(e), writer='imagemagick', fps=100)
#else:
#    ani.save('img/test_mode.gif'.format(e), writer='imagemagick', fps=100)
