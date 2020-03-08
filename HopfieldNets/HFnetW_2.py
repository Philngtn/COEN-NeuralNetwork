import numpy as np 
import matplotlib.pyplot as plt
import Number_data as nbdt
import math
from Number_data import corrupted_pattern

width = 10
height = 12
reserved_pixel = 0.25

def corrupted_random(data,reserved_pixel):
    corrupted_pixel = int(len(data)*(1-reserved_pixel))
    for i in range(corrupted_pixel):
        index = np.random.randint(0,width*height)
        data[index] = data[index]*-1
    return data

def energy(state, W):
    return float(-0.5 * state @ W @ state)

def weigh_cal(state):
    return np.matrix(state).T*state

Data = np.stack((nbdt.pattern_0,nbdt.pattern_1,nbdt.pattern_2,nbdt.pattern_3
                ,nbdt.pattern_4,nbdt.pattern_6,nbdt.pattern_dot,nbdt.pattern_9)).T


W = Data @ Data.T
np.fill_diagonal(W,0)

# plt.imshow(W)
# plt.colorbar()
# plt.show()

pattern = np.copy(nbdt.pattern_9)
pattern_label = 9

# corrupted = pattern 
corrupted = corrupted_random(pattern,reserved_pixel)
# plt.imshow(corrupted.reshape(12,10))
# plt.show()


ev = []
en = []
energy_old = 0
energy_new = energy(corrupted, W)

steps = 20
iteration = 0
count = 0
# we keep running until we reach the lowest energy level
while energy_old > energy_new and iteration < steps:
    iteration += 1
    energy_old = energy_new
    ev.append(np.copy(corrupted))
    en.append(energy_old)
    # -------------------------------------------------
    # Asynchronous update 
    for pixel in np.split(np.random.randint(0,len(corrupted),width*height), 12):
        corrupted[pixel] = np.sign(W[pixel,:] @ corrupted)
    # ------------------------------------------------
    # Synchronous update
    # corrupted = np.sign(W @ corrupted)
    # -------------------------------------------------
    energy_new = energy(corrupted, W)
    if energy_old == energy_new and iteration == 1:
        ev.append(np.copy(corrupted))
        en.append(energy_new)

print('Stopped at iteration {}'.format(iteration))

show_every = 1

fig, ax = plt.subplots(1,int(len(ev)//show_every))
axes = ax.ravel()
fig.suptitle('Reconstruct pattern {} lowering network energy'.format(pattern_label))
# fig.suptitle('Test Clean Pattern {}'.format(pattern_label))
plot_idx = 0
for idx in range(len(ev)):
    if (idx%show_every)==0:
        axes[plot_idx].imshow(np.reshape(ev[idx],(12,10)))
        axes[plot_idx].set_title('Iteration {} \n Net. E: {}'.format(idx, en[idx]))
        plot_idx += 1

plt.show()


