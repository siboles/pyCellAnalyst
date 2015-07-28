import pickle
import numpy as np
import matplotlib.pyplot as plt

fid = open("results.pkl", "rb")
data = pickle.load(fid)
fid.close()

order = np.zeros(data["truth"].shape, np.uint32)
ind = np.arange(data["truth"][:, 0, 0].size)
for i in xrange(3):
    for j in xrange(3):
        order[:, i, j] = np.argsort(data["truth"][:, i, j])
f, ax = plt.subplots(3, 2, sharex=True)
f.set_size_inches([2 * 3.34646, 3 * 3.34646])

ax[0, 0].fill_between(ind, data["residual"][order[:, 0, 0], 0, 0],
                      0, alpha=.9,
                      linewidth=0.0, facecolor='red')
ax[0, 0].fill_between(ind, data["residual"][order[:, 0, 0], 0, 0],
                      data["truth"][order[:, 0, 0], 0, 0], alpha=0.5,
                      linewidth=0.0, facecolor='green')

ax[1, 0].fill_between(ind, data["residual"][order[:, 1, 1], 1, 1],
                      data["truth"][order[:, 1, 1], 1, 1], alpha=0.5,
                      linewidth=0.0, facecolor='green')
ax[1, 0].fill_between(ind, data["residual"][order[:, 1, 1], 1, 1],
                      0, alpha=.9,
                      linewidth=0.0, facecolor='red')

ax[2, 0].fill_between(ind, data["residual"][order[:, 2, 2], 2, 2],
                      data["truth"][order[:, 2, 2], 2, 2], alpha=0.5,
                      linewidth=0.0, facecolor='green')
ax[2, 0].fill_between(ind, data["residual"][order[:, 2, 2], 2, 2],
                      0, alpha=.9,
                      linewidth=0.0, facecolor='red')

ax[0, 1].fill_between(ind,
                      data["residual"][order[:, 0, 1], 0, 1],
                      data["truth"][order[:, 0, 1], 0, 1],
                      alpha=0.5, linewidth=0.0, facecolor='green')
ax[0, 1].fill_between(ind,
                      data["residual"][order[:, 0, 1], 0, 1],
                      0, alpha=0.9, linewidth=0.0, facecolor='red')

ax[1, 1].fill_between(ind,
                      data["residual"][order[:, 0, 2], 0, 2],
                      data["truth"][order[:, 0, 2], 0, 2],
                      alpha=0.5, linewidth=0.0, facecolor='green')
ax[1, 1].fill_between(ind,
                      data["residual"][order[:, 0, 2], 0, 2],
                      0, alpha=0.9, linewidth=0.0, facecolor='red')

ax[2, 1].fill_between(ind,
                      data["residual"][order[:, 1, 2], 1, 2],
                      data["truth"][order[:, 1, 2], 1, 2],
                      alpha=0.5, linewidth=0.0, facecolor='green')
ax[2, 1].fill_between(ind,
                      data["residual"][order[:, 1, 2], 1, 2],
                      0, alpha=0.9, linewidth=0.0, facecolor='red')
shear_titles = ("$E_{xy}}$", "$E_{xz}$", "$E_{yz}$")
normal_titles = ("$E_{xx}$", "$E_{yy}$", "$E_{zz}$")

for i in xrange(3):
    ax[i, 1].yaxis.tick_right()
    ax[i, 1].set_title(shear_titles[i], y=1.02)
    ax[i, 0].set_title(normal_titles[i], y=1.02)
    [l.set_visible(False) for l in ax[i, 0].get_xticklabels()]
    [l.set_visible(False) for l in ax[i, 1].get_xticklabels()]

plt.savefig("rms_error.svg")
