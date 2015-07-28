import pickle
import numpy as np
import matplotlib.pyplot as plt

fid = open("results.pkl", "rb")
data = pickle.load(fid)
fid.close()

voxel_diagonals = np.linalg.norm(data["voxel_dims"], axis=1)
voxel_minimum = np.min(data["voxel_dims"], axis=1)
rms2precision = data["rms"] / voxel_diagonals

f, ax = plt.subplots()
f.set_size_inches([3.34646, 3.34646])
ax.plot(data["rms"], 'bo', markersize=2)
ax.fill_between(np.arange(voxel_diagonals.size), voxel_minimum,
                voxel_diagonals, facecolor='green', alpha=0.5, linewidth=0.0)
ax.fill_between(np.arange(voxel_diagonals.size), 0, voxel_minimum,
                facecolor='orange', alpha=0.5, linewidth=0.0)
ax.set_ylabel("RMS Distance Error")
ax.set_xlabel("Object Pair")
plt.savefig("rms_error.svg")
