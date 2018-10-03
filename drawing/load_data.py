import numpy as np
import h5py
import os

#name = "RMSProp"
#name = "GradientDescent"
name = "ADAM"


cutoff = 5000
total_frames = 600

trail_iterations = 200
skip_frames = 10

save_dest = "../zero_data"
f_h5 = os.path.join(save_dest, f"{name}_zeros.h5")

figure_dest = os.path.join('images', name)
os.system(f'mkdir -p {figure_dest}')

assert(os.path.exists(f_h5))

# Read the data in from complex_roots2.py
with h5py.File(f_h5,'r') as h5:
    print(h5['real'].shape)
    x = h5['real'][:cutoff]
    y = h5['imag'][:cutoff]
    n_iters = h5.attrs['n_iters']


# Pad the start and end so we have a nice trail
#pad_dim = ((0,0), (trail_iterations,total_frames-n_iters))
pad_dim = ((0,0), (trail_iterations, trail_iterations+n_iters))

x = np.pad(x, pad_dim, mode='constant')
y = np.pad(y, pad_dim, mode='constant')

# Stagger the lines so they don't all start at once
rl = np.random.randint(
    trail_iterations,
    total_frames-trail_iterations, size=[len(x),])

x = np.array([np.roll(_, k) for _,k in zip(x, rl)])
y = np.array([np.roll(_, k) for _,k in zip(y, rl)])


def get_frame(k):

    x_pts = x[:,k:k+trail_iterations]
    y_pts = y[:,k:k+trail_iterations]

    # Only keep the points that converge
    idx_x = np.isclose(x_pts.sum(axis=1),0,1e-2)
    idx_y = np.isclose(y_pts.sum(axis=1),0,1e-2)
    idx = idx_x & idx_y
    x_pts = x_pts[~idx]
    y_pts = y_pts[~idx]

    return x_pts, y_pts
