import os
import h5py
import numpy as np
from tqdm import tqdm
import pylab as plt
import seaborn as sns

#name = "RMSProp"
name = "ADAM"
name = "GradientDescent"


cutoff = 40000
total_frames = 600

trail_iterations = 200
skip_frames = 10

save_dest = "zero_data"
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


def plot_frame(k):

    x_pts = x[:,k:k+trail_iterations]
    y_pts = y[:,k:k+trail_iterations]

    # Only keep the points that converge
    idx_x = np.isclose(x_pts.sum(axis=1),0,1e-2)
    idx_y = np.isclose(y_pts.sum(axis=1),0,1e-2)
    idx = idx_x & idx_y
    x_pts = x_pts[~idx]
    y_pts = y_pts[~idx]

    print(f"Plotting frame {k}, {x_pts.shape}")

    '''
    # If we want to have a color trail, this slows it down a lot!
    pts_N = x_pts.ravel().size
    alpha = np.linspace(0,1, x_pts.shape[1]).tolist()*x_pts.shape[0]
    alpha = np.array(alpha)

    rgba_colors = np.zeros((pts_N,4))
    rgba_colors[:,0] = 0.4
    rgba_colors[:,2] = 0.8
    rgba_colors[:,3] = alpha
    pfig.set_color(rgba_colors)
    '''

    pts = np.array([x_pts.ravel(), y_pts.ravel()]).T
    pfig.set_offsets(pts)

    f_png = os.path.join(figure_dest, f'{k:06d}.png')
    plt.savefig(f_png, pad_inches=0.0)

    #plt.show()
    #exit()

plt.figure(figsize=(6,6))
pfig = plt.scatter([], [], lw=0,s=.1)

plt.axis('equal')
plt.axis('off')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0.0)

ITR = range(
    trail_iterations,
    total_frames+n_iters,
    skip_frames
)

for k in tqdm(ITR):
    plot_frame(k)





