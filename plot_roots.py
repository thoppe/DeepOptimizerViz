import os
import h5py
import pylab as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

name = "RMSProp"
save_dest = "zero_data"
f_h5 = os.path.join(save_dest, f"{name}_zeros.h5")

figure_dest = 'images'
os.system(f'mkdir -p {figure_dest}')

assert(os.path.exists(f_h5))

cutoff = 2000
total_frames = 20000
trail_iterations = 1000

with h5py.File(f_h5,'r') as h5:
    print(h5['real'].shape)
    x = h5['real'][:cutoff]
    y = h5['imag'][:cutoff]
    n_iters = h5.attrs['n_iters']
    N = h5.attrs['N']



print(x.shape)
x = np.pad(x,((0,0), (0,total_frames-n_iters)), mode='constant')
y = np.pad(y,((0,0), (0,total_frames-n_iters)), mode='constant')

rl = np.random.randint(0, total_frames-n_iters-10, size=[N,])
x = np.array([np.roll(_, k) for _,k in zip(x, rl)])
y = np.array([np.roll(_, k) for _,k in zip(y, rl)])

def plot_frame(k):
    plt.clf()
    plt.cla()

    
    x_pts = x[:,k:k+trail_iterations]
    y_pts = y[:,k:k+trail_iterations]

    idx = (
        np.isclose(x_pts.sum(axis=1),0,1e-2) &
        np.isclose(y_pts.sum(axis=1),0,1e-2)
    )

    x_pts = x_pts[~idx]
    y_pts = y_pts[~idx]

    print(f"Plotting frame {k}, {x_pts.shape}")

    pts_N = x_pts.ravel().size
    #alpha = np.logspace(1, 2, x_pts.shape[1]).tolist()*x_pts.shape[0]
    alpha = np.linspace(0,1, x_pts.shape[1]).tolist()*x_pts.shape[0]
    alpha = np.array(alpha)
    #alpha /= alpha.max()

    rgba_colors = np.zeros((pts_N,4))
    rgba_colors[:,0] = 0.4
    rgba_colors[:,2] = 0.8
    rgba_colors[:,3] = alpha

    pfig = plt.scatter(x_pts, y_pts, color=rgba_colors, lw=0,s=.1)

    plt.axis('equal')
    plt.axis('off')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    #plt.tight_layout()

    f_png = os.path.join(figure_dest, f'{k:06d}.png')
    plt.savefig(f_png)

plt.figure(figsize=(6,6))

for k in tqdm(range(0, 8000, 10)):
    plot_frame(k)

