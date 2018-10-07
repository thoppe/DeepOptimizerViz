import numpy as np
import h5py
import os


class dataset_loader():

    def __init__(
            self,
            f_h5,
            cutoff=5000,
            offset=0,
            
            total_frames=600,
            trail_iterations=200,

            extent_x=2.0,
            extent_y=2.0,
            width=512,
            height=512,
    ):

        self.extent_x = extent_x
        self.extent_y = extent_y
        self.width = width
        self.height = height

        assert(os.path.exists(f_h5))

        # Read the data in from complex_roots2.py
        with h5py.File(f_h5, 'r') as h5:
            print(h5['real'].shape)
            x = h5['real'][offset:offset+cutoff]
            y = h5['imag'][offset:offset+cutoff]
            n_iters = h5.attrs['n_iters']

        self.trail_iterations = trail_iterations

        # Pad the start and end so we have a nice trail
        # pad_dim = ((0,0), (trail_iterations,total_frames-n_iters))
        pad_dim = ((0, 0), (trail_iterations, trail_iterations + n_iters))

        x = np.pad(x, pad_dim, mode='constant')
        y = np.pad(y, pad_dim, mode='constant')

        # Stagger the lines so they don't all start at once
        rl = np.random.randint(
            trail_iterations,
            total_frames - trail_iterations, size=[len(x), ])

        x = np.array([np.roll(_, k) for _, k in zip(x, rl)])
        y = np.array([np.roll(_, k) for _, k in zip(y, rl)])

        # Clip the start, end of the roll
        clip_start = total_frames
        clip_end = n_iters - trail_iterations
        x = x[:, clip_start:][:, :clip_end]
        y = y[:, clip_start:][:, :clip_end]

        '''
        print(x.sum(axis=0))
        q = x.sum(axis=0)
        import pylab as plt
        plt.plot(q)
        plt.show()
        #exit()
        #x = x[:, :-trail_iterations]
        #y = y[:, :-trail_iterations]
        '''

        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, k):

        x_pts = self.x[:, k:k + self.trail_iterations]
        y_pts = self.y[:, k:k + self.trail_iterations]

        # Only keep the points that converge
        idx_x = np.isclose(x_pts.sum(axis=1), 0, 1e-2)
        idx_y = np.isclose(y_pts.sum(axis=1), 0, 1e-2)
        idx = idx_x & idx_y
        x_pts = x_pts[~idx]
        y_pts = y_pts[~idx]

        return self.transform_points(x_pts, y_pts)

    def transform_points(self, x, y):

        x *= self.width / 2.0
        x /= self.extent_x
        x += self.width / 2

        y *= -self.height / 2.0
        y /= self.extent_y
        y += self.height / 2

        return x, y
