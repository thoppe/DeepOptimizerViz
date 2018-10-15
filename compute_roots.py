import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import h5py

np.random.seed(42)
n_poly = 10
coeffs = [a + b * 1J for a, b in np.random.normal(size=(n_poly, 2))]

N = 10**5
n_iters = 4000
name = "RMSProp"
name = "ADAM"
name = "GradientDescent"
name = "FTRL"
tolerance = 0.01

optimizers = {
    "GradientDescent": tf.train.GradientDescentOptimizer(0.0001),

    "Momentum": tf.train.MomentumOptimizer(0.001, 0.001),
    "ADAM": tf.train.AdamOptimizer(0.001),
    "FTRL": tf.train.FtrlOptimizer(0.01),
    "RMSProp": tf.train.RMSPropOptimizer(0.001),

    "Adagrad": tf.train.AdagradOptimizer(0.01),

    "ProximalAdagrad": tf.train.ProximalAdagradOptimizer(0.01),
    "ProximalGradientDescent":
        tf.train.ProximalGradientDescentOptimizer(0.001),

    # These don't work yet
    # "AdagradDA" : tf.train.AdagradDAOptimizer(0.001),
    # "Adadelta" : tf.train.AdadeltaOptimizer(0.1),
}


def sample_model(name, N, n_iters):

    opt = optimizers[name]

    x = tf.complex(
        tf.Variable(tf.random_normal([N, ], mean=0, stddev=5.0)),
        tf.Variable(tf.random_normal([N, ], mean=0, stddev=5.0)),
    )

    # quadratic = x**2 + 1
    poly = 0
    for k, coeff in enumerate(coeffs[::-1]):
        poly += coeff * tf.pow(x, k)

    term_error = tf.abs(poly)
    loss = tf.reduce_sum(term_error)
    train_op = opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    sols = []

    for n in tqdm(range(n_iters)):
        _, x_val, err = sess.run([train_op, x, term_error])

        converged_idx = err < tolerance
        print(
            "Fraction converged {:0.3f}".format(converged_idx.mean()))

        sols.append(x_val)

    sols = np.array(sols)
    error = sess.run(term_error)

    return sols, error


save_dest = "zero_data"
os.system(f'mkdir -p {save_dest}')
f_h5 = os.path.join(save_dest, f"{name}_zeros.h5")

# if not os.path.exists(f_h5):
if True:
    sols, error = sample_model(name, N, n_iters)

    # Only keep those in tolerance
    idx = error < tolerance
    sols = sols[:, idx]
    error = error[idx]

    with h5py.File(f_h5, 'w') as h5:
        h5.attrs["N"] = N
        h5.attrs["n_iters"] = n_iters
        h5.attrs["tolerance"] = tolerance

        sols = sols.T

        h5['real'] = sols.real
        h5['imag'] = sols.imag
        h5['error'] = error

        print(h5['real'][...].shape)
        print(h5['imag'][...].shape)
