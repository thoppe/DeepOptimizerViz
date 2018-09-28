import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pylab as plt

N = 1000
n_iters = 2000
name = "GradientDescent"
tolerance = 0.001

optimizers = {
    "GradientDescent" : tf.train.GradientDescentOptimizer(0.001),

    "Momentum" : tf.train.MomentumOptimizer(0.001, 0.001),
    "ADAM" : tf.train.AdamOptimizer(0.001),
    "FTRL" : tf.train.FtrlOptimizer(0.01),
    "RMSProp" : tf.train.RMSPropOptimizer(0.001),


    "Adagrad" : tf.train.AdagradOptimizer(0.01),

    "ProximalAdagrad" : tf.train.ProximalAdagradOptimizer(0.01),
    "ProximalGradientDescent":tf.train.ProximalGradientDescentOptimizer(0.001),

    # These don't work yet
    #"AdagradDA" : tf.train.AdagradDAOptimizer(0.001),
    # "Adadelta" : tf.train.AdadeltaOptimizer(0.1),
}

def sample_model(name, N, n_iters):

    opt = optimizers[name]

    x = tf.complex(
        tf.Variable(tf.random_normal([N,], mean=0, stddev=3.0)),
        tf.Variable(tf.random_normal([N,], mean=0, stddev=3.0)),
    )

    quadratic = x**2 + 1
    term_error = tf.abs(quadratic)
    loss = tf.reduce_sum(term_error)
    train_op = opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    sols = []

    for n in tqdm(range(n_iters)):
        _, x_val = sess.run([train_op, x])
        sols.append(x_val)

    sols = np.array(sols)
    error = sess.run(term_error)

    print("Fraction converged {:0.3f}".format((error<tolerance).mean()))
    return sols

def plot_solution(sols):
    fig = plt.figure(figsize=(4.5,4.5))

    for row in tqdm(sols.T):
        alpha = np.linspace(0.1, 1, len(row))

        rgba_colors = np.zeros((len(row),4))
        rgba_colors[:,0] = 1.0
        rgba_colors[:, 3] = alpha

        plt.scatter(row.real, row.imag, color=rgba_colors, lw=0,s=.1)
    #plt.title(f"Loss {result[-1]/N:0.3f}")
    plt.axis('square')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.tight_layout()
    #plt.show()
    #exit()
    return fig



for name in optimizers:
    sols = sample_model(name, N, n_iters)
    fig = plot_solution(sols)
    fig.savefig(f"figures/{name}_quadratic_roots.png")

#plt.show()
