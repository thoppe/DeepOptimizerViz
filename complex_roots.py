import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Fails, variable can't be complex type
#x = tf.Variable(-1.0, dtype=tf.complex64)
#x = tf.Variable(-1.0 + 0j)

#x = tf.complex(tf.Variable(-1.0), tf.Variable(3.0))

#x = tf.complex(
#    tf.Variable(tf.random_normal([1,])),
#    tf.Variable(tf.random_normal([1,])),
#)

N = 300
n_iters = 1000

x = tf.complex(
    tf.Variable(tf.random_normal([N,], mean=0, stddev=3.0)),
    tf.Variable(tf.random_normal([N,], mean=0, stddev=3.0)),
)


quadratic = x**2 + 1
loss = tf.reduce_sum(tf.abs(quadratic))

#opt = tf.train.GradientDescentOptimizer(0.001)
#opt = tf.train.AdamOptimizer(0.001)
#opt = tf.train.FtrlOptimizer(0.01)
opt = tf.train.RMSPropOptimizer(0.001)

train_op = opt.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

result = []
sols = []

for n in tqdm(range(n_iters)):
    _, lx, x_val = sess.run([train_op, loss, x])
    result.append(lx)
    sols.append(x_val)

    
result = np.array(result)
sols = np.array(sols)

import pylab as plt
for row in tqdm(sols.T):
    alpha = np.linspace(0.1, 1, len(row))

    rgba_colors = np.zeros((len(row),4))
    rgba_colors[:,0] = 1.0
    rgba_colors[:, 3] = alpha

    plt.scatter(row.real, row.imag, color=rgba_colors, lw=0,s=.1)
#plt.title(f"Loss {result[-1]/N:0.3f}")


plt.xlim(-2,2)
plt.ylim(-2,2)

plt.tight_layout()
#plt.savefig("figures/GradientDescentOptimizer_quadratic_roots.png")
#plt.savefig("figures/AdamOptimizer_quadratic_roots.png")
#plt.savefig("figures/FtrlOptimizer_quadratic_roots.png")
plt.savefig("figures/RMSPropOptimizer_quadratic_roots.png")

plt.show()



exit()

#plt.plot(result)
#plt.ylim(ymin=0)
#plt.show()

#print(result)


'''
import pylab as plt
import numpy as np
x = np.linspace(-3,3,100)
y = x**2 - x + 1
plt.plot(x,y)
plt.ylim(-1,3)
plt.show()
'''
