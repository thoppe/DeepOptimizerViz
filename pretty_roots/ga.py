import mpmath as mp
import numpy as np
import joblib
from tqdm import tqdm
import h5py, os


N = 10**3
n_poly = 10
name = 'uniform'

save_dest = "zeros_data"
os.system(f'mkdir -p {save_dest}')

f_h5 = os.path.join(
    save_dest,
    f"{name}_{n_poly}_{span:0.3f}_{shift:0.3f}.h5"
)
if os.path.exists(f_h5):
    print (f"Already computed {f_h5}")
    exit()

coeff_funcs = {
    'uniform' : np.random.uniform
}

span = 1.0
shift = 0.3

coeffs = coeff_funcs[name](-span/2+shift,span/2+shift,size=[N, n_poly])

dfunc = joblib.delayed(mp.polyroots)
with joblib.Parallel(1) as MP:
    ITR = tqdm(coeffs)
    mproots = MP(dfunc(p) for p in ITR)

roots = np.array([complex(root) for p in mproots for root in p])

X = roots.real
Y = roots.imag

import pylab as plt
import seaborn as sns
plt.scatter(X,Y,s=.1)
plt.axis('tight')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()
