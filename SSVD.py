import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def compute_preconditioner(k, A):
    omega = np.random.standard_normal(size = (A.shape[1],k))
    Y = np.dot(A,omega)
    return linalg.qr(Y)

def compute_SSVD(A, k):
    Q, _ = compute_preconditioner(k, A)
    B = np.dot(Q.transpose(), A)
    U_hat, S_squared, U_hat_transpose = np.linalg.svd(np.dot(B, B.transpose()))
    U = np.dot(Q, U_hat)
    S = np.diag(np.sqrt(S_squared))
    V_hat_transpose = linalg.inv(S) @ U_hat_transpose @ B
    return U, S, V_hat_transpose[:k,:]

from sklearn.datasets import load_digits
digits = load_digits()

plt.plot(linalg.svd(digits.data)[1])
plt.show()

plt.plot(np.diag(compute_SSVD(digits.data, 20)[1][:64]))
plt.show()

k = 64 # 8x8 pixels
pca = PCA(n_components=k, svd_solver='randomized')
plt.plot(pca.fit(digits.data).singular_values_)
plt.show()