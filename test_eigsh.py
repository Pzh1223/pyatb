import numpy as np
from scipy.sparse.linalg import eigsh

n = 100
A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
A = A + A.T.conj()
M = np.eye(n)

# eigsh works for complex Hermitian matrices
try:
    w, v = eigsh(A, k=5, M=M, sigma=0.0, which='LM')
    print("eigsh worked for complex!")
except Exception as e:
    print("Error:", e)
