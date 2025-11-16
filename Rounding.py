import numpy as np

def cholesky_psd(A, eps=1e-12):
    """
    Robust Cholesky for Hermitian PSD matrices that may have small
    negative eigenvalues from numerical noise.

    Steps:
      1) Symmetrize A
      2) Project to PSD by clipping eigenvalues
      3) Standard Cholesky
    """
    # 1. Force exact Hermitian
    A = (A + A.conj().T) / 2

    # 2. Eigenvalue repair
    w, V = np.linalg.eigh(A)
    w_clipped = np.maximum(w, eps)   # eliminate small negative pivots
    A_psd = V @ np.diag(w_clipped) @ V.conj().T

    # 3. Cholesky on repaired matrix
    return np.linalg.cholesky(A_psd)


def round_sdp_with_cholesky(M, num_rounds=1):
    """
    Round an SDP solution using Cholesky decomposition.
    (No actual rounding for QMC sind already in n dimensions)
    """
    n = M.shape[0]

    # Get Cholesky decomposition: M = L L^T
    L = cholesky_psd(M)

    # The vectors are the rows of L^T (or columns of L)
    # Each row i of L^T is the vector for vertex i
    V = L.T

    # Verify our decomposition (optional check)
    # print(f"Reconstruction error: {np.max(np.abs(V.T @ V - M)):.6f}")

    cuts = []
    for _ in range(num_rounds):
        # Random hyperplane rounding
        # Pick a random direction in the embedding space
        random_normal = np.random.randn(V.shape[1])
        random_normal /= np.linalg.norm(random_normal)

        # Project each vector onto this direction
        projections = V @ random_normal

        # Partition based on sign
        partition = (projections > 0).astype(int)
        cuts.append(partition)

    return cuts[0] if num_rounds == 1 else cuts