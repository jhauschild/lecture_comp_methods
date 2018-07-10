"""Exact diagonalization code for the transverse field Ising model with momentum conservation.

H = -J sum_i sigma^x_i sigma^x_{i+1} - g sum_i sigma^z i; periodic boundary cond.

"""

import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt


def flip(s, i, N):
    """Flip the bits of the state `s` at positions i and (i+1)%N."""
    return s ^ (1 << i | 1 << ((i+1) % N))


def translate(s, N):
    """Shift the bits of the state `s` one position to the right (cyclically for N bits)."""
    bs = bin(s)[2:].zfill(N)
    return int(bs[-1] + bs[:-1], base=2)


def count_ones(s, N):
    """Count the number of `1` in the binary representation of the state `s`."""
    return bin(s).count('1')


def is_representative(s, k, N):
    """Check if |s> is the representative for the momentum state.

    Returns -1 if s is not a representative.
    If |s> is a representative, return the periodicity R,
    i.e. the smallest integer R > 0 such that T**R |s> = |s>."""
    t = s
    for i in range(N):
        t = translate(t, N)
        if t < s:
            return -1  # there is a smaller representative in the orbit
        elif (t == s):
            if (np.mod(k, N/(i+1)) != 0):
                return -1  # periodicty incompatible with k
            else:
                return i+1


def get_representative(s, N):
    """Find the representative r in the orbit of s and return (r, l) such that |r>= T**l|s>"""
    r = s
    t = s
    l = 0
    for i in range(N):
        t = translate(t, N)
        if (t < r):
            r = t
            l = i + 1
    return r, l


def calc_basis(N):
    """Determine the (representatives of the) basis for each block.

    A block is detemined by the quantum numbers `qn`, here simply `k`.
    `basis` and `ind_in_basis` are dictionaries with `qn` as keys.
    For each block, `basis[qn]` contains all the representative spin configurations `sa`
    and periodicities `Ra` generating the state
    ``|a(k)> = 1/sqrt(Na) sum_l=0^{N-1} exp(i k l) T**l |sa>``

    `ind_in_basis[qn]` is a dictionary mapping from the representative spin configuration `sa`
    to the index within the list `basis[qn]`.
    """
    basis = dict()
    ind_in_basis = dict()
    for sa in range(2**N):
        for k in range(-N//2+1, N//2+1):
            qn = k
            Ra = is_representative(sa, k, N)
            if Ra > 0:
                if qn not in basis:
                    basis[qn] = []
                    ind_in_basis[qn] = dict()
                ind_in_basis[qn][sa] = len(basis[qn])
                basis[qn].append((sa, Ra))
    return basis, ind_in_basis


def calc_H(N, J, g):
    """Determine the blocks of the Hamiltonian as scipy.sparse.csr_matrix."""
    print("Generating Hamiltonian ... ", end="", flush=True)
    basis, ind_in_basis = calc_basis(N)
    H = {}
    for qn in basis:
        M = len(basis[qn])
        H_block_data = []
        H_block_inds = []
        a = 0
        for sa, Ra in basis[qn]:
            H_block_data.append(-g * (-N + 2*count_ones(sa, N)))
            H_block_inds.append((a, a))
            for i in range(N):
                sb, l = get_representative(flip(sa, i, N), N)
                if sb in ind_in_basis[qn]:
                    b = ind_in_basis[qn][sb]
                    Rb = basis[qn][b][1]
                    k = qn*2*np.pi/Ra
                    H_block_data.append(-J*np.exp(-1j*k*l)*np.sqrt(Ra/Rb))
                    H_block_inds.append((b, a))
                # else: flipped state incompatible with the k value, |b(k)> is zero
            a += 1
        H_block_inds = np.array(H_block_inds)
        H_block_data = np.array(H_block_data)
        H_block = scipy.sparse.csr_matrix((H_block_data, (H_block_inds[:, 0], H_block_inds[:, 1])),
                                          shape=(M,M),dtype=np.complex)
        H[qn] = H_block
    print("done", flush=True)
    return H
