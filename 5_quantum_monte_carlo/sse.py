"""Python implementation of the stochastic series expansion (SSE) for the Heisenberg model.

This code was implemented based on Anders Sandvik's tutorial at
http://physics.bu.edu/~sandvik/programs/index.html
and chapter 5.2 of his lecture notes, arXiv:1101.3281, adjusted for python.
Note the different numbering of certain things: fortran starts counting with 1.
For that reason, we also switch some of the "magic" constants
(e.g. which numbers represent and "identity" operator in the operator string).

!!! Note that the pure python implementation is very slow. !!!
For that reason, if the package "numba" is available, some functions are
just-in-time compiled. This brings a significant speed-up of almost a factor of 100!


Explanations
------------

We simulate a nearest-neighbor antiferromagnetic Heisenberg model on a square lattice with
periodic boundary conditions; the coupling strength 'J' is set to 1 and used as a unit of energy.
The Hamiltonian reads
    H = J sum_bons SxSx + SySy + SzSz
      = N_bonds/4 + sum_bonds (SxSx+SySy)  - (1/4 - SzSz)
      = N_bonds/4 + sum_b (H^off_b - H^diag_b)
The constant ensures H^diag_b >= 0 (and also H^off_b >= 0).


In the stochastic series expansion, we expand the partition function as
    Z = tr(exp(-beta H) ~= tr(exp(beta sum_b (H^diag_b - H^off_b)))
      = sum_{|alpha>} <alpha| sum_n 1/n! (beta sum_b (H^diag_b - H^off_b))^n |alpha>
      = sum_{|alpha>} sum_n sum_{a(p),b(p) for p=0,...,n-1} beta^n/n! (-1)^{n_offdiag} <alpha| prod_{p=0}^{M-1} H^a(p)_b(p)|alpha>
      = sum_{|alpha>} sum_{a(p),b(p) for p=0,...,M-1} beta^n (M-n)!/M! <alpha| prod_{p=0}^{M-1} H^a(p)_b(p)|alpha>

The (-1)^{n_offdiag} vanishes, because the number of offdiagonal terms needs to be even on
bipartite lattices. (The square lattice is bipartite...)

Looking at the last line of the above equations, we need to sample
spin configurations {|alpha>} and "operator strings" {a(p),b(p)} of a fixed length `M`.
This length `M` is chosen large enough that we actually never encounter configurations
with n=M non-identity operators.
Here, a(p) = identity, diag, offdiag, and b(p) labels bonds. In the code, they are combined into
a single number
    op = {-1        for the identity operator
         {2*b(p)    for an H^diag_b(p)
         {2*b(p)+1  for an H^off_b(p)
To get a(p), we can check op == -1 (identity?) and otherwise mod(op, 2) == 0 (diagonal/offdiagonal?)
and use b(p) = op // 2  (with integer division rounding down).

The spin configuration |alpha> is given by a 1D array with values +-1.
Sites are enumerated, bonds are given by two numbers specifying which sites they connect --
this separates the geometry of the lattice completely from the implementation of updates.
"""

import numpy as np

from numba import jit

def site(x, y, Lx, Ly):
    """Defines a numbering of the sites, given positions x and y."""
    return y * Lx + x


def init_SSE_square(Lx, Ly):
    """Initialize a starting configuration on a 2D square lattice."""
    n_sites = Lx*Ly
    # initialize spins randomly with numbers +1 or -1, but the average magnetization is 0
    spins = 2*np.mod(np.random.permutation(n_sites), 2) - 1
    op_string = -1 * np.ones(10, np.intp)  # initialize with identities

    bonds = []
    for x0 in range(Lx):
        for y0 in range(Ly):
            s0 = site(x0, y0, Lx, Ly)
            s1 = site(np.mod(x0+1, Lx), y0, Lx, Ly) # bond to the right
            bonds.append([s0, s1])
            s2 = site(x0, np.mod(y0+1, Ly), Lx, Ly) # bond to the top
            bonds.append([s0, s2])
    bonds = np.array(bonds, dtype=np.intp)
    return spins, op_string, bonds


@jit(nopython=True)
def diagonal_update(spins, op_string, bonds, beta):
    """Perform the diagonal update: insert or remove diagonal operators into/from the op_string."""
    n_bonds = bonds.shape[0]
    M = op_string.shape[0]
    # count the number of non-identity operators
    n = np.sum(op_string != -1)
    # calculate ratio of acceptance probabilities for insert/remove  n <-> n+1
    # <alpha|Hdiag|alpha> = 1/4 + <alpha |SzSz|alpha> = 0.5 for antiparallel spins
    prob_ratio = 0.5*beta*n_bonds  # /(M-n) , but the latter depends on n which still changes
    for p in range(M):  # go through the operator string
        op = op_string[p]
        if op == -1:  # identity: propose to insert a new operator
            b = np.random.randint(0, n_bonds)  # select a bond
            if spins[bonds[b, 0]] != spins[bonds[b, 1]]:
                # can only insert if the two spins are anti-parallel!
                prob = prob_ratio / (M - n)
                if np.random.rand() < prob:  # (metropolis-like)
                    # insert a diagonal operator
                    op_string[p] = 2*b
                    n += 1
        elif np.mod(op, 2) == 0:  # diagonal operator: propose to remove
            prob = 1/prob_ratio * (M-n+1)  # n-1 = number operators after removal = n in above formula
            if np.random.rand() < prob:
                # remove diagonal operator
                op_string[p] = -1
                n -= 1
        else:  # offdiagonal operator: update spin configuration to get propagated |alpha(p)>
            b = op // 2
            # H^off ~= (S+S- + S-S+) = spin flip on both sites for antiparallel spins.
            # (We never have configurations with operators acting on parallel spins!)
            spins[bonds[b, 0]] = -spins[bonds[b, 0]]
            spins[bonds[b, 1]] = -spins[bonds[b, 1]]
    return n


@jit(nopython=True)
def loop_update(spins, op_string, bonds):
    """Perform the offdiagonal update: construct loops and flip each of them with prob. 0.5."""
    # create the loops
    vertex_list, first_vertex_at_site = create_linked_vertex_list(spins, op_string, bonds)
    # and flip them
    flip_loops(spins, op_string, vertex_list, first_vertex_at_site)


@jit(nopython=True)
def create_linked_vertex_list(spins, op_string, bonds):
    """Given a configuration, construct a linked list between vertices defining the loops.

    Given a configuration of spins and operators, we need to construct the loops.
    An efficient way to do this is to create a double-linked `vertex_list` which contains
    the connections between the vertices of the operators. Each operator has 4 vertices (=legs in
    the tensor network language), so if we simply enumerate all the vertices in the operator
    string, we get v0 = 4*p, v1=4*p+1, v2=4*p+2, v4=4*p+3 for the vertices

        v0  v1
         |--|
         |Op|  <-- op_string[p]
         |--|
        v2  v3

    In this function, we set the entries of the `vertex_list` for any
    (vertically) connected pair `v, w` (i.e. vertical parts of the loops) we have
    ``v = vertex_list[w]`` and ``w = vertex_list[v]``.
    Later on, an entry -1 indicates that the loop along this connection was flipped;
    an entry -2 indices that the loop was visited and proposed to flip, but the flip was rejected.
    Identity operators are completely ignored for the connections, its vertices are directly
    marked with a -2.

    The returned array `first_vertex_at_site` contains the first vertex encountered at each site,
    entries -1 indicate that there is no (non-identity) operator acting on that site.
    """
    n_sites = spins.shape[0]
    M = op_string.shape[0]
    vertex_list = np.zeros(4*M, np.intp)
    # (initial value of vertex_list doesn't matter: get's completely overwritten)
    first_vertex_at_site = -1 * np.ones(n_sites, np.intp) # -1 = no vertex found (yet)
    last_vertex_at_site = -1 * np.ones(n_sites, np.intp) # -1 = no vertex found (yet)

    # iterate over all operators
    for p in range(M):
        v0 = p*4  # left incoming vertex
        v1 = v0 + 1  # right incoming vertex
        op = op_string[p]
        if op == -1:  # identity operator
            # ignore it for constructing/flipping loops: mark as visited
            vertex_list[v0:v0+4] = -2
        else:
            b = op//2
            s0 = bonds[b, 0]
            s1 = bonds[b, 1]
            v2 = last_vertex_at_site[s0]
            v3 = last_vertex_at_site[s1]
            if v2 == -1:  # no operator encountered at this site before
                first_vertex_at_site[s0] = v0
            else:  # encountered an operator at this vertex before -> create link
                vertex_list[v2] = v0
                vertex_list[v0] = v2
            if v3 == -1:   # and similar for other site
                first_vertex_at_site[s1] = v1
            else:
                vertex_list[v3] = v1
                vertex_list[v1] = v3
            last_vertex_at_site[s0] = v0 + 2  # left outgoing vertex of op
            last_vertex_at_site[s1] = v0 + 3  # right outgoing vertex of op

    # now we need to connect vertices between top and bottom
    for s0 in range(n_sites):
        v0 = first_vertex_at_site[s0]
        if v0 != -1:  # there is an operator acting on that site -> create link
            v1 = last_vertex_at_site[s0]
            vertex_list[v1] = v0
            vertex_list[v0] = v1
    return vertex_list, first_vertex_at_site


@jit(nopython=True)
def flip_loops(spins, op_string, vertex_list, first_vertex_at_site):
    """Given the vertex_list, flip each loop with prob. 0.5.

    Once we have the vertex list, we can go through all the vertices and flip each loop with
    probability 0.5. When we propose to flip a loop, we go through it and mark it as flipped (-1)
    or visited (-2) in the vertex list to avoid a secend proposal to flip it.

    Note that for an integer number `i`, the operation ``i ^ 1`` gives i+1 or i-1 depending on
    whether `i` is even or odd: it flips 0<->1, 2<->3, 4<->5, ...
    This is used to switch between diagonal/offdiagonal operators in the operator string when
    flipping a loop, and to propagate the open end of the loop vertically between vertices
    v0<->v1, v2<->v3 of the operators.
    """
    n_sites = spins.shape[0]
    M = op_string.shape[0]
    # iterate over all possible beginnings of loops
    # (step 2: v0+1 belongs to the same loop as v0)
    for v0 in range(0, 4*M, 2):
        if vertex_list[v0] < 0:  # marked: we've visited the loop starting here before.
            continue
        v1 = v0  # we move v1 as open end of the loop around until we come back to v0
        if np.random.rand() < 0.5:
            # go through the loop and flip it
            while True:
                op = v1 // 4
                op_string[op] = op_string[op] ^ 1  # flip diagonal/offdiagonal
                vertex_list[v1] = -1
                v2 = v1 ^ 1
                v1 = vertex_list[v2]
                vertex_list[v2] = -1
                if v1 == v0:
                    break
        else:
            # don't flip the loop, but go through it to mark it as visited
            while True:
                vertex_list[v1] = -2
                v2 = v1 ^ 1
                v1 = vertex_list[v2]
                vertex_list[v2] = -2
                if v1 == v0:
                    break
    for s0 in range(0, n_sites):
        if first_vertex_at_site[s0] == -1:  # no operator acting on that site -> flip with p=0.5
            if np.random.rand() < 0.5:
                spins[s0] = -spins[s0]
        else:  # there is an operator acting on that site
            if vertex_list[first_vertex_at_site[s0]] == -1:  # did we flip the loop?
                spins[s0] = -spins[s0]  # then we also need to flip the spin
    # done


def thermalize(spins, op_string, bonds, beta, n_updates_warmup):
    """Perform a lot of upates to thermalize, without measurements."""
    if beta == 0.:
        raise ValueError("Simulation doesn't work for beta = 0")
    for _ in range(n_updates_warmup):
        n = diagonal_update(spins, op_string, bonds, beta)
        loop_update(spins, op_string, bonds)
        # check if we need to increase the length of op_string
        M_old = len(op_string)
        M_new = n + n // 3
        if M_new > M_old:
            op_string = np.resize(op_string, M_new)
            op_string[M_old:] = -1
    return op_string


def measure(spins, op_string, bonds, beta, n_updates_measure):
    """Perform a lot of updates with measurements."""
    ns = []
    for _ in range(n_updates_measure):
        n = diagonal_update(spins, op_string, bonds, beta)
        loop_update(spins, op_string, bonds)
        ns.append(n)
    return np.array(ns)


def run_simulation(Lx, Ly, betas=[1.], n_updates_measure=10000, n_bins=10):
    """A full simulation: initialize, thermalize and measure for various betas."""
    spins, op_string, bonds = init_SSE_square(Lx, Ly)
    n_sites = len(spins)
    n_bonds = len(bonds)
    Es_Eerrs = []
    for beta in betas:
        print("beta = {beta:.3f}".format(beta=beta), flush=True)
        op_string = thermalize(spins, op_string, bonds, beta, n_updates_measure//10)
        Es = []
        for _ in range(n_bins):
            ns = measure(spins, op_string, bonds, beta, n_updates_measure)
            # energy per site
            E = (-np.mean(ns)/beta + 0.25*n_bonds) / n_sites
            Es.append(E)
        E, Eerr = np.mean(Es), np.std(Es)/np.sqrt(n_bins)
        Es_Eerrs.append((E, Eerr))
    return np.array(Es_Eerrs)


if __name__ == "__main__":
    # parameters
    beta = 1.
    Es_Eerrs = run_simulation(4, 4, [1.])
    print("Energy per site ={E:.8f} at T={T:.3f}".format(E=Es_Eerrs[0,0], T=1./beta))
