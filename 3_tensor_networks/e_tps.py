"""Toy code implementing the simplified update for PEPS and a contraction with boundary MPS."""

import numpy as np
from scipy.linalg import svd


from a_mps import MPS, split_truncate_theta
from c_tebd import calc_U_bonds


class TPS:
    """Class for a tensor product state with coordination number z=3.

    The legs i and j are physical (sticking out of the plane), other legs are "virtual bonds"
    Connecting the bonds in the plane yields a honeycomb lattice.
    (The simplified updates corresponds to a Bethe lattice, though.)

    #    b                       c'
    #     \                     /
    #     (G[0])--a    a'--(G[1])
    #     /   |             |   \
    #    c    i             j    b'


    Parameters
    ----------
    d : int
        Local dimension
    state : (int, int)
        The state on site A and B.

    Attributes
    ----------
    z : int
        Coordination number, i.e. number of virtual bonds, e.g. z=3 for the honey comb lattice.
    Gs : list of 2 np.Array[ndim=1+z]
        The two gamma tensors, with legs ``i a b c`` and ``j a b c ...``.
    Ss : list of z np.Array[ndim=1]
        The Schmidt values at each of the bonds, ``Ss[j]`` is on bond ``j``.
    """

    def __init__(self, d, state):
        self.z = z = 3
        self.d = d
        assert len(state) == 2
        self.Gs = []
        for st_i in state:
            G_i = np.zeros([d] + [1] * z)
            G_i[st_i, 0, ...] = 1.
            self.Gs.append(G_i)
        self.Ss = [np.ones(1) for i in range(z)]

    def get_theta(self):
        """Decorate the two Gamma tensors with lambdas and contract them."""
        GA, GB = self.Gs
        # Decorate Tensors with lambdas
        for b in range(0, self.z):
            GA = scale_axis(GA, self.Ss[b], axis=b+1)
        for b in range(1, self.z):  # starting from bond 1: we contract bond 'a'
            GB = scale_axis(GB, self.Ss[b], axis=b+1)
        theta = np.tensordot(GA, GB, axes=(1, 1))  # i b c j b' c'
        return theta

    def simplified_update(self, U_bonds, chi):
        """Use simplified to apply one U on each bond.

        Parameters
        ----------
        state : TPS
            The state to which the U's should be applied.
        U_bonds : list of np.Array[ndim=4]
            The evolution operators, with legs ``i, j, i' j'``
        """
        z = self.z
        for bond in range(z):
            theta = self.get_theta()  # i b c j b' c'
            theta = np.tensordot(U_bonds[bond], theta,
                                 axes=([2, 3], [0, z]))  # i j [i'] [j'], [i] b c [j] b' c'
            theta = theta / np.linalg.norm(theta)
            dimbonds = list(self.Gs[0].shape[1:])
            theta = np.transpose(theta, [0, 2, 3, 1, 4, 5]) # i b c j b' c'
            theta = np.reshape(theta, (self.d * np.prod(dimbonds[1:]),
                                       self.d * np.prod(dimbonds[1:])))
            # Singular value decomposition
            X, Y, Z = svd(theta, full_matrices=0, lapack_driver='gesvd')
            Z = Z.T  # transpose
            # truncate
            chi_new = min(np.sum(Y > 10.**(-10)), chi)
            X = X[:, :chi_new]  # (i' b c) a
            Y = Y[:chi_new] / np.linalg.norm(Y[:chi_new])  # new singular values on bond a
            Z = Z[:, :chi_new]  # (j' b' c') a
            # Get the new tensors
            GA = np.reshape(X, [self.d] + dimbonds[1:] + [chi_new])  # i' b c a
            GB = np.reshape(Z, [self.d] + dimbonds[1:] + [chi_new])  # j' b' c' a
            # rescale to obtain Gammas
            for b in range(1, z):
                GA = scale_axis(GA, self.Ss[b]**(-1), axis=b)
                GB = scale_axis(GB, self.Ss[b]**(-1), axis=b)
            # put back
            self.Gs[0] = np.transpose(GA, [0, 3, 1, 2])  # i a b c
            self.Gs[1] = np.transpose(GB, [0, 3, 1, 2])  # j a b c
            self.Ss[0] = Y[:chi_new] / np.linalg.norm(Y[0:chi_new]) # new singular values at a-bond
            # trick: rotate the tensors to apply U on other bonds
            self.rotate()
        # continue with next bond, rotate z times -> back to original, but U applied on each bond

    def rotate(self):
        for i in [0, 1]:
            self.Gs[i] = np.transpose(self.Gs[i], [0, 2, 3, 1]) # i b c a
        sa = self.Ss.pop(0) # remove first item from the list and return it
        self.Ss.append(sa) # append at the end

    def get_Z(self, bond_op=None):
        """Calculate the tensors building the norm.

        The norm <psi|psi> corresponds to the contraction of the returned Z tensor
        (for bond_op=None) in a checkerboard pattern.
        To evaluate the expectation value <psi|bond_op|psi>, a *single* of these Z tensors has
        to be replaced with a Z tensor.

        Parameters
        ----------
        bond_op : None or np.Array
            If None, return the Z tensor for <psi|psi>, otherwise sandwich the given operation
            between bra and ket. The operator should have legs  ``i j i' j'``.

        Returns
        -------
        Z : np.Array
            Has legs # (b b*) (c' c'*) (c c*) (b' b'*)
        """
        # first decorate the gammas with sqrt(s)
        Ta, Tb = self.Gs
        _, chia, chib, chic = Ta.shape
        for b in range(self.z):
            Ta = scale_axis(Ta, np.sqrt(self.Ss[b]), axis=1+b)
            Tb = scale_axis(Tb, np.sqrt(self.Ss[b]), axis=1+b)
        # now contract
        Z = np.tensordot(Ta, Tb, axes=[1, 1]) # i [a] b c, j [a'] b' c'
        Zc = Z.conj()  # i* b* c* j* b'* c'*
        if bond_op is not None:  # possibly apply a bond-operator
            Z = np.tensordot(bond_op, Z, axes=([2, 3], [0, 3])) # i j [i'] [j'], [i] b c [j] b' c'
            Z = np.transpose(Z, [0, 2, 3, 1, 4, 5])  # i b c j b' c'
        Z = np.tensordot(Z, Zc, axes=([0, 3], [0, 3])) # [i] b c [j] b' c', [i*] b* c* [j*] b'* c'*
        Z = np.transpose(Z, [0, 4, 1, 5, 2, 6, 3, 7]) # b b* c c* b' b'* c' c'*
        Z = np.reshape(Z, (chib**2, chic**2, chib**2, chic**2)) # (b b*) (c c*) (b' b'*) (c' c'*)
        Z = np.transpose(Z, (0, 3, 1, 2))  # (b b*) (c' c'*) (c c*) (b' b'*)
        return Z

    def init_boundary_mps(self, state=[0, 0]):
        """Initialize MPS for the boundary"""
        _, _, chi_b, chi_c = self.Gs[0].shape
        B1 = np.zeros((1, chi_c**2, 1)) # vL i vR
        B1[0, state[0], 0] = 1.
        B2 = np.zeros((1, chi_b**2, 1)) # vL i vR
        B2[0, state[1], 0] = 1.
        Ss = [np.ones(1)]*2
        return MPS([B1, B2], Ss), MPS([B2, B1], Ss).copy()


def scale_axis(G, S, axis):
    """Apply a 'diagonal matrix' S to a certain axis/leg of a tensor
    e.g., result[a,b,c,...] = G[a,b,c,...] S[b] for axis = 1 (no sum!)
    """
    result = np.tensordot(G, np.diag(S), axes=(axis, 0))  # a, c, d, ..., b'
    # transpose back to same order of legs as before
    tr = list(range(G.ndim))
    tr[axis:axis] = [G.ndim - 1]
    return np.transpose(result, tr[:-1])


def run_infinite_TEBD(psi, U_bonds, N_steps, chi_max, eps=1.e-10):
    """basically the same as c_tebd.run_TEBD, but adjusted to work for infinite (uniform) MPS.
    (should work regardless of whether you did exercise 10.1)"""
    Nbonds = L = 2
    for n in range(N_steps):
        for k in [0, 1]:  # even, odd
            for i_bond in range(k, Nbonds, 2):
                U_bond = U_bonds[i_bond]
                i, j = i_bond, (i_bond + 1) % L
                theta = psi.get_theta1(i)
                theta = np.tensordot(theta, psi.Bs[j], [2, 0])  # vL i [vR], [vL] j vR
                Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i j [i*] [j*], vL [i] [j] vR
                Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
                # split and truncate
                Ai, Sj, Bj = split_truncate_theta(Utheta, chi_max, eps)
                # put back into MPS
                Gi = np.tensordot(np.diag(psi.Ss[i]**(-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
                psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])  # vL i [vC], [vC] vC
                psi.Ss[j] = Sj  # vC
                psi.Bs[j] = Bj  # vC j vR
    # done


def exp_value_inf_mps(mps_top, mps_bottom, Z0, Zobs, N_boundary):
    """Evaluate an expectation value between two (different) MPS.


    Evaluates the expectation values
    <mps_bottom|... Z0 Z0 Z_ob Z0 Z0...|mps_top>
    ---------------------------------------------
    <mps_bottom|... Z0 Z0 Z0   Z0 Z0...|mps_top>

    for each Z_ob in Zobs.
    """
    C_t = np.tensordot(mps_top.Bs[0], mps_top.Bs[1], axes=(2, 0))  # vL i [vR], [vL] j vR
    C_b = np.tensordot(mps_bottom.Bs[0], mps_bottom.Bs[1], axes=(2, 0))  # vL* i* [vR*], [vL*] j* vR*
    C_b = np.conj(C_b) # (actually, in our case mps_bottom is real, but whatever...)

    # get transfer matrix with Z0 inbetween
    TM_0 = np.tensordot(C_t, Z0, axes=([1, 2], [2, 3]))  # vL [i] [j] vR, i j [i'] [j']
    TM_0 = np.tensordot(TM_0, C_b, axes=([2, 3], [1, 2]))  # vL vR [i] [j], vL* [i*] [j*] vR*
    TM_0 = np.transpose(TM_0, (0, 2, 1, 3))  # vL vL* vR vR*
    # get transfermatrix with Zob in between
    TM_obs = []
    for Zob in Zobs:
        TM_ob = np.tensordot(C_t, Zob, axes=([1, 2], [2, 3]))  # vL [i] [j] vR, i j [i'] [j']
        TM_ob = np.tensordot(TM_ob, C_b, axes=([2, 3], [1, 2]))  # vL vR [i] [j], vL* [i*] [j*] vR*
        TM_ob = np.transpose(TM_ob, (0, 2, 1, 3))  # vL vL* vR vR*
        TM_obs.append(TM_ob)

    # get dominant left (v_l) and right (v_r) eigenvector of TM_0
    v_l = np.ones((C_t.shape[0], C_b.shape[0])) # vR vR*
    for j in range(N_boundary):
        v_l = np.tensordot(v_l, TM_0, axes=([0, 1], [0, 1]))  # [vR] [vR*], [vL] [vL*] vR vR*
        v_l = v_l / np.linalg.norm(v_l)
    v_r = np.ones((C_t.shape[3], C_b.shape[3])) # [vL] [vL*]
    for j in range(N_boundary):
        v_r = np.tensordot(TM_0, v_r, axes=([2, 3], [0, 1]))  # vL vL* [vR] [vR*], [vL] [vL*]
        v_r = v_r / np.linalg.norm(v_r)
    # norm
    N = np.tensordot(v_l, TM_0, axes=([0, 1], [0, 1])) # [vR] [vR*], [vL] [vL*] vR vR*
    N = np.tensordot(N, v_r, axes=([0, 1], [0, 1]))  # [vR] [vR*], [vL] [vL*]
    exp_vals = []
    for TM_ob in TM_obs:
        E = np.tensordot(v_l, TM_ob, axes=([0, 1], [0, 1])) # [vR] [vR*], [vL] [vL*] vR vR*
        E = np.tensordot(E, v_r, axes=([0, 1], [0, 1]))  # [vR] [vR*], [vL] [vL*]
        exp_vals.append(E/N)
    return exp_vals


###########
# Example how to use the above functions

class TFIModelHoneycomb:
    def __init__(self, J, g):
        self.z = 3
        self.J, self.g = J, g
        self.sigmax = np.array([[0., 1.], [1., 0.]])
        self.sigmay = np.array([[0., -1j], [1j, 0.]])
        self.sigmaz = np.array([[1., 0.], [0., -1.]])
        self.id = np.eye(2)
        self.init_H_bonds()

    def init_H_bonds(self):
        s0, sx, sz = self.id, self.sigmax, self.sigmaz
        d = 2
        self.H_bonds = []
        for i in range(self.z):
            H = -self.J * np.kron(sx, sx) - self.g * (np.kron(sz, s0) + np.kron(s0, sz)) / self.z
            self.H_bonds.append(np.reshape(H, (d, d, d, d)))


def run_simplified_update(model, chi_tps, N_imaginary):
    """Use the simplified update with imaginary time evolution to find a ground state PEPS.

    Parameters
    ----------
    model :
        Instance of a model class with attribute H_bonds
    chi_tps : int
        The maximum bond dimension of the PEPS.
    N_imaginary : int
        How many timesteps should be applied for each `delta`.
    """
    print("imaginary time evolution")
    psi = TPS(d=2, state=[0, 0])
    for delta in [0.1, 0.01, 0.001]:
        print("--> delta =", delta)
        U_bonds = calc_U_bonds(model, delta)
        for i in range(N_imaginary):
            psi.simplified_update(U_bonds, chi_tps)
    return psi


def evaluate_exp_vals(psi_tps, observables, chi_mps, N_boundary):
    """Evaluate expectation values of a TPS using boundary MPS.

    Parameters
    ----------
    psi_tps : TPS
        The PEPS to be contracted
    observables : list of list of bond operators
        observables[i][b] should be a bond operator with indices ``i j i' j'``,
        acting on bond 'b'=0,1,2 for the a,b,c bond of the PEPS.
    chi_mps : int
        The maximum bond dimension of the resulting MPS.
    N_boundary : int
        This function uses a power method, applying the transfer matrix generated by the Z
        to the top and bottom MPS `N_boundary` times, and similar to find the dominant left
        and right eigenvectors for the contraction of the overlap of top and bottom MPS.

    Returns
    -------
    exp_vals : np.Array
        ``exp_vals[i]`` is the sum over b of the expectation value of observables[i][b], divided by 2
        to return the density per site.
    """
    exp_vals = []
    for bond in range(psi_tps.z):
        Z0 = psi_tps.get_Z()
        mps_top, mps_bot = psi_tps.init_boundary_mps()
        run_infinite_TEBD(mps_top, [Z0]*2, N_boundary, chi_mps)
        run_infinite_TEBD(mps_bot, [Z0.transpose([2, 3, 0, 1])]*2, N_boundary, chi_mps)
        Zobs = []  # for now only one observable
        for obs in observables:
            Zobs.append(psi_tps.get_Z(obs[bond]))
        E = exp_value_inf_mps(mps_top, mps_bot, Z0, Zobs, N_boundary)
        exp_vals.append(E)
        psi_tps.rotate()
    return 0.5 * np.sum(exp_vals, axis=0)  # sum over different bonds, return density per site


def example_run_ising_honeycomb(chi_tps=2, chi_mps=20, J=1., g=1., N_imaginary=300, N_boundary=10):
    """Example how to combine all the above into a simulation."""
    print("Parameters: ", locals())
    model = TFIModelHoneycomb(J, g)
    psi_tps = run_simplified_update(model, chi_tps, N_imaginary)
    print("evaluating expectation values")
    exp_vals = evaluate_exp_vals(psi_tps, [model.H_bonds], chi_mps, N_boundary)
    return exp_vals


if __name__ == "__main__":
    res = example_run_ising_honeycomb()
    print(res)
