
import numpy as np
import scipy
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import time
from numba import jit

import pickle  # for input/output


def init_system(Lx, Ly):
    """Determine the bond array and an initial state of spins"""
    N = Lx * Ly

    def xy_to_n(x, y):
        return x*Ly + y

    def n_to_xy(n):
        return n // Ly, np.mod(n, Ly)

    # easy way:
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            n = xy_to_n(x, y)
            m1 = xy_to_n((x+1)% Lx, y)
            m2 = xy_to_n(x, (y+1) % Ly)
            bonds.append([n, m1])
            bonds.append([n, m2])
    bonds = np.array(bonds)
    spins = np.random.randint(0, 2, size=(N,))*2 - 1
    return spins, bonds, N


# In[3]:


# part c)
@jit(nopython=True)
def get_weights(spins, bonds, T):
    weights = np.zeros(len(bonds))
    p = np.exp(-2./T)  # set J = 1
    for b in range(len(bonds)):
        n = bonds[b, 0]
        m = bonds[b, 1]
        #if spins[n] != spins[m]:
        #    weights[b] = 0.
        #else:
        #    if np.random.rand() < p:
        #        weights[b] = 0.
        #    else:
        #        weights[b] = 1.
        if spins[n] == spins[m] and np.random.rand() > p:
            weights[b] = 1.
    return weights


# In[4]:


# part d)
@jit(nopython=True)
def flip_spins(spins, N_components, labels):
    flip_cluster = np.random.random(N_components) < 0.5   # N_components True/False values with 50/50 chance
    for n in range(len(spins)):
        cluster = labels[n]
        if flip_cluster[cluster]:
            spins[n] = - spins[n]
    # done


# In[5]:


def swendsen_wang_update(spins, bonds, T):
    """Perform one update of the Swendsen-Wang algorithm"""
    N = len(spins)
    weights = get_weights(spins, bonds, T)
    graph = csr_matrix((weights, (bonds[:, 0], bonds[:, 1])), shape=(N, N))
    graph += csr_matrix((weights, (bonds[:, 1], bonds[:, 0])), shape=(N, N))
    N_components, labels = connected_components(graph, directed=False)
    flip_spins(spins, N_components, labels)


# In[6]:


@jit(nopython=True)
def energy(spins, bonds):
    Nbonds = len(bonds)
    energy = 0.
    for b in range(Nbonds):
        energy -= spins[bonds[b, 0]]* spins[bonds[b, 1]]
    return energy

def energy2(spins, bonds):
    """alternative implementation, gives the same results, but does not require jit to be fast"""
    return -1. * np.sum(spins[bonds[:, 0]]* spins[bonds[:, 1]])

def magnetization(spins):
    return np.sum(spins)



# In[7]:

#########
# starting from here, I changed stuff compared to the notebook sol2_swendsen_wang.ipynb


def simulation(spins, bonds, T, N_measure=100):
    """Perform a Monte-carlo simulation at given temperature"""
    # no thermalization here
    Es = []
    Ms = []
    for n in range(N_measure):
        swendsen_wang_update(spins, bonds, T)
        Es.append(energy(spins, bonds))
        Ms.append(magnetization(spins))
    return np.array(Es), np.array(Ms)


# The full simulation at different temperatures
def gen_data_L(Ts, L, N_measure=10000, N_bins=10):
    print("generate data for L = {L: 3d}".format(L=L), flush=True)
    assert(N_measure//N_bins >= 10)
    spins, bonds, N = init_system(L, L)
    spins = np.random.randint(0, 2, size=(N,))*2 - 1
    obs = ['E', 'C', 'M', 'absM', 'chi', 'UB']
    data = dict((key, []) for key in obs)
    t0 = time.time()
    for T in Ts:
        if N_measure > 1000:
            print("simulating L={L: 3d}, T={T:.3f}".format(L=L, T=T), flush=True)
        # thermalize. Rule of thumb: spent ~10-20% of the simulation time without measurement
        simulation(spins, bonds, T, N_measure//10)
        # Simlulate with measurements
        bins = dict((key, []) for key in obs)
        for b in range(N_bins):
            E, M = simulation(spins, bonds, T, N_measure//N_bins)
            bins['E'].append(np.mean(E)/N)
            bins['C'].append(np.var(E)/(T**2*N))
            bins['M'].append(np.mean(M)/N)
            bins['absM'].append(np.mean(np.abs(M))/N)
            bins['chi'].append(np.var(np.abs(M))/(T*N))
            bins['UB'].append(1.5*(1.-np.mean(M**4)/(3.*np.mean(M**2)**2)))
        for key in obs:
            bin = bins[key]
            data[key].append((np.mean(bin), np.std(bin)/np.sqrt(N_bins)))
    print("generating data for L ={L: 3d} took {t: 6.1f}s".format(L=L, t=time.time()-t0))
    # convert into arrays
    for key in obs:
        data[key] = np.array(data[key])
    # good practice: save meta-data along with the original data
    data['L'] = L
    data['observables'] = obs
    data['Ts'] = Ts
    data['N_measure'] = N_measure
    data['N_bins'] = N_bins
    return data


def save_data(filename, data):
    """Save an (almost) arbitrary python object to disc."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    # done


def load_data(filename):
    """Load and return data saved to disc with the function `save_data`."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    Tc_guess = None
    #  Tc_guess = 2.27   # good guess for the 2D Ising model; uncomment this to get
    #                    # many T-points around this value for large L (-> long runtime!)
    if Tc_guess is None:
        N_measure = 1000  # just a quick guess
        Ls = [8, 16, 32]
        output_filename = 'data_ising_square.pkl'
    else:
        N_measure = 10000
        Ls = [8, 16, 32, 64, 128, 256]
        output_filename = 'data_ising_square_largeL.pkl'
    data = dict()
    for L in Ls:
        if Tc_guess is None:
            # no guess for Tc available -> scan a wide range to get a first guess
            Ts = np.linspace(1., 4., 50)
        else:
            # choose T-values L-dependent: more points around Tc
            Ts = np.linspace(Tc_guess - 0.5, Tc_guess + 0.5, 21)
            Ts = np.append(Ts, np.linspace(Tc_guess - 8./L, Tc_guess + 8./L, 50))
            Ts = np.sort(Ts)[::-1]
        data[L] = gen_data_L(Ts, L, N_measure)
    data['Ls'] = Ls
    save_data(output_filename, data)
    # data structure:
    #  data = {'Ls': [8, 16, ...],
    #          8: {'observables': ['E', 'M', 'C', ...],
    #              'Ts': (np.array of temperature values),
    #              'E': (np.array of mean & error, shape (len(Ts), 2)),
    #              'C': (np.array of mean & error, shape (len(Ts), 2)),
    #              ... (further observables & metadata)
    #             }
    #          ... (further L values with same structure)
    #         }
