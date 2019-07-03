
import numpy as np
import scipy
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import pickle
import gzip

# This is the code from the swensen_wang_ising notebook (exercise 3)

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

def flip_spins(spins, N_components, labels):
    flip_cluster = np.random.random(N_components) < 0.5   # N_components True/False values with 50/50 chance
    for n in range(len(spins)):
        cluster = labels[n]
        if flip_cluster[cluster]:
            spins[n] = - spins[n]
    # done

def swendsen_wang_update(spins, bonds, T):
    """Perform one update of the Swendsen-Wang algorithm"""
    N = len(spins)
    weights = get_weights(spins, bonds, T)
    graph = csr_matrix((weights, (bonds[:, 0], bonds[:, 1])), shape=(N, N))
    graph += csr_matrix((weights, (bonds[:, 1], bonds[:, 0])), shape=(N, N))
    N_components, labels = connected_components(graph, directed=False)
    flip_spins(spins, N_components, labels)

try:
    # Try to speed things up using just-in-time compilation from the numba package, if available
    from numba import jit
    get_weights = jit(get_weights, nopython=True)
    flip_spins = jit(flip_spins, nopython=True)
except ImportError:
    print("warning: can't import numba: code will run slower (~3 minutes), but still works")
    pass

####################
# now we call the code to generate the data

def generate_data(L=28, temps=[3., 2.3, 1.5], N_training=10000):
    """Run Monte Carlo for a few different temperatures.

    Keep the "images" of configuration snapshots and return them in the same data format as the
    MNIST data set.

    Parameters
    ----------
    L : int
        Size of the physical system in each direction; the images produced have L x L pixels.
    temps : list of float
        The temperatures for which to generate images.
    N_training : int
        The number of images in the training set to generate for each temperature,
        i.e., the total number of imagages in the training_data set is ``N_training*len(temps)``.

    Returns
    -------
    training_data, validation_data, test_data:
        Data in the same format as in the MNIST data set, see data_loader.load_data()
    """
    spins, bonds, _ = init_system(L, L)
    N_val = N_test = N_training // 5
    # generate the data
    imgs = [[], [], []]
    labels = [[], [], []]
    for lbl, T in enumerate(temps):
        print("generate data for T= {T:.3f}".format(T=T), flush=True)
        for _ in range(50):
            swendsen_wang_update(spins, bonds, T)
        for i, N in enumerate([N_training, N_val, N_test]):
            for _ in range(N):
                swendsen_wang_update(spins, bonds, T)
                imgs[i].append(spins.copy())
                labels[i].append(lbl)
    # shuffle the data and bring it in the same form as the MNIST data set
    for i in range(3):
        new_order = np.arange(len(imgs[i]), dtype=np.intp)
        np.random.shuffle(new_order)
        imgs[i] = np.array(imgs[i])[new_order, :]
        labels[i] = np.array(labels[i])[new_order] #[labels[i] for i in new_order]
    training_data = imgs[0], labels[0]
    validation_data = imgs[1], labels[1]
    test_data = imgs[2], labels[2]
    return training_data, validation_data, test_data

def save_data(data, filename):
    print("save data to ", filename)
    with gzip.open(filename, 'wb', compresslevel=2) as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    data = generate_data()
    save_data(data, filename="mcIsing.pkl.gz")
