import numpy as np
from numba import jit

@jit(nopython=True, cache=True, nogil=True)
def get_tree_distances(n, decision_paths, labeled_inds, counts, max_tree_distance=None):
    tree_distances = np.zeros((n, len(labeled_inds)))

    if max_tree_distance is None:
        max_tree_distance = n

    for i in range(n):
        if counts[i] < max_tree_distance:
            min1 = counts[i]
        else:
            min1 = max_tree_distance

        temp_path1=decision_paths[i, :min1]

        for j in range(len(labeled_inds)):
            if counts[labeled_inds[j]] < max_tree_distance:
                min2 = counts[labeled_inds[j]]
            else:
                min2 = max_tree_distance

            temp_path2=decision_paths[labeled_inds[j], :min2]

            for k in range(min2):    
                if np.sum(temp_path2[k] == temp_path1) > 0:
                    path1_ind = np.argwhere(temp_path1 == temp_path2[k])[0]

                    if path1_ind > max_tree_distance:
                        path1_ind = max_tree_distance
                    else:
                        path1_ind = path1_ind[0]


def stratified_sample(y, p=0.67, replace=False):
    unique_y, counts = np.unique(y, return_counts=True)
    n_per_class = np.array([int(np.math.floor(p*c)) for c in counts])
    n_per_class = np.array([max([npc, 1]) for npc in n_per_class])
    
    inds = [np.random.choice(np.where(y == unique_y[i])[0], size=npc, replace=replace) for i, npc in enumerate(n_per_class)]
    
    return np.concatenate(inds)


def gem(x, p=1):
    """ generalized mean pooling -- interpolation between mean and max """
    nobs, ndim = x.shape
    
    y = np.zeros(ndim)
    for r in range(nobs):
        for c in range(ndim):
            y[c] += x[r,c] ** p
    
    y /= nobs
    
    for c in range(ndim):
        y[c] = y[c] ** (1 / p)
    
    return y
                    tree_distances[i,j] = (k + path1_ind + 2) / 2
                    break
                tree_distances[i,j] = (min1 + min2) / 2
                
    return tree_distances
    
    
@jit(nopython=True, cache=True, nogil=True)
def get_decision_paths(n, children_array):
    ind_to_path_array = np.zeros((n, n))
    ind_to_path_array[:, 0] = np.arange(n, dtype=np.int32)
    counts = np.ones(n, dtype=np.int32)

    for i in range(children_array.shape[0]):
        node_id = i + n
        left = children_array[i,0]
        right = children_array[i,1]
                        
        for k in range(n):
            if left == ind_to_path_array[k, counts[k]-1] or right == ind_to_path_array[k, counts[k]-1]:
                ind_to_path_array[k, counts[k]] = node_id
                counts[k] += 1
                                
    return ind_to_path_array, counts
