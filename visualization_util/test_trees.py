import numpy as np

def build_tree(root, samples, max_dist=1.0):
    edges = []
    dists = np.linalg.norm(samples - root, axis=1)
    order = np.argsort(dists)
    connected = np.array([root])
    for i in order:
        pt = samples[i]
        cdist = np.linalg.norm(connected - pt, axis=1)
        best_idx = np.argmin(cdist)
        edges.append(np.array([connected[best_idx], pt]))
        connected = np.vstack([connected, pt])
    return np.array(edges)
