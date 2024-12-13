import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# ===== ScalarQuantizer =====
class ScalarQuantizer:
    def __init__(self, dataset=None):
        self._dataset = None
        if dataset is not None:
            self._mins = np.min(dataset, axis=1)
            self._maxs = np.max(dataset, axis=1)
            self._steps = (self._maxs - self._mins) / 255
            dataset_step = np.array(
                [(dataset[i] - self._mins[i]) / self._steps[i] for i in range(np.shape(dataset)[0])])
            self._dataset = np.uint8(dataset_step)

    def quantize(self, vector):
        return np.uint8((vector - self._mins) / self._steps)

    @property
    def dataset(self):
        if self._dataset is not None:
            return self._dataset
        raise ValueError("Call ScalarQuantizer first")

# ===== BinaryQuantizer =====
def flipByAverage(array):
    avg = np.average(array)
    return np.array([7 if array[i] >= avg else 1 for i in range(len(array))])

class BinaryQuantizer:
    def __init__(self, dataset=None):
        self._dataset = None
        if dataset is not None:
            dataset = np.apply_along_axis(flipByAverage, 0, dataset)
            self._dataset = np.uint8(dataset)

    def quantize(self, vector):
        return flipByAverage(vector)

    @property
    def dataset(self):
        if self._dataset is not None:
            return self._dataset
        raise ValueError("Call BinaryQuantizer first")

from tqdm import tqdm

# ===== ProductQuantizer =====
class ProductQuantizer:    
    def __init__(self, dataset=None, chunks=16, clusters=64):
        K = clusters
        K_ = int(K / chunks)

        assert K % chunks == 0

        split_vectors = np.array(np.array_split(dataset, chunks, axis=1)).transpose(1, 0, 2)
        
        # Use KMeans to initialize centroids (We had to resort to using KMeans from sklearn)
        c = []
        for i in range(chunks):
            kmeans = KMeans(n_clusters=K_, random_state=42)
            kmeans.fit(split_vectors[:,i])
            c.append(kmeans.cluster_centers_)
        c = np.array(c)
        print("Determined centroids")

        res_vectors = []
        for vector in tqdm(split_vectors):
            # Map each of the chunks to centroids
            centroids = np.array([self.nearest(c[j], vector[j]) for j in range(chunks)])
            res_vectors.append(centroids)
        self._dataset = np.array(res_vectors).astype(np.uint8)
        
    # Function that maps a chunk to the nearest centroid
    def nearest(self, a, b):
        distances = np.sum((a - b) ** 2, axis=-1)
        nearest_idx = np.argmin(distances, axis=0)
        return nearest_idx

    @property
    def dataset(self):
        if self._dataset is not None:
            return self._dataset
        raise ValueError("Call ProductQuantizer first")