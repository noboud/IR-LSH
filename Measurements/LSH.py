import numpy as np

# The hash function defined as described in "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions"
class L2Hash:
    def __init__(self, dim, r, nbits, seed=1):
        self.seed = seed
        self.nbits = nbits
        
        gen = np.random.RandomState(seed)
        self.a = gen.normal(0, 1, (nbits, dim))
        self.b = gen.uniform(0.0, r)
        self.r = r

    def hash(self, vectors):
        hash_values = (np.dot(vectors, self.a.T) + self.b) / self.r
        hash_binary = (hash_values >= 0).astype(int)
        return np.apply_along_axis(lambda row: ''.join(row.astype(str)), axis=1, arr=hash_binary)

# Our LSH index, which can be used for vector similarity search
class LSHIndex:
    def __init__(self, dim, r=5, nbits=5, seed=1):
        # Store the vector dimension
        self._dim = dim

        # Created hash codes by applying our projections
        self._hasher = L2Hash(self._dim, r, nbits, seed)
        self._r = r
        self._nbits = nbits
        self._seed = seed
        self._binned_vectors = dict()

    # Sort the hash codes into bins
    def __hashes_to_bins(self, vectors, hash_codes):
        bins_dict = dict()
        unique_bins = np.unique(hash_codes)
        
        for cur_bin in unique_bins:
            bins_dict[cur_bin] = vectors[hash_codes == cur_bin]
        
        return bins_dict

    # Find the closest bins to a hash code by Hamming distance
    def __closest_bins(self, hash_code):
        bins = np.array(list(self._binned_vectors.keys()))
        
        # Calculate Hamming distance
        distances = np.array([sum(c1 != c2 for c1, c2 in zip(hash_code, bin)) for bin in bins])
        sorted_indices = np.argsort(distances)
        return np.array(bins)[sorted_indices]

    # Find the K nearest neighbours for a given bin
    def __find_k_neighbours(self, target, K):
        closest_bins = self.__closest_bins(target)
        neighbours = self._binned_vectors[closest_bins[0]]

        index = 1
        while (neighbours.shape[0] < K and index < len(closest_bins)):
            neighbours = np.concatenate((neighbours, self._binned_vectors[closest_bins[index]]), axis=0)
            index += 1
        
        return neighbours

    # Add function which adds vectors with their given indices to the index
    def add(self, indices, vectors):
        if (self._dim != vectors.shape[1]):
            raise Exception(f"Dimension mismatch: Index ({self._dim}) and vectors ({vectors.shape[1]})")
        
        # Add indexes into vectors (such that we can find the original index after binning)
        indexed_vectors = np.hstack((indices[:, np.newaxis], vectors))
        hash_codes = self._hasher.hash(vectors)
        self._binned_vectors = self.__hashes_to_bins(indexed_vectors, hash_codes)

    # Search function which accepts a vector and a K value (number of results requested)
    def search(self, vector, K=10):
        # Hash the query to its code
        hash_code = self._hasher.hash([vector])[0]
        
        # Find the nearest bins to satisfy K results
        candidate_vectors = self.__find_k_neighbours(hash_code, K)
    
        # Calculate squared Euclidean distance
        distances = np.sum((candidate_vectors[:,1:] - vector) ** 2, axis=1)
        sorted_indices = np.argsort(distances)[:K]
        
        # Return the indices of the ranked results
        result_indices = candidate_vectors[:,0][sorted_indices]
        return result_indices, distances[sorted_indices]

    # Save function for storing the index as a npz file
    def save(self, path):
        np.savez_compressed(
            path,
            properties = {
                "dim": self._dim,
                "r": self._r,
                "seed": self._seed,
                "nbits": self._nbits
            },
            binned_vectors = self._binned_vectors
        )

    # Get all the binned vectors for debugging
    def get_binned_vectors(self):
        return self._binned_vectors

    # Loading the object from a npz file (to avoid having to rebuild it each time, and to be able to see index size after quantizing)
    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        instance = cls.__new__(cls)
        properties = data["properties"].item()

        instance._dim = properties["dim"]
        instance._r = properties["r"]
        instance._nbits = properties["nbits"]
        instance._seed = properties["seed"]
        instance._hasher = L2Hash(instance._dim, instance._r, instance._seed, instance._nbits)

        instance._binned_vectors = data["binned_vectors"].item()
        return instance

    # toString method, showcases the distribution of vectors over the bins
    def __str__(self):
        unique_bins = self._binned_vectors.keys()
        return f"LSHIndex ({', '.join([f'bin({unique_bin}) = {len(self._binned_vectors[unique_bin])}' for unique_bin in unique_bins])})"