import numpy as np
import pickle
import faiss

class FaissBruteForce:
    """
    A brute-force FAISS index for storing embeddings and their associated metadata,
    supporting Euclidean, Cosine, and Dot Product distance measures.
    
    Attributes:
        dim (int): The dimensionality of the embeddings.
        metadata (list): A list to store metadata corresponding to each embedding.
        metric (str): The distance metric to use: 'euclidean', 'cosine', or 'dot_product'.
        index (faiss.IndexFlat): A FAISS flat index initialized based on the specified metric.
    """

    def __init__(self, dim, metric='euclidean'):
        """
        Initializes the FaissBruteForce index.
        
        Parameters:
            dim (int): The dimensionality of the embeddings.
            metric (str): Distance metric to use. Options are 'euclidean', 'cosine', or 'dot_product'.
        """
        self.dim = dim
        self.metadata = []  # Will store associated metadata.
        self.metric = metric.lower()

        if self.metric in ['euclidean', 'minkowski']:
            self.index = faiss.IndexFlatL2(dim)
        elif self.metric in ['cosine', 'dot_product']:
            # Both cosine and dot_product use the inner-product index.
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'cosine', or 'dot_product'.")

    def add_embeddings(self, embeddings, metadata):
        """
        Adds new embeddings and their associated metadata to the index.
        
        Parameters:
            embeddings (list or np.ndarray): A list of embeddings, where each embedding is an array-like
                of length `dim`.
            metadata (list): A list of metadata corresponding to each embedding.
        
        Raises:
            ValueError: If an embedding does not match the specified dimensionality.
            ValueError: If the number of embeddings and metadata entries do not match.
        """
        if len(embeddings) != len(metadata):
            raise ValueError("The number of embeddings must match the number of metadata entries.")

        for emb, meta in zip(embeddings, metadata):
            emb = np.array(emb)
            if emb.shape[0] != self.dim:
                raise ValueError(f"Embedding has dimension {emb.shape[0]}, expected {self.dim}.")
            self.metadata.append(meta)
            vector = emb.astype(np.float32).reshape(1, -1)
            if self.metric == 'cosine':
                # Normalize vector so that inner product corresponds to cosine similarity.
                faiss.normalize_L2(vector)
            # For 'euclidean' and 'dot_product', the vector is added as is.
            self.index.add(vector)


    def get_metadata(self, idx):
        """
        Retrieves the metadata associated with a particular embedding index.
        
        Parameters:
            idx (int): The index of the embedding.
        
        Returns:
            The metadata associated with the embedding.
        
        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError("Index out of bounds.")
        return self.metadata[idx]

    def save(self, filepath):
        """
        Saves the current FaissBruteForce instance to a file.
        
        Parameters:
            filepath (str): The path to the file where the instance should be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a FaissBruteForce instance from a file.
        
        Parameters:
            filepath (str): The path to the file from which to load the instance.
        
        Returns:
            An instance of FaissBruteForce loaded from the file.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance


if __name__ == "__main__":
    # Example usage of FaissBruteForce class.

    # Choose the metric: 'euclidean', 'cosine', or 'dot_product'
    metric = 'cosine'
    index = FaissBruteForce(dim=4, metric=metric)

    # Create some dummy embeddings and corresponding metadata.
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ]
    identity_metadata = [
        "Alice",
        "Bob",
        "Charlie"
    ]

    # Add the embeddings and metadata to the index.
    index.add_embeddings(embeddings, identity_metadata)

    # Define a query vector.
    query = [0.1, 0.2, 0.3, 0.4]
    k = 2  # number of nearest neighbors to retrieve

    # Perform the search using our class method.
    distances, indices = index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]
    
    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    # Save the index to disk.
    filepath = "faiss_bruteforce_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")

    # Load the index from disk.
    loaded_index = FaissBruteForce.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))
