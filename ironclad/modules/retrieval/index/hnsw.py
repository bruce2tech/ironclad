import faiss
import numpy as np
import pickle


class FaissHNSW:
    """
    A FAISS HNSW index for storing embeddings and their associated metadata using
    the Hierarchical Navigable Small World (HNSW) algorithm.

    This class leverages FAISS's HNSW index for similarity search in large datasets.
    HNSW indexes provide fast approximate nearest neighbor search without the need for a
    separate training step. The index directly adds embeddings, partitions them into a graph-based
    structure, and uses this structure to perform rapid similarity queries.

    Attributes:
        dim (int): The dimensionality of the embeddings.
        metadata (list): A list storing metadata associated with each embedding.
        index (faiss.IndexHNSWFlat): The underlying FAISS HNSW index object.
    """

    def __init__(self, dim, **kwargs):
        """
        Initializes the FaissHNSW instance.

        This method creates a FAISS HNSW index for flat (L2) search. It accepts keyword arguments
        to configure parameters of the HNSW algorithm, including:
          - M (int): The number of neighbors in the HNSW graph. A higher value leads to a more connected graph
                     and can improve search quality at the expense of increased memory usage. Default is 32.
          - efConstruction (int): The size of the dynamic candidate list during the index construction phase.
                                  Larger values improve index quality but increase construction time. Default is 40.

        After initialization, the index is ready to accept embeddings without a training step.

        Parameters:
            dim (int): The dimensionality of the embeddings.
            **kwargs: Optional keyword arguments to configure the HNSW index. Recognized keys include 'M' and 'efConstruction'.
        """
        self.dim = int(dim)
        self.metadata = []  # Will store associated metadata.
        self.metric = str(kwargs.get('metric', 'euclidean')).lower()
        self.m = int(kwargs.get('M', 32))
        self.efConstruction = int(kwargs.get('efConstruction', 40))
        self.efSearch = int(kwargs.get('efSearch', 50))

        self.normalize = bool(kwargs.get('normalize', self.metric == 'cosine'))

        # if self.metric in ['euclidean', 'minkowski']:
        #     self.index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_L2)
        #     self.index.hnsw.efConstruction = self.efConstruction
        # elif self.metric in ['cosine', 'dot_product']:
        #     # Both cosine and dot_product use the inner-product index.
        #     self.index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        #     self.index.hnsw.efConstruction = self.efConstruction
        # else:
        #     raise ValueError("Unsupported metric. Use 'euclidean', 'cosine', or 'dot_product'.")
        
                # Choose FAISS metric
        if self.metric in ('euclidean', 'l2', 'minkowski'):
            metric_type = faiss.METRIC_L2
        elif self.metric in ('cosine', 'dot_product', 'inner_product', 'ip'):
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'cosine', or 'dot_product'.")

        # Build HNSW index
        self.index = faiss.IndexHNSWFlat(self.dim, self.m, metric_type)
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.hnsw.efSearch = self.efSearch

        # Positional metadata list
        self.metadata = []

    def add_embeddings(self, new_embeddings, new_metadata):
        """
        Adds new embeddings and their associated metadata to the HNSW index.

        This method processes each new embedding by converting it into a NumPy array and verifying
        its dimensionality. If an embedding's dimension does not match the expected dimension, a
        ValueError is raised. The corresponding metadata for each embedding is stored in an internal
        list. The embeddings are reshaped to a two-dimensional array before being added to the index.

        Parameters:
            new_embeddings (list or np.ndarray): A list or array of new embeddings to be added.
                Each embedding should be an array-like object with a length equal to `dim`.
            new_metadata (list): A list of metadata items corresponding to each embedding.

        Raises:
            ValueError: If the number of embeddings does not match the number of metadata entries.
            ValueError: If any individual embedding does not match the specified dimensionality.
        """
        # if len(new_embeddings) != len(new_metadata):
        #     raise ValueError("The number of embeddings must match the number of metadata entries.")

        # for emb, meta in zip(new_embeddings, new_metadata):
        #     emb = np.array(emb)
        #     if emb.shape[0] != self.dim:
        #         raise ValueError(f"Embedding has dimension {emb.shape[0]}, expected {self.dim}.")
        #     self.metadata.append(meta)
        #     vector = emb.astype(np.float32).reshape(1, -1)
        #     if self.metric == 'cosine':
        #         # Normalize vector so that inner product corresponds to cosine similarity.
        #         faiss.normalize_L2(vector)
        #     # For 'euclidean' and 'dot_product', the vector is added as is.
        #     self.index.add(vector)

        X = np.asarray(new_embeddings, dtype=np.float32)
        if X.ndim == 1:
            if X.shape[0] != self.dim:
                raise ValueError(f"Embedding has dimension {X.shape[0]}, expected {self.dim}.")
            X = X.reshape(1, -1)
        elif X.ndim == 2:
            if X.shape[1] != self.dim:
                raise ValueError(f"Embeddings have dimension {X.shape[1]}, expected {self.dim}.")
        else:
            raise ValueError("Embeddings must be 1D or 2D array-like.")

        if len(new_metadata) != X.shape[0]:
            raise ValueError("The number of embeddings must match the number of metadata entries.")

        # If using cosine similarity via inner product, normalization improves results
        if self.normalize:
            faiss.normalize_L2(X)

        self.index.add(X)
        self.metadata.extend(new_metadata)  # positional mapping preserved

    def get_metadata(self, idx):
        """
        Retrieves the metadata associated with a specific embedding based on its index.

        The given index corresponds to the position in the metadata list where the metadata was stored
        when the embedding was added. This method validates the index and raises an IndexError if it is out of bounds.

        Parameters:
            idx (int): The integer index of the embedding for which metadata is being requested.

        Returns:
            The metadata object associated with the specified embedding.

        Raises:
            IndexError: If the provided index is negative or exceeds the number of stored embeddings.
        """
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError("Index out of bounds.")
        return self.metadata[idx]

    def save(self, filepath):
        """
        Serializes and saves the FaissHNSW instance to a file using pickle.

        This method saves the entire state of the instance, including the FAISS index, metadata,
        and configuration parameters (dim, M, and efConstruction). The saved file can later be loaded
        to restore the index for future use without needing to reinitialize or rebuild.

        Parameters:
            filepath (str): The file path where the serialized instance will be saved.
        """
        buf = faiss.serialize_index(self.index)
        if isinstance(buf, (bytes, bytearray)):
            index_bytes = bytes(buf)
        else:
            index_bytes = np.asarray(buf, dtype=np.uint8).tobytes()


        state = {
            "dim": self.dim,
            "M": self.m,
            "efConstruction": self.efConstruction,
            "efSearch": self.efSearch,
            "metric": self.metric,
            "normalize": self.normalize,
            "metadata": self.metadata,
            "index_bytes": index_bytes,
            "version": 1,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        # with open(filepath, 'wb') as f:
        #     pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a serialized FaissHNSW instance from a file.

        This class method reads the pickle file at the given file path and returns a FaissHNSW
        instance with its state restored. This allows users to resume operations on a previously
        saved index.

        Parameters:
            filepath (str): The file path from which the instance will be loaded.

        Returns:
            An instance of FaissHNSW with the state restored from the specified file.
        """

        with open(filepath, "rb") as f:
            loaded = pickle.load(f)

        if isinstance(loaded, FaissHNSW):
            return loaded
        state = loaded
        obj = cls(
            dim=state["dim"],
            M=state["M"],
            efConstruction=state["efConstruction"],
            efSearch=state["efSearch"],
            metric=state.get("metric", "euclidean"),
            normalize=state.get("normalize", state.get("metric", "euclidean") == "cosine"),
        )
        obj.metadata = state["metadata"]

        arr = np.frombuffer(state["index_bytes"], dtype=np.uint8)
        obj.index = faiss.deserialize_index(arr)

        # (Re)apply efSearch in case you want to override post-load later
        if hasattr(obj.index, "hnsw"):
            obj.index.hnsw.efSearch = obj.efSearch

        return obj

        # with open(filepath, 'rb') as f:
        #     instance = pickle.load(f)
        # return instance


if __name__ == "__main__":

    # Initialize a FaissHNSW index with embedding dimension 4 using custom HNSW parameters.
    index = FaissHNSW(dim=4, M=16, efConstruction=50, efSearch=64, metric="cosine", normalize=True)
    # index = FaissHNSW(dim=4, M=16, efConstruction=50)

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

    if index.normalize:
        faiss.normalize_L2(query)

    # Let's search the index with a query vector.
    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    k = 2  # number of nearest neighbors to retrieve

    # The FAISS search directly returns distances and indices.
    distances, indices = index.index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]

    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    # Save the index to disk.
    filepath = "faiss_hnsw_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")

    # Load the index from disk.
    loaded_index = FaissHNSW.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))
