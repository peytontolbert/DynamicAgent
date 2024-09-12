from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import faiss
import pickle
import os
import networkx as nx


class EmbeddingManager:
    """
    A class for managing text embeddings, including encoding, caching, similarity search, and dimensionality reduction.

    Attributes:
        model (SentenceTransformer): The sentence transformer model used for encoding text.
        cache_dir (str): Directory to store the embedding cache.
        cache (Dict[str, np.ndarray]): A dictionary to cache embeddings.
        index (Optional[faiss.IndexFlatIP]): FAISS index for efficient similarity search.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "./embedding_cache"
    ):
        """
        Initialize the EmbeddingManager.

        Args:
            model_name (str): Name of the sentence transformer model to use.
            cache_dir (str): Directory to store the embedding cache.
        """
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.cache: Dict[str, np.ndarray] = {}
        self.load_cache()
        self.index: Optional[faiss.IndexFlatIP] = None

    def load_cache(self):
        """Load the embedding cache from disk."""
        if os.path.exists(f"{self.cache_dir}/embedding_cache.pkl"):
            with open(f"{self.cache_dir}/embedding_cache.pkl", "rb") as f:
                self.cache = pickle.load(f)

    def save_cache(self):
        """Save the embedding cache to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(f"{self.cache_dir}/embedding_cache.pkl", "wb") as f:
            pickle.dump(self.cache, f)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into an embedding.

        Args:
            text (str): The text to encode.

        Returns:
            np.ndarray: The embedding of the input text.
        """
        if text in self.cache:
            return self.cache[text]
        embedding = self.model.encode(text)
        self.cache[text] = embedding
        self.save_cache()
        return embedding

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings into embeddings.

        Args:
            texts (List[str]): The list of texts to encode.

        Returns:
            np.ndarray: An array of embeddings for the input texts.
        """
        new_texts = [text for text in texts if text not in self.cache]
        if new_texts:
            new_embeddings = self.model.encode(new_texts)
            for text, embedding in zip(new_texts, new_embeddings):
                self.cache[text] = embedding
            self.save_cache()
        return np.array([self.cache[text] for text in texts])

    def cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Calculate the cosine similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): First embedding.
            embedding2 (np.ndarray): Second embedding.

        Returns:
            float: The cosine similarity between the two embeddings.
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def euclidean_distance(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Calculate the Euclidean distance between two embeddings.

        Args:
            embedding1 (np.ndarray): First embedding.
            embedding2 (np.ndarray): Second embedding.

        Returns:
            float: The Euclidean distance between the two embeddings.
        """
        return np.linalg.norm(embedding1 - embedding2)

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        k: int = 5,
        metric: str = "cosine",
    ) -> List[Tuple[int, float]]:
        """
        Find the k most similar embeddings to a query embedding.

        Args:
            query_embedding (np.ndarray): The query embedding.
            embeddings (List[np.ndarray]): List of embeddings to search.
            k (int): Number of similar embeddings to return.
            metric (str): Similarity metric to use ('cosine' or 'euclidean').

        Returns:
            List[Tuple[int, float]]: List of tuples containing the index and similarity score of the top k similar embeddings.
        """
        if metric == "cosine":
            similarities = [
                self.cosine_similarity(query_embedding, emb) for emb in embeddings
            ]
            top_k = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[
                :k
            ]
        elif metric == "euclidean":
            distances = [
                self.euclidean_distance(query_embedding, emb) for emb in embeddings
            ]
            top_k = sorted(enumerate(distances), key=lambda x: x[1])[:k]
        else:
            raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")
        return top_k

    def build_faiss_index(self, embeddings: List[np.ndarray]):
        """
        Build a FAISS index for efficient similarity search.

        Args:
            embeddings (List[np.ndarray]): List of embeddings to index.
        """
        self.index = faiss.IndexFlatIP(len(embeddings[0]))
        self.index.add(np.array(embeddings))

    def faiss_search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a similarity search using the FAISS index.

        Args:
            query_embedding (np.ndarray): The query embedding.
            k (int): Number of similar embeddings to return.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing distances and indices of the k most similar embeddings.
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index first.")
        return self.index.search(query_embedding.reshape(1, -1), k)

    def reduce_dimensions(
        self, embeddings: List[np.ndarray], method: str = "pca", n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce the dimensionality of embeddings.

        Args:
            embeddings (List[np.ndarray]): List of embeddings to reduce.
            method (str): Dimensionality reduction method ('pca' or 'tsne').
            n_components (int): Number of dimensions to reduce to.

        Returns:
            np.ndarray: Array of reduced embeddings.
        """
        if method == "pca":
            pca = PCA(n_components=n_components)
            return pca.fit_transform(embeddings)
        elif method == "tsne":
            tsne = TSNE(n_components=n_components)
            return tsne.fit_transform(embeddings)
        else:
            raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    def update_embedding(self, text: str, new_text: str):
        """
        Update the embedding for a given text.

        Args:
            text (str): The original text whose embedding needs to be updated.
            new_text (str): The new text to replace the original.
        """
        if text in self.cache:
            del self.cache[text]
        self.encode(new_text)
        self.save_cache()

    def clear_cache(self):
        """Clear the embedding cache and save the empty cache to disk."""
        self.cache.clear()
        self.save_cache()

    def encode_graph_node(self, node: Dict[str, Any]) -> np.ndarray:
        """
        Encode a knowledge graph node into an embedding.

        Args:
            node (Dict[str, Any]): The node to encode.

        Returns:
            np.ndarray: The embedding of the input node.
        """
        # Combine relevant node attributes into a single string
        node_text = f"{node.get('label', '')} {node.get('content', '')}"
        return self.encode(node_text)

    def encode_graph_edge(self, edge: Dict[str, Any]) -> np.ndarray:
        """
        Encode a knowledge graph edge into an embedding.

        Args:
            edge (Dict[str, Any]): The edge to encode.

        Returns:
            np.ndarray: The embedding of the input edge.
        """
        # Combine relevant edge attributes into a single string
        edge_text = f"{edge.get('type', '')} {edge.get('properties', '')}"
        return self.encode(edge_text)

    def get_graph_embeddings(self, graph: nx.Graph) -> Dict[str, np.ndarray]:
        """
        Get embeddings for all nodes and edges in a knowledge graph.

        Args:
            graph (nx.Graph): The knowledge graph.

        Returns:
            Dict[str, np.ndarray]: A dictionary of node/edge IDs to their embeddings.
        """
        embeddings = {}
        for node, data in graph.nodes(data=True):
            embeddings[f"node_{node}"] = self.encode_graph_node(data)
        for u, v, data in graph.edges(data=True):
            embeddings[f"edge_{u}_{v}"] = self.encode_graph_edge(data)
        return embeddings

    def update_graph_embeddings(self, graph: nx.Graph, changed_elements: List[str]):
        """
        Update embeddings for changed elements in the knowledge graph.

        Args:
            graph (nx.Graph): The knowledge graph.
            changed_elements (List[str]): List of changed node/edge IDs.
        """
        for element in changed_elements:
            if element.startswith("node_"):
                node = element[5:]
                if node in graph.nodes:
                    self.cache[element] = self.encode_graph_node(graph.nodes[node])
            elif element.startswith("edge_"):
                u, v = element[5:].split("_")
                if graph.has_edge(u, v):
                    self.cache[element] = self.encode_graph_edge(graph.edges[u, v])
        self.save_cache()
