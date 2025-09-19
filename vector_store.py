"""
Vector Store Module for RAG Application
Implements local FAISS vector database for similarity search
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, dimension: int = 1536, index_type: str = "flat"):
        """
        Initialize FAISS vector store

        Args:
            dimension: Dimension of embedding vectors (1536 for OpenAI text-embedding-3-small)
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.id_to_index = {}  # Map chunk IDs to FAISS indices
        self.index_to_id = {}  # Map FAISS indices to chunk IDs
        self.next_index = 0

        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index based on type"""
        if self.index_type == "flat":
            # L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "cosine":
            # Cosine similarity (inner product with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            # IVF index for faster search on large datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        logger.info(f"Initialized FAISS index type: {self.index_type}")

    def add_vectors(self, embeddings: List[List[float]], chunk_data: List[Dict[str, Any]]):
        """Add vectors and associated metadata to the index"""
        if len(embeddings) != len(chunk_data):
            raise ValueError("Number of embeddings must match number of chunk data entries")

        vectors = np.array(embeddings, dtype=np.float32)

        if self.index_type == "cosine":
            faiss.normalize_L2(vectors)

        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(vectors)

        start_idx = self.next_index
        self.index.add(vectors)

        for i, chunk in enumerate(chunk_data):
            idx = start_idx + i
            chunk_id = chunk.get('chunk_id', f'chunk_{idx}')

            self.id_to_index[chunk_id] = idx
            self.index_to_id[idx] = chunk_id
            self.metadata.append({
                'chunk_id': chunk_id,
                'document_id': chunk.get('document_id'),
                'text': chunk.get('text', ''),
                'token_count': chunk.get('token_count', 0),
                'added_at': datetime.now().isoformat(),
                **chunk
            })

        self.next_index += len(embeddings)
        logger.info(f"Added {len(embeddings)} vectors to index. Total vectors: {self.index.ntotal}")

    def search(self, query_embedding: List[float], k: int = 5, filter_docs: List[str] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []

        query_vector = np.array([query_embedding], dtype=np.float32)

        if self.index_type == "cosine":
            faiss.normalize_L2(query_vector)

        if k > self.index.ntotal:
            k = self.index.ntotal

        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue

            metadata = self.metadata[idx].copy()

            if filter_docs and metadata.get('document_id') not in filter_docs:
                continue

            if self.index_type == "cosine":
                similarity = float(distance)
            else:
                similarity = 1.0 / (1.0 + float(distance))

            metadata['similarity_score'] = similarity
            metadata['distance'] = float(distance)
            metadata['rank'] = i + 1

            results.append(metadata)

        logger.info(f"Found {len(results)} similar chunks")
        return results

    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        faiss.write_index(self.index, f"{filepath}.faiss")

        metadata_dict = {
            'metadata': self.metadata,
            'id_to_index': self.id_to_index,
            'index_to_id': {str(k): v for k, v in self.index_to_id.items()},
            'next_index': self.next_index,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'saved_at': datetime.now().isoformat()
        }

        with open(f"{filepath}.metadata", 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        logger.info(f"Saved index to {filepath}")

    def load_index(self, filepath: str):
        """Load FAISS index and metadata from disk"""
        self.index = faiss.read_index(f"{filepath}.faiss")

        with open(f"{filepath}.metadata", 'r') as f:
            metadata_dict = json.load(f)

        self.metadata = metadata_dict['metadata']
        self.id_to_index = metadata_dict['id_to_index']
        self.index_to_id = {int(k): v for k, v in metadata_dict['index_to_id'].items()}
        self.next_index = metadata_dict['next_index']
        self.dimension = metadata_dict['dimension']
        self.index_type = metadata_dict['index_type']

        logger.info(f"Loaded index from {filepath}")
        logger.info(f"Index contains {self.index.ntotal} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        doc_counts = {}
        for metadata in self.metadata:
            doc_id = metadata.get('document_id', 'unknown')
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'documents': len(doc_counts),
            'chunks_per_document': doc_counts,
            'memory_usage_mb': self.index.ntotal * self.dimension * 4 / (1024 * 1024) if self.index else 0
        }
