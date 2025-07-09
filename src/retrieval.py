import json
import logging
from typing import List, Tuple, Dict, Union, Optional, Any
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker

_log = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        # Caching loaded BM25 indexes and documents can improve performance for repeated queries to the same doc
        self._bm25_indexes_cache = {}
        self._documents_cache = {}

    def _load_document_and_index(self, document_id: str) -> Tuple[Optional[Dict], Optional[BM25Okapi]]:
        """Loads a document and its BM25 index, using caching."""
        if document_id in self._documents_cache and document_id in self._bm25_indexes_cache:
            return self._documents_cache[document_id], self._bm25_indexes_cache[document_id]

        doc_path = self.documents_dir / f"{document_id}.json"
        bm25_path = self.bm25_db_dir / f"{document_id}.pkl"

        if not doc_path.exists():
            _log.error(f"Document file {doc_path} not found for ID {document_id}.")
            return None, None
        if not bm25_path.exists():
            _log.error(f"BM25 index file {bm25_path} not found for ID {document_id}.")
            return None, None
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            with open(bm25_path, 'rb') as f:
                bm25_index = pickle.load(f)
            
            self._documents_cache[document_id] = document
            self._bm25_indexes_cache[document_id] = bm25_index
            return document, bm25_index
        except Exception as e:
            _log.error(f"Error loading document or BM25 index for {document_id}: {e}", exc_info=True)
            return None, None

    def retrieve_by_document_id(self, document_id: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document, bm25_index = self._load_document_and_index(document_id)

        if document is None or bm25_index is None:
            # Error already logged in _load_document_and_index
            return []
            
        # The chunked JSON from TextSplitter has 'content' as a list of chunk dicts.
        # Each chunk dict has 'text', 'original_page', 'original_speaker', 'original_timestamp', 'podcast_id'.
        chunks = document.get("content", []) # This is the list of chunk dicts
        if not chunks:
            _log.warning(f"No content chunks found in document {document_id}.")
            return []
        
        tokenized_query = query.split()
        try:
            scores = bm25_index.get_scores(tokenized_query)
        except Exception as e: # BM25Okapi might raise error if query tokens not in vocab from empty index
            _log.error(f"Error getting BM25 scores for document {document_id}, query '{query}': {e}. This might happen if the index is empty or query tokens are all OOV.")
            return []

        
        actual_top_n = min(top_n, len(scores))
        # Ensure indices are within the bounds of the actual chunks list
        top_indices = sorted([i for i in range(len(scores)) if i < len(chunks)], key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages_content = {} # To store concatenated text for pages if return_parent_pages is True

        for index in top_indices:
            if index >= len(chunks): # Should not happen with the check above, but as a safeguard
                _log.warning(f"Index {index} out of bounds for chunks list (len {len(chunks)}) in doc {document_id}.")
                continue

            score = round(float(scores[index]), 4)
            chunk_data = chunks[index] # This is a dict with 'text', 'original_page', etc.
            
            page_num = chunk_data.get('original_page', chunk_data.get('page', 0)) # Prefer 'original_page'

            if return_parent_pages:
                if page_num not in seen_pages_content:
                    # Collect all text from this original page
                    page_full_text = " ".join(
                        c.get('text', '') for c in chunks if c.get('original_page', c.get('page')) == page_num
                    ).strip()
                    seen_pages_content[page_num] = page_full_text

                # Only add this page once to results, using the highest score of its chunks
                # This requires a bit more logic to ensure we add the page with its best score.
                # For simplicity now, let's add if not already added, score might not be max for the page.
                # A better way would be to collect all chunks, group by page, then pick top pages.
                # Current simple way:
                if not any(res['page'] == page_num for res in retrieval_results):
                    result = {
                        "score": score, # BM25 score (higher is better)
                        "page": page_num,
                        "text": seen_pages_content[page_num],
                        "podcast_id": document_id, # Add podcast_id for context
                        # Potentially add first speaker/timestamp of the page if meaningful
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "score": score,
                    "page": page_num,
                    "text": chunk_data.get('text',''),
                    "speaker": chunk_data.get('original_speaker', chunk_data.get('speaker')),
                    "timestamp": chunk_data.get('original_timestamp', chunk_data.get('timestamp')),
                    "podcast_id": document_id,
                }
                retrieval_results.append(result)
        
        # If return_parent_pages, sort by score again as we might have added pages out of order
        if return_parent_pages:
            retrieval_results.sort(key=lambda x: x['score'], reverse=True)

        return retrieval_results[:top_n] # Ensure we don't exceed top_n if pages expanded results


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs_cache = {} # Cache for loaded FAISS indexes and associated documents
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            _log.error("OPENAI_API_KEY not found.")
            raise ValueError("OPENAI_API_KEY is required.")
        return OpenAI(api_key=api_key, timeout=None, max_retries=3)
    
    @staticmethod
    def get_llm_for_similarity(): # Renamed for clarity, used by static method
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for similarity calculation.")
        return OpenAI(api_key=api_key, timeout=None, max_retries=2)

    def _load_single_db_and_doc(self, document_id: str) -> Tuple[Optional[faiss.Index], Optional[Dict]]:
        """Loads a single FAISS DB and its corresponding document JSON, using caching."""
        if document_id in self.all_dbs_cache:
            return self.all_dbs_cache[document_id]['vector_db'], self.all_dbs_cache[document_id]['document']

        faiss_path = self.vector_db_dir / f"{document_id}.faiss"
        doc_path = self.documents_dir / f"{document_id}.json"

        if not faiss_path.exists():
            _log.error(f"FAISS DB file {faiss_path} not found for ID {document_id}.")
            return None, None
        if not doc_path.exists():
            _log.error(f"Document file {doc_path} not found for ID {document_id}.")
            return None, None

        try:
            vector_db = faiss.read_index(str(faiss_path))
            with open(doc_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
            
            # Validate document structure (basic check)
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Document {doc_path.name} has unexpected structure. Skipping.")
                return None, None

            self.all_dbs_cache[document_id] = {"vector_db": vector_db, "document": document, "name": document_id}
            return vector_db, document
        except Exception as e:
            _log.error(f"Error loading FAISS DB or document for {document_id}: {e}", exc_info=True)
            return None, None

    @staticmethod
    def get_strings_cosine_similarity(str1: str, str2: str) -> float: # Added type hints
        llm = VectorRetriever.get_llm_for_similarity()
        try:
            embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large") # Make model configurable if needed
            embedding1 = embeddings.data[0].embedding
            embedding2 = embeddings.data[1].embedding
            # Ensure embeddings are numpy arrays for dot product and norm
            emb1_np = np.array(embedding1)
            emb2_np = np.array(embedding2)
            similarity_score = np.dot(emb1_np, emb2_np) / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
            return round(float(similarity_score), 4) # Ensure float
        except Exception as e:
            _log.error(f"Error calculating string cosine similarity: {e}", exc_info=True)
            return 0.0 # Return a default/neutral score on error

    def retrieve_by_document_id(self, document_id: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        vector_db, document = self._load_single_db_and_doc(document_id)

        if vector_db is None or document is None:
            return []
        
        chunks = document.get("content", []) # List of chunk dicts
        if not chunks:
            _log.warning(f"No content chunks found in document {document_id} for vector retrieval.")
            return []
        
        # Ensure k is not greater than the number of items in the index
        num_items_in_index = vector_db.ntotal
        actual_top_n_faiss = min(top_n, num_items_in_index)
        if actual_top_n_faiss == 0 and num_items_in_index > 0: # If top_n was 0 but index has items
             actual_top_n_faiss = num_items_in_index # retrieve all if top_n is 0 but items exist
        elif num_items_in_index == 0:
            _log.warning(f"FAISS index for {document_id} is empty. Cannot search.")
            return []


        try:
            embedding_response = self.llm.embeddings.create(input=query, model="text-embedding-3-large")
            query_embedding = embedding_response.data[0].embedding
        except Exception as e:
            _log.error(f"Failed to get embedding for query '{query}': {e}", exc_info=True)
            return []

        embedding_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding_array) # Normalize query vector for cosine similarity search with IndexFlatIP
        
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n_faiss)
    
        retrieval_results = []
        seen_pages_content = {} # For return_parent_pages logic

        for distance, index in zip(distances[0], indices[0]):
            if index < 0 or index >= len(chunks): # FAISS can return -1 if k > ntotal
                _log.debug(f"Skipping invalid index {index} from FAISS search for doc {document_id}.")
                continue

            score = round(float(distance), 4) # For IndexFlatIP, higher is better (cosine similarity)
            chunk_data = chunks[index]
            page_num = chunk_data.get('original_page', chunk_data.get('page', 0))

            if return_parent_pages:
                if page_num not in seen_pages_content:
                    page_full_text = " ".join(
                        c.get('text','') for c in chunks if c.get('original_page', c.get('page')) == page_num
                    ).strip()
                    seen_pages_content[page_num] = page_full_text

                if not any(res['page'] == page_num for res in retrieval_results):
                    result = {
                        "score": score, # Cosine similarity score
                        "page": page_num,
                        "text": seen_pages_content[page_num],
                        "podcast_id": document_id,
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "score": score,
                    "page": page_num,
                    "text": chunk_data.get('text',''),
                    "speaker": chunk_data.get('original_speaker', chunk_data.get('speaker')),
                    "timestamp": chunk_data.get('original_timestamp', chunk_data.get('timestamp')),
                    "podcast_id": document_id,
                }
                retrieval_results.append(result)
        
        if return_parent_pages:
            retrieval_results.sort(key=lambda x: x['score'], reverse=True)

        return retrieval_results[:top_n]

    def retrieve_all(self, document_id: str) -> List[Dict]:
        """Retrieves all chunks from a specified document."""
        _, document = self._load_single_db_and_doc(document_id)
        if document is None:
            return []
        
        chunks = document.get("content", []) # List of chunk dicts
        all_chunks_with_meta = []
        for idx, chunk_data in enumerate(chunks):
            result = {
                "score": 0.5, # Default score, as this is not a similarity search
                "page": chunk_data.get('original_page', chunk_data.get('page', 0)),
                "text": chunk_data.get('text',''),
                "speaker": chunk_data.get('original_speaker', chunk_data.get('speaker')),
                "timestamp": chunk_data.get('original_timestamp', chunk_data.get('timestamp')),
                "podcast_id": document_id,
                "chunk_id_in_doc": idx
            }
            all_chunks_with_meta.append(result)
        return all_chunks_with_meta


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, bm25_db_dir: Optional[Path] = None): # Added bm25_db_dir
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        if bm25_db_dir:
            self.bm25_retriever = BM25Retriever(bm25_db_dir, documents_dir)
        else:
            self.bm25_retriever = None
        self.reranker = LLMReranker() # Assumes LLMReranker is generic enough
        
    def retrieve_by_document_id( # Renamed from retrieve_by_company_name
        self, 
        document_id: str, # Renamed from company_name
        query: str, 
        llm_reranking_sample_size: int = 20, # Increased default for better reranking pool
        # documents_batch_size: int = 2, # This param seems to be for reranker, not initial retrieval
        top_n: int = 5, # Default top_n after reranking
        # llm_weight: float = 0.7, # This param is for reranker
        return_parent_pages: bool = False,
        use_bm25_for_initial_retrieval: bool = False # New flag
    ) -> List[Dict]:
        
        initial_candidates = []
        if use_bm25_for_initial_retrieval and self.bm25_retriever:
            initial_candidates = self.bm25_retriever.retrieve_by_document_id(
                document_id=document_id,
                query=query,
                top_n=llm_reranking_sample_size, # Use this to get enough candidates for reranking
                return_parent_pages=return_parent_pages
            )
        else: # Default to vector retrieval
            initial_candidates = self.vector_retriever.retrieve_by_document_id(
                document_id=document_id,
                query=query,
                top_n=llm_reranking_sample_size, # Use this to get enough candidates for reranking
                return_parent_pages=return_parent_pages
            )
        
        if not initial_candidates:
            return []

        # Rerank results using LLM (LLMReranker.rerank_documents expects 'documents' list)
        # We need to ensure the reranker can handle the 'score' (distance or BM25 score) appropriately.
        # For now, assuming reranker primarily uses text content for its own scoring.
        # The llm_weight and documents_batch_size are part of reranker's call.
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=initial_candidates, # Pass the list of dicts
            # documents_batch_size=documents_batch_size, # Pass if reranker uses it
            # llm_weight=llm_weight # Pass if reranker uses it
        )
        
        return reranked_results[:top_n]

    def search_across_all_podcasts(
        self,
        query: str,
        top_n_overall: int = 10,
        top_n_per_doc: int = 3, # How many candidates to fetch from each doc initially
        rerank_with_llm: bool = True
        # llm_reranking_sample_size not directly used here, top_n_per_doc controls candidates from each doc
        # llm_weight and documents_batch_size are for the reranker call
    ) -> List[Dict]:
        _log.info(f"Starting search across all podcasts for query: '{query}'")
        all_candidate_chunks = []

        # Discover all document IDs (podcast_ids)
        # Assuming document IDs are the stems of the JSON files in documents_dir
        # or stems of .faiss files in vector_db_dir
        available_doc_ids = list(set(p.stem for p in self.vector_retriever.documents_dir.glob("*.json")))
        if not available_doc_ids:
            _log.warning("No documents found to search across.")
            return []

        _log.info(f"Found {len(available_doc_ids)} potential podcast documents to search.")

        for doc_id in available_doc_ids:
            _log.debug(f"Retrieving from document: {doc_id}")
            # Get candidates from each document using vector search primarily for cross-doc
            # BM25 might be too slow if not pre-loaded for all docs.
            doc_candidates = self.vector_retriever.retrieve_by_document_id(
                document_id=doc_id,
                query=query,
                top_n=top_n_per_doc,
                return_parent_pages=False # Get fine-grained chunks for cross-doc ranking
            )
            # Add doc_id to each chunk explicitly if not already there (it should be from retrieve_by_document_id)
            for candidate in doc_candidates:
                candidate['podcast_id'] = doc_id # Ensure it's there
            all_candidate_chunks.extend(doc_candidates)

        if not all_candidate_chunks:
            _log.info("No candidates found across any podcast for the query.")
            return []

        _log.info(f"Collected {len(all_candidate_chunks)} total candidates from all podcasts.")

        # Sort by score (descending for similarity, ascending for distance if used)
        # VectorRetriever results have 'score' (cosine similarity, higher is better)
        all_candidate_chunks.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        # If LLM reranking is enabled
        if rerank_with_llm:
            # Rerank the top N candidates from the initial sort.
            # The number of candidates to rerank can be larger than top_n_overall.
            candidates_for_reranking = all_candidate_chunks[:max(top_n_overall * 3, 30)] # Rerank more than needed
            _log.info(f"Reranking {len(candidates_for_reranking)} candidates with LLM.")
            final_results = self.reranker.rerank_documents(
                query=query,
                documents=candidates_for_reranking
                # Default reranker params for batch_size, weight will be used from LLMReranker
            )
        else:
            final_results = all_candidate_chunks # Already sorted by initial retrieval score

        return final_results[:top_n_overall]
