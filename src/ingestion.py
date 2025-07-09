import os
import json
import pickle
from typing import List, Union, Dict, Any
from pathlib import Path
from tqdm import tqdm
import logging # Added logging

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt

_log = logging.getLogger(__name__) # Added logger

class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        # Ensure chunks are not empty which can cause issues with BM25
        processed_chunks = [chunk for chunk in chunks if chunk.strip()]
        if not processed_chunks:
            _log.warning("No valid text chunks found for BM25 index creation after filtering empty strings.")
            # Return an empty or dummy index, or handle as an error
            # For now, let's allow creating an index with empty tokenized_chunks if that's how BM25Okapi handles it
            # but it's better to avoid this. If BM25Okapi fails with empty list, this needs adjustment.
            tokenized_chunks = []
        else:
            tokenized_chunks = [chunk.split() for chunk in processed_chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all JSON documents (reports or podcasts) and save individual BM25 indices.
        
        Args:
            all_reports_dir (Path): Directory containing the JSON document files.
            output_dir (Path): Directory where to save the BM25 indices.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_json_paths = list(all_reports_dir.glob("*.json")) # Renamed for clarity

        if not all_json_paths:
            _log.info(f"No JSON files found in {all_reports_dir} for BM25 processing.")
            return

        for json_path in tqdm(all_json_paths, desc="Processing documents for BM25"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f) # Renamed report_data to data
                
                # The parsed podcast JSON has 'content' as a list of chunk dicts.
                # Each chunk dict has a 'text' key.
                if 'content' not in data or not isinstance(data['content'], list):
                    _log.warning(f"Skipping {json_path.name}: 'content' key missing or not a list.")
                    continue

                text_chunks = [chunk['text'] for chunk in data['content'] if isinstance(chunk, dict) and 'text' in chunk]
                
                if not text_chunks:
                    _log.warning(f"Skipping {json_path.name}: No text chunks found in 'content'.")
                    continue

                bm25_index = self.create_bm25_index(text_chunks)

                # Use 'podcast_id' from metainfo for naming, fallback to filename stem
                metainfo = data.get("metainfo", {})
                doc_id = metainfo.get("podcast_id", metainfo.get("sha1_name", json_path.stem))

                output_file = output_dir / f"{doc_id}.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(bm25_index, f)
            except Exception as e:
                _log.error(f"Error processing {json_path.name} for BM25: {e}", exc_info=True)

        _log.info(f"Processed {len(all_json_paths)} JSON documents for BM25.")

class VectorDBIngestor:
    def __init__(self):
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        # Consider adding error handling for missing API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            _log.error("OPENAI_API_KEY not found in environment variables.")
            raise ValueError("OPENAI_API_KEY is required for VectorDBIngestor.")

        llm = OpenAI(
            api_key=api_key,
            timeout=None, # Consider setting a reasonable timeout
            max_retries=3 # Increased retries
        )
        return llm

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(3)) # Increased stop_after_attempt
    def _get_embeddings(self, text_list: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
        # Ensure text_list contains non-empty strings
        # OpenAI API can handle batching, but check limits. Max tokens per request is an important factor.
        # text-embedding-3-large has a max input token limit (e.g. 8191 tokens).
        # We should batch texts if the total tokens exceed this, or if list is too long.
        # For simplicity, this example processes texts individually if list is too large for one call,
        # but true batching with OpenAI client is more efficient.
        
        valid_texts = [t for t in text_list if isinstance(t, str) and t.strip()]
        if not valid_texts:
            _log.warning("No valid texts provided to _get_embeddings after filtering.")
            return []

        all_embeddings = []
        # Example of simple batching for API call (OpenAI client handles some internal batching for lists)
        # Max items per call for embeddings is often around 2048.
        batch_size = 1024 # Number of texts per API call, adjust based on typical text length and API limits

        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            try:
                response = self.llm.embeddings.create(input=batch_texts, model=model)
                all_embeddings.extend([embedding.embedding for embedding in response.data])
            except Exception as e:
                _log.error(f"Error getting embeddings for a batch: {e}", exc_info=True)
                # Decide on error handling: re-raise, or skip batch, or return partial results
                # For now, re-raise to let tenacity handle retries for the whole operation.
                raise
        
        return all_embeddings


    def _create_vector_db(self, embeddings: List[List[float]]):
        if not embeddings or not embeddings[0]: # Check if embeddings list is empty or first embedding is empty
            _log.warning("Cannot create FAISS index: No embeddings provided.")
            return None # Return None or handle as an error

        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array) # Normalize for cosine similarity with IndexFlatIP
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Using Inner Product (cosine similarity after normalization)
        index.add(embeddings_array)
        return index
    
    def _process_document(self, doc_data: Dict[str, Any]): # Renamed report to doc_data
        # Similar to BM25, 'content' is a list of chunk dicts.
        if 'content' not in doc_data or not isinstance(doc_data['content'], list):
            _log.warning(f"Skipping document for VectorDB: 'content' key missing or not a list.")
            return None

        text_chunks = [chunk['text'] for chunk in doc_data['content'] if isinstance(chunk, dict) and 'text' in chunk and chunk['text'].strip()]

        if not text_chunks:
            _log.warning(f"Skipping document for VectorDB: No text chunks found in 'content'.")
            return None

        embeddings = self._get_embeddings(text_chunks)
        if not embeddings:
            _log.warning(f"Skipping document for VectorDB: No embeddings generated for its chunks.")
            return None

        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path): # Name kept for consistency with caller
        all_json_paths = list(all_reports_dir.glob("*.json")) # Renamed for clarity
        output_dir.mkdir(parents=True, exist_ok=True)

        if not all_json_paths:
            _log.info(f"No JSON files found in {all_reports_dir} for VectorDB processing.")
            return

        for json_path in tqdm(all_json_paths, desc="Processing documents for VectorDB"):
            try:
                with open(json_path, 'r', encoding='utf-8') as file:
                    doc_data = json.load(f) # Renamed report_data to doc_data

                index = self._process_document(doc_data)
                if index is None: # Skip if index creation failed (e.g. no text, no embeddings)
                    _log.warning(f"Skipping FAISS index saving for {json_path.name} due to processing error or no data.")
                    continue

                metainfo = doc_data.get("metainfo", {})
                doc_id = metainfo.get("podcast_id", metainfo.get("sha1_name", json_path.stem))

                faiss_file_path = output_dir / f"{doc_id}.faiss"
                faiss.write_index(index, str(faiss_file_path))
            except Exception as e:
                _log.error(f"Error processing {json_path.name} for VectorDB: {e}", exc_info=True)

        _log.info(f"Processed {len(all_json_paths)} JSON documents for VectorDB.")