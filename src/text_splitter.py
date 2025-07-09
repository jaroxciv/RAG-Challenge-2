import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

_log = logging.getLogger(__name__)

class TextSplitter():
    def __init__(self, processing_flow: str = "annual_report", chunk_size: int = 300, chunk_overlap: int = 50):
        self.processing_flow = processing_flow
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Initialize the text splitter once
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o", # Consider making this configurable
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """Group serialized tables by page number (specific to annual_report flow)."""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
            page = table.get('page')
            if page is None:
                continue # Skip tables without page numbers

            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block.get("information_block", "")
                for block in table.get("serialized", {}).get("information_blocks", [])
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table.get("table_id"),
                "length_tokens": self.count_tokens(table_text)
            })
        return tables_by_page

    def _split_document_content(self, document_json: Dict[str, Any], serialized_tables_path: Optional[Path] = None) -> Dict[str, Any]:
        """Splits document content into smaller chunks based on the processing flow."""
        final_chunks = []
        chunk_id_counter = 0
        
        metainfo = document_json.get('metainfo', {})
        podcast_id = metainfo.get('podcast_id', metainfo.get('sha1_name', 'unknown_doc'))

        if self.processing_flow == "podcast":
            dialogue_items = document_json.get('content', []) # This is List[Dict] from JsonPodcastProcessor
            for dialogue_chunk in dialogue_items:
                original_text = dialogue_chunk.get('text', '')
                original_page = dialogue_chunk.get('page', 0)
                speaker = dialogue_chunk.get('speaker', 'N/A')
                timestamp = dialogue_chunk.get('timestamp', 'N/A')

                # Split the dialogue text itself
                sub_chunks_text = self.text_splitter.split_text(original_text)

                for text_piece in sub_chunks_text:
                    final_chunks.append({
                        'id': chunk_id_counter,
                        'podcast_id': podcast_id, # Add podcast_id for linking
                        'text': text_piece,
                        'original_speaker': speaker,
                        'original_timestamp': timestamp,
                        'original_page': original_page,
                        'length_tokens': self.count_tokens(text_piece),
                        'type': 'dialogue_segment'
                    })
                    chunk_id_counter += 1
        
        elif self.processing_flow == "annual_report":
            tables_by_page = {}
            if serialized_tables_path and serialized_tables_path.exists():
                with open(serialized_tables_path, 'r', encoding='utf-8') as f:
                    parsed_report_for_tables = json.load(f)
                tables_by_page = self._get_serialized_tables_by_page(parsed_report_for_tables.get('tables', []))
            
            # Assuming annual report content is under document_json['content']['pages']
            # This structure comes from PageTextPreparation in the original pipeline
            pages_data = document_json.get('content', {}).get('pages', [])
            for page_content in pages_data:
                page_text = page_content.get('text', '')
                page_num = page_content.get('page', 0)

                page_sub_chunks_text = self.text_splitter.split_text(page_text)
                for text_piece in page_sub_chunks_text:
                    final_chunks.append({
                        'id': chunk_id_counter,
                        'document_id': podcast_id, # Use generic document_id or sha1_name
                        'text': text_piece,
                        'page': page_num,
                        'length_tokens': self.count_tokens(text_piece),
                        'type': 'content'
                    })
                    chunk_id_counter += 1

                if tables_by_page and page_num in tables_by_page:
                    for table_chunk in tables_by_page[page_num]:
                        table_chunk['id'] = chunk_id_counter
                        table_chunk['type'] = 'serialized_table'
                        table_chunk['document_id'] = podcast_id
                        # 'page' and 'length_tokens' are already in table_chunk
                        final_chunks.append(table_chunk)
                        chunk_id_counter += 1
        else:
            _log.warning(f"Unsupported processing_flow: {self.processing_flow} in TextSplitter.")
            # Return original content or empty chunks? For now, empty.
            return {'metainfo': metainfo, 'content': {'chunks': []}}

        # The ingestion scripts expect a structure like: {'metainfo': ..., 'content': {'chunks': [...]}}
        # However, our adapted ingestion script (ingestion.py) now expects:
        # {'metainfo': ..., 'content': [chunk1, chunk2, ...]} where each chunk is a dict with 'text'.
        # So, the output of this function should be just the list of final_chunks.
        # The calling function `split_all_documents` will then construct the final JSON structure.
        # Let's adjust: this method returns the list of chunks.
        # The wrapper method will put it into the correct file structure.
        return final_chunks


    def count_tokens(self, string: str, encoding_name="o200k_base") -> int:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            tokens = encoding.encode(string)
            token_count = len(tokens)
            return token_count
        except Exception as e:
            _log.error(f"Error counting tokens: {e}. Defaulting to len(string)/4.")
            return len(string) // 4 # Fallback heuristic


    def split_all_documents(self, input_json_dir: Path, output_chunk_dir: Path, serialized_tables_dir: Optional[Path] = None):
        """
        Splits all documents from input_json_dir and saves them to output_chunk_dir.
        - input_json_dir: Directory with parsed JSONs (from PDFParser).
        - output_chunk_dir: Directory to save chunked JSONs (for ingestion).
        - serialized_tables_dir: Only used for 'annual_report' flow if tables are separate.
        """
        all_input_json_paths = list(input_json_dir.glob("*.json"))
        if not all_input_json_paths:
            _log.info(f"No JSON documents found in {input_json_dir} to split.")
            return

        output_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        for doc_path in all_input_json_paths:
            _log.info(f"Splitting document: {doc_path.name}")
            try:
                with open(doc_path, 'r', encoding='utf-8') as file:
                    doc_data = json.load(file)

                # Determine path for serialized tables if applicable (for annual reports)
                current_serialized_tables_path = None
                if self.processing_flow == "annual_report" and serialized_tables_dir:
                    current_serialized_tables_path = serialized_tables_dir / doc_path.name
                    if not current_serialized_tables_path.exists():
                        _log.warning(f"Serialized tables file not found for {doc_path.name} at {current_serialized_tables_path}")
                        current_serialized_tables_path = None # Proceed without it

                # _split_document_content now returns a list of chunk dictionaries
                list_of_final_chunks = self._split_document_content(doc_data, current_serialized_tables_path)
                
                # The ingestion scripts expect each output JSON file to have:
                # {'metainfo': original_metainfo, 'content': [chunk1, chunk2, ...]}
                # where chunk1, chunk2 are the final small chunk dictionaries.
                # The 'content' key in the ingestion script refers to this list of chunks.
                output_json_content = {
                    "metainfo": doc_data.get("metainfo", {}), # Preserve original metainfo
                    "content": list_of_final_chunks # This is the list of small chunk dicts
                }
                
                with open(output_chunk_dir / doc_path.name, 'w', encoding='utf-8') as file:
                    json.dump(output_json_content, file, indent=2, ensure_ascii=False)
            
            except Exception as e:
                _log.error(f"Error splitting document {doc_path.name}: {e}", exc_info=True)
                
        _log.info(f"Finished splitting {len(all_input_json_paths)} documents. Output in {output_chunk_dir}")
