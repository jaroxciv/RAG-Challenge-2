import os
import time
import logging
import re
import json
# from tabulate import tabulate # No longer needed for podcast parsing
from pathlib import Path
from typing import Iterable, List, Dict, Any

# from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
# from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult #, Table # Table might not be needed

_log = logging.getLogger(__name__)

# Regex to capture podcast transcript lines
# Assumes timestamps like [00:00:00.00] or [00:00:00]
# Speaker names can contain spaces. Text can be anything.
PODCAST_LINE_RE = re.compile(r"^\s*\[(\d{2}:\d{2}:\d{2}(?:\.\d{2,3})?)\]\s*([^:]+):\s*(.*)")
EPISODE_TITLE_RE = re.compile(r"Episode\s+(\d+)", re.IGNORECASE)
YOUTUBE_URL_RE = re.compile(r"(https?://youtu\.be/[^\s]+)")


def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, debug_data_path): # Removed metadata_lookup
    """Helper function to process a chunk of PDFs in a separate process."""
    # Create a new parser instance for this process
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads,
        # csv_metadata_path=None # Removed
    )
    # parser.metadata_lookup = metadata_lookup # Removed
    parser.debug_data_path = debug_data_path
    parser.parse_and_export(pdf_paths)
    return f"Processed {len(pdf_paths)} PDFs."

class PDFParser:
    def __init__(
        self,
        pdf_backend=DoclingParseV2DocumentBackend,
        output_dir: Path = Path("./parsed_pdfs"),
        num_threads: int = None,
        # csv_metadata_path: Path = None, # Removed
    ):
        self.pdf_backend = pdf_backend
        self.output_dir = output_dir
        self.doc_converter = self._create_document_converter()
        self.num_threads = num_threads
        # self.metadata_lookup = {} # Removed
        self.debug_data_path = None

        # if csv_metadata_path is not None: # Removed
        #     self.metadata_lookup = self._parse_csv_metadata(csv_metadata_path) # Removed
            
        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    # @staticmethod # Removed
    # def _parse_csv_metadata(csv_path: Path) -> dict: # Removed
    #     """Parse CSV file and create a lookup dictionary with sha1 as key.""" # Removed
    #     import csv # Removed
    #     metadata_lookup = {} # Removed
        
    #     with open(csv_path, 'r', encoding='utf-8') as csvfile: # Removed
    #         reader = csv.DictReader(csvfile) # Removed
    #         for row in reader: # Removed
    #             # Handle both old and new CSV formats for company name # Removed
    #             company_name = row.get('company_name', row.get('name', '')).strip('"') # Removed
    #             metadata_lookup[row['sha1']] = { # Removed
    #                 'company_name': company_name # Removed
    #             } # Removed
    #     return metadata_lookup # Removed

    def _create_document_converter(self) -> "DocumentConverter": # type: ignore
        """Creates and returns a DocumentConverter with default pipeline options."""
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions # Removed TableFormerMode
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True # Keep OCR, it's useful for PDFs from images
        ocr_options = EasyOcrOptions(lang=['en'], force_full_page_ocr=False)
        pipeline_options.ocr_options = ocr_options
        # pipeline_options.do_table_structure = True # Disable table processing
        # pipeline_options.table_structure_options.do_cell_matching = True # Disable
        # pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE # Disable
        pipeline_options.do_table_structure = False # Explicitly disable
        
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend
            )
        }
        
        return DocumentConverter(format_options=format_options)

    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable[ConversionResult]:
        conv_results = self.doc_converter.convert_all(source=input_doc_paths)
        return conv_results
    
    def process_documents(self, conv_results: Iterable[ConversionResult]):
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failure_count = 0

        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                # Pass None for metadata_lookup as it's removed from JsonReportProcessor init
                processor = JsonPodcastProcessor(debug_data_path=self.debug_data_path)
                
                # Normalize the document data to ensure sequential pages (still useful)
                data = conv_res.document.export_to_dict()
                # normalized_data = self._normalize_page_sequence(data) # Normalization might not be critical if content flows well
                
                # The new processor will take raw 'data'
                processed_podcast_json = processor.assemble_podcast_json(conv_res, data)
                doc_filename = conv_res.input.file.stem
                if self.output_dir is not None:
                    with (self.output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
                        json.dump(processed_podcast_json, fp, indent=2, ensure_ascii=False)
            else:
                failure_count += 1
                _log.info(f"Document {conv_res.input.file} failed to convert.")

        _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
        return success_count, failure_count

    # _normalize_page_sequence might still be useful if PDFs have blank pages or page order issues.
    # For now, let's assume clean PDFs and potentially remove it later if not needed.
    def _normalize_page_sequence(self, data: dict) -> dict:
        """Ensure that page numbers in content are sequential by filling gaps with empty pages."""
        if 'content' not in data or not data['content']: # check if content is empty
            return data
        
        normalized_data = data.copy()
        
        existing_pages = {page['page'] for page in data['content'] if 'page' in page}
        if not existing_pages: # if no pages have 'page' key
             _log.warning("No 'page' key found in content items during normalization.")
             return data # return original data if no page numbers are found

        max_page = max(existing_pages) if existing_pages else 0
        
        empty_page_template = {
            "content": [],
            "page_dimensions": {}
        }
        
        new_content = []
        for page_num in range(1, max_page + 1):
            page_content = next(
                (page for page in data['content'] if page.get('page') == page_num),
                {"page": page_num, **empty_page_template}
            )
            new_content.append(page_content)
        
        normalized_data['content'] = new_content
        return normalized_data

    def parse_and_export(self, input_doc_paths: List[Path] = None, doc_dir: Path = None):
        start_time = time.time()
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))
        
        if not input_doc_paths:
            _log.info("No PDF documents found to process.")
            return

        total_docs = len(input_doc_paths)
        _log.info(f"Starting to process {total_docs} documents")
        
        conv_results = self.convert_documents(input_doc_paths)
        success_count, failure_count = self.process_documents(conv_results=conv_results)
        elapsed_time = time.time() - start_time

        if failure_count > 0:
            error_message = f"Failed converting {failure_count} out of {total_docs} documents."
            failed_docs_paths = [str(res.input.file) for res in conv_results if res.status != ConversionStatus.SUCCESS]
            failed_docs_log = "Paths of failed docs:\n" + '\n'.join(failed_docs_paths)
            _log.error(error_message)
            _log.error(failed_docs_log)
            # Still raise runtime error, but ensure failed_docs_paths is populated correctly
            # raise RuntimeError(error_message) # Commenting for now to allow partial success

        _log.info(f"{'#'*50}\nCompleted in {elapsed_time:.2f} seconds. Successfully converted {success_count}/{total_docs} documents.\n{'#'*50}")

    def parse_and_export_parallel(
        self,
        input_doc_paths: List[Path] = None,
        doc_dir: Path = None,
        optimal_workers: int = 10,
        chunk_size: int = None
    ):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        if not input_doc_paths:
            _log.info("No PDF documents found for parallel processing.")
            return

        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")
        
        cpu_count = multiprocessing.cpu_count()
        
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs) if total_pdfs > 0 else 1
        
        if chunk_size is None:
            chunk_size = max(1, total_pdfs // optimal_workers if optimal_workers > 0 else total_pdfs)
        
        chunks = [
            input_doc_paths[i : i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]

        start_time = time.time()
        processed_count = 0
        
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    chunk,
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
                    # self.metadata_lookup, # Removed
                    self.debug_data_path
                )
                for chunk in chunks
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Assuming result format is "Processed X PDFs."
                    num_processed_in_chunk = int(re.search(r"Processed (\d+) PDFs", result).group(1))
                    processed_count += num_processed_in_chunk
                    _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                except Exception as e:
                    _log.error(f"Error processing chunk: {str(e)}")
                    # raise # Decide if one failed chunk should stop all
        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonPodcastProcessor: # Renamed from JsonReportProcessor
    def __init__(self, debug_data_path: Path = None): # Removed metadata_lookup
        # self.metadata_lookup = metadata_lookup or {} # Removed
        self.debug_data_path = debug_data_path

    def assemble_podcast_json(self, conv_result: ConversionResult, raw_doc_data: Dict[str, Any]):
        """Assemble the JSON output for a podcast transcript."""
        podcast_json = {}
        # Use raw_doc_data as it's already a dict from conv_result.document.export_to_dict()
        podcast_json['metainfo'] = self._assemble_metainfo(raw_doc_data)
        # The 'content' key in the output JSON should be a list of chunks for ingestion.py
        podcast_json['content'] = self._assemble_content_chunks(raw_doc_data)

        # Tables and Pictures are not relevant for podcasts, return empty lists.
        podcast_json['tables'] = []
        podcast_json['pictures'] = []

        if self.debug_data_path:
            self._debug_data(raw_doc_data) # Debug raw docling output if needed
        return podcast_json
    
    def _extract_episode_info(self, texts_data: List[Dict[str, Any]]):
        episode_number = None
        youtube_url = None
        # Search for episode number and URL typically in the first few text blocks
        for text_block in texts_data[:10]: # Check first 10 text blocks
            text = text_block.get('text', '')
            if not episode_number:
                match = EPISODE_TITLE_RE.search(text)
                if match:
                    episode_number = match.group(1)
            if not youtube_url:
                match = YOUTUBE_URL_RE.search(text)
                if match:
                    youtube_url = match.group(1)
            if episode_number and youtube_url:
                break
        return episode_number, youtube_url

    def _assemble_metainfo(self, doc_data: Dict[str, Any]):
        metainfo = {}
        # Use filename as a base for ID, can be made more robust
        filename_stem = Path(doc_data['origin']['filename']).stem
        metainfo['podcast_id'] = filename_stem
        
        all_texts = doc_data.get('texts', [])
        episode_num, youtube_url = self._extract_episode_info(all_texts)

        metainfo['episode_number'] = episode_num if episode_num else 'N/A'
        metainfo['youtube_url'] = youtube_url if youtube_url else 'N/A'

        # Page count can be derived from the structure docling provides if pages are distinct entities
        # Or by finding max page_no in text provenances.
        # For simplicity, let's count unique page numbers from text blocks.
        page_numbers = set()
        if 'texts' in doc_data:
            for text_item in doc_data['texts']:
                if 'prov' in text_item and text_item['prov']:
                    page_numbers.add(text_item['prov'][0]['page_no'])
        metainfo['pages_amount'] = len(page_numbers) if page_numbers else 0

        metainfo['dialogue_lines_amount'] = 0 # This will be updated after content processing
        return metainfo

    def _debug_data(self, data: Dict[str, Any]):
        if self.debug_data_path is None:
            return
        # docling data has 'name' at the root, which is usually the filename
        doc_name = data.get('name', Path(data['origin']['filename']).stem)
        path = self.debug_data_path / f"{doc_name}_docling_raw.json"
        path.parent.mkdir(parents=True, exist_ok=True)    
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _assemble_content_chunks(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes text blocks from docling output, parses podcast dialogue lines,
        and returns a list of chunks, where each chunk is a dictionary
        expected by the ingestion script (e.g., {'text': 'dialogue_line_content', 'page': N}).
        """
        chunks = []
        dialogue_lines_count = 0

        # docling's 'texts' field contains a list of all text blocks in the document
        all_text_blocks = doc_data.get('texts', [])
        
        current_speaker = None
        current_timestamp = None
        accumulated_text = ""

        for text_block in all_text_blocks:
            block_text = text_block.get('text', '').strip()
            page_num = text_block.get('prov', [{}])[0].get('page_no', 0) # Default to 0 if not found

            if not block_text:
                continue

            match = PODCAST_LINE_RE.match(block_text)
            if match: # New speaker line
                # If there was accumulated text, save it as a chunk for the previous speaker
                if accumulated_text:
                    chunks.append({
                        "text": f"[{current_timestamp}] {current_speaker}: {accumulated_text.strip()}",
                        "speaker": current_speaker,
                        "timestamp": current_timestamp,
                        "page": page_num # This might be tricky if dialogue spans pages, use page of start
                    })
                    dialogue_lines_count +=1
                    accumulated_text = "" # Reset accumulator

                current_timestamp, current_speaker, text_after_speaker = match.groups()
                current_speaker = current_speaker.strip()
                accumulated_text = text_after_speaker.strip()
            else: # Continuation of the previous speaker's dialogue
                if current_speaker: # Only append if we are in a speaker block
                    accumulated_text += " " + block_text
        
        # Add any remaining accumulated text as the last chunk
        if accumulated_text and current_speaker:
            chunks.append({
                "text": f"[{current_timestamp}] {current_speaker}: {accumulated_text.strip()}",
                "speaker": current_speaker,
                "timestamp": current_timestamp,
                "page": page_num # Page of the last text block processed for this segment
            })
            dialogue_lines_count +=1

        # Update dialogue lines count in metainfo (if we want to pass metainfo around or log it)
        # For now, the ingestion script only needs the list of chunks.
        # If metainfo needs to be updated, it would require passing it here or returning it.
        # _log.info(f"Extracted {dialogue_lines_count} dialogue lines as chunks.")
        
        # The ingestion script (BM25Ingestor, VectorDBIngestor) expects each item in
        # report_data['content']['chunks'] to be a dict with at least a 'text' key.
        # Our chunks are already in this format.
        return chunks

    # Methods like expand_groups, _process_text_reference, assemble_tables, _table_to_md,
    # assemble_pictures, _process_picture_block are removed as they are specific to
    # the previous report structure and not needed for podcast transcripts.
    # The new _assemble_content_chunks focuses on parsing the specific podcast format.
