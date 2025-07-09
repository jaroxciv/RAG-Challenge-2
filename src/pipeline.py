from dataclasses import dataclass, field
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
from typing import Optional

from src.pdf_parsing import PDFParser # PDFParser now takes no csv_metadata_path for podcasts
from src.parsed_reports_merging import PageTextPreparation # May need changes or replacement for podcasts
from src.text_splitter import TextSplitter # Should be adaptable
from src.ingestion import VectorDBIngestor, BM25Ingestor # Already adapted
from src.questions_processing import QuestionsProcessor # Adapted with processing_flow
from src.tables_serialization import TableSerializer # Not used for podcasts

@dataclass
class PipelineConfig:
    root_path: Path
    processing_flow: str = "annual_report"  # "annual_report" or "podcast"
    questions_file_name: str = "questions.json"
    pdf_input_dir_name: str = "pdf_reports" # Generic name, will be 'pdf_reports' or 'podcast_pdfs'
    subset_name: Optional[str] = "subset.csv" # Optional, used for annual reports
    serialized_tables: bool = False # Specific to annual reports
    config_suffix: str = ""

    # Derived paths
    subset_path: Optional[Path] = field(init=False)
    questions_file_path: Path = field(init=False)
    pdf_input_dir: Path = field(init=False)
    answers_file_path: Path = field(init=False)
    debug_data_path: Path = field(init=False)
    databases_path: Path = field(init=False)

    vector_db_dir: Path = field(init=False)
    documents_dir: Path = field(init=False) # Dir for chunked JSONs, input to ingestion
    bm25_db_dir: Path = field(init=False) # Changed from bm25_db_path to be a directory

    # Stage-specific output directory names (relative to debug_data_path or databases_path)
    # These will be adapted based on processing_flow
    parsed_output_dirname: str = field(init=False)
    parsed_output_debug_dirname: str = field(init=False)
    # merged_reports_dirname: str = field(init=False) # Merging might not apply to podcasts
    # reports_markdown_dirname: str = field(init=False) # Markdown export might not apply

    # Full paths for stage outputs
    parsed_output_path: Path = field(init=False) # Output of PDFParser (e.g. parsed_podcasts)
    parsed_output_debug_path: Optional[Path] = field(init=False)
    # merged_output_path: Path = field(init=False) # Output of merging step

    def __post_init__(self):
        # Determine suffix for paths based on table serialization (only for annual reports)
        path_suffix = "_ser_tab" if self.processing_flow == "annual_report" and self.serialized_tables else ""

        if self.subset_name:
            self.subset_path = self.root_path / self.subset_name
        else:
            self.subset_path = None

        self.questions_file_path = self.root_path / self.questions_file_name
        self.pdf_input_dir = self.root_path / self.pdf_input_dir_name
        
        self.answers_file_path = self.root_path / f"answers{self.config_suffix}{path_suffix}.json"
        self.debug_data_path = self.root_path / "debug_data" # Common debug root
        self.databases_path = self.root_path / f"databases{path_suffix}" # Common databases root
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        # documents_dir is where chunked JSONs go, becoming input for ingestion
        # For podcasts, this will be output of TextSplitter applied to parsed_output_path
        self.documents_dir = self.databases_path / f"chunked_{self.processing_flow}"
        self.bm25_db_dir = self.databases_path / "bm25_dbs" # BM25 indexes will be stored here

        # Adapt directory names based on processing_flow
        if self.processing_flow == "podcast":
            self.parsed_output_dirname = "01_parsed_podcasts"
            self.parsed_output_debug_dirname = "01_parsed_podcasts_debug"
            # Merging and markdown might not be directly applicable or will be different
        else: # Default to annual_report flow
            self.parsed_output_dirname = "01_parsed_reports"
            self.parsed_output_debug_dirname = "01_parsed_reports_debug"
            # self.merged_reports_dirname = f"02_merged_reports{path_suffix}"
            # self.reports_markdown_dirname = f"03_reports_markdown{path_suffix}"

        self.parsed_output_path = self.debug_data_path / self.parsed_output_dirname
        if self.parsed_output_debug_dirname: # Can be optional
             self.parsed_output_debug_path = self.debug_data_path / self.parsed_output_debug_dirname
        else:
            self.parsed_output_debug_path = None
        # self.merged_output_path = self.debug_data_path / self.merged_reports_dirname if hasattr(self, 'merged_reports_dirname') else None


@dataclass
class RunConfig:
    processing_flow: str = "annual_report" # New field: "annual_report" or "podcast"
    use_serialized_tables: bool = False # Relevant for annual_report flow
    parent_document_retrieval: bool = False
    # use_vector_dbs: bool = True # Assumed True if create_vector_dbs is called
    # use_bm25_db: bool = False # Assumed True if create_bm25_db is called
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 10 # For QuestionsProcessor
    team_email: str = "default.email@example.com"
    submission_name: str = "Default Submission"
    pipeline_details: str = "Default RAG Pipeline"
    submission_file: bool = True
    full_context: bool = False # For QuestionsProcessor to use full document context
    api_provider: str = "openai"
    answering_model: str = "gpt-4o-mini-2024-07-18"
    config_suffix: str = "" # Appended to answer filenames, etc.
    # pdf_input_dir_name for this run, allows overriding PipelineConfig default
    pdf_input_dir_name_override: Optional[str] = None
    questions_file_name_override: Optional[str] = None
    subset_name_override: Optional[str] = None


class Pipeline:
    def __init__(self, root_path: Path, run_config: RunConfig = RunConfig()):
        self.run_config = run_config

        # Determine names for path config based on overrides in run_config or defaults
        pdf_input_name = run_config.pdf_input_dir_name_override or \
                         ("podcast_pdfs" if run_config.processing_flow == "podcast" else "pdf_reports")
        questions_name = run_config.questions_file_name_override or "questions.json"
        subset_name_val = run_config.subset_name_override # Can be None
        if run_config.processing_flow == "podcast":
            subset_name_val = None # Podcasts don't use subset file for company metadata

        self.paths = PipelineConfig(
            root_path=root_path,
            processing_flow=run_config.processing_flow,
            subset_name=subset_name_val,
            questions_file_name=questions_name,
            pdf_input_dir_name=pdf_input_name,
            serialized_tables=run_config.use_serialized_tables if run_config.processing_flow == "annual_report" else False,
            config_suffix=run_config.config_suffix
        )

        if self.run_config.processing_flow == "annual_report" and self.paths.subset_path:
            self._convert_json_to_csv_if_needed()

    def _convert_json_to_csv_if_needed(self):
        if not self.paths.subset_path or not self.paths.subset_path.name.endswith(".csv"):
             # This logic assumes subset_path is for a CSV. If it's for subset.json, adapt.
             # For now, this method is specific to the annual_report flow's subset.csv
            json_equivalent_path = self.paths.root_path / "subset.json" # Default name to check
            if json_equivalent_path.exists() and (not self.paths.subset_path or not self.paths.subset_path.exists()):
                _log.info(f"Attempting to convert {json_equivalent_path} to {self.paths.subset_path}")
                try:
                    with open(json_equivalent_path, 'r') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    df.to_csv(self.paths.subset_path, index=False)
                    _log.info(f"Converted {json_equivalent_path} to {self.paths.subset_path}")
                except Exception as e:
                    _log.error(f"Error converting JSON to CSV: {str(e)}")
            return # Skip if subset_path is not for CSV or if source JSON doesn't exist

    @staticmethod
    def download_docling_models(): 
        # This method seems generic enough.
        logging.basicConfig(level=logging.INFO) # Changed to INFO
        # Create a dummy PDF path for the dummy run if needed, or ensure dummy_report.pdf exists
        dummy_pdf_path = here() / "src/dummy_report.pdf"
        if not dummy_pdf_path.exists():
            _log.warning(f"Dummy PDF for model download not found at {dummy_pdf_path}. Creating a placeholder.")
            # Create a simple placeholder PDF if it doesn't exist to avoid error,
            # though docling might handle this. For safety:
            try:
                from reportlab.pdfgen import canvas
                c = canvas.Canvas(str(dummy_pdf_path))
                c.drawString(100, 750, "Dummy PDF for model download.")
                c.save()
                _log.info(f"Created placeholder dummy PDF at {dummy_pdf_path}")
            except ImportError:
                _log.error("ReportLab not installed. Cannot create dummy PDF. Please ensure dummy_report.pdf exists or install reportlab.")
                return
            except Exception as e:
                _log.error(f"Could not create dummy PDF: {e}")
                return

        parser = PDFParser(output_dir=here() / "temp_docling_output") # Use a temp output
        parser.parse_and_export(input_doc_paths=[dummy_pdf_path])
        # Clean up dummy PDF and temp output if desired, or leave for inspection
        # os.remove(dummy_pdf_path) # Optional: remove dummy after use
        # shutil.rmtree(here() / "temp_docling_output") # Optional: remove temp dir

    def _get_pdf_parser_instance(self) -> PDFParser:
        """Helper to create PDFParser instance based on flow."""
        parser_kwargs = {
            "output_dir": self.paths.parsed_output_path,
            # csv_metadata_path is not passed to PDFParser constructor anymore directly
            # PDFParser internal logic for metadata was removed/simplified
        }
        # Our modified PDFParser doesn't take csv_metadata_path in __init__
        # if self.run_config.processing_flow == "annual_report" and self.paths.subset_path:
        #     parser_kwargs["csv_metadata_path"] = self.paths.subset_path
            
        pdf_parser = PDFParser(**parser_kwargs)
        
        if self.paths.parsed_output_debug_path:
            pdf_parser.debug_data_path = self.paths.parsed_output_debug_path
        return pdf_parser

    def parse_pdf_input_sequential(self): # Renamed from parse_pdf_reports_sequential
        logging.basicConfig(level=logging.INFO)
        pdf_parser = self._get_pdf_parser_instance()
        pdf_parser.parse_and_export(doc_dir=self.paths.pdf_input_dir)
        _log.info(f"PDFs parsed and saved to {self.paths.parsed_output_path}")

    def parse_pdf_input_parallel(self, chunk_size: int = 2, max_workers: int = 10): # Renamed
        logging.basicConfig(level=logging.INFO)
        pdf_parser = self._get_pdf_parser_instance()
        input_doc_paths = list(self.paths.pdf_input_dir.glob("*.pdf"))
        if not input_doc_paths:
            _log.info(f"No PDF files found in {self.paths.pdf_input_dir} to parse.")
            return
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        _log.info(f"PDFs parsed (parallel) and saved to {self.paths.parsed_output_path}")

    def serialize_tables(self, max_workers: int = 10): # Only for annual_report flow
        if self.run_config.processing_flow != "annual_report":
            _log.info("Skipping table serialization for non-annual_report flow.")
            return
        if not self.run_config.use_serialized_tables:
            _log.info("Skipping table serialization as use_serialized_tables is False.")
            return

        logging.basicConfig(level=logging.INFO)
            self.paths.parsed_output_path, # Should be parsed_reports_path for annual reports
            max_workers=max_workers
        )
        _log.info(f"Table serialization completed for {self.paths.parsed_output_path}")


    def simplify_parsed_output(self): # Renamed from merge_reports
        """
        For annual reports: Merges complex JSONs into a simpler structure.
        For podcasts: The output from JsonPodcastProcessor is a list of chunks.
                      This step might involve ensuring it's in the right format for TextSplitter,
                      or it could be a no-op if the parsed output is already suitable.
                      The TextSplitter expects input JSONs to have a list of pages,
                      and each page has 'text' and 'page_number'.
                      Our podcast parsed output is: {'metainfo': {...}, 'content': [{'text': ..., 'page': ...}]}
                      We need to adapt this or TextSplitter.
                      For now, let's assume this step prepares the data for chunk_documents.
        """
        if self.run_config.processing_flow == "annual_report":
            logging.basicConfig(level=logging.INFO)
            ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
            # The output of this step for annual reports is used by chunk_reports.
            # Let's define self.paths.simplified_output_path for this.
            # For consistency, chunk_documents should always read from a path like self.paths.simplified_output_path
            simplified_output_path = self.debug_data_path / f"02_simplified_{self.run_config.processing_flow}"
            simplified_output_path.mkdir(parents=True, exist_ok=True)

            _ = ptp.process_reports(
                reports_dir=self.paths.parsed_output_path, # Input is parsed reports
                output_dir=simplified_output_path
            )
            _log.info(f"Annual reports simplified and saved to {simplified_output_path}")
        elif self.run_config.processing_flow == "podcast":
            # The output of PDFParser for podcasts (self.paths.parsed_output_path)
            # contains JSONs like: {'metainfo': {...}, 'content': [{'text': ..., 'page': ...}]}
            # TextSplitter.split_all_reports expects a directory of JSONs, where each JSON
            # is a list of pages, and each page has 'text' and 'page_number'.
            # We need to transform our podcast parsed JSONs into this format or adapt TextSplitter.
            # For now, this step will be a simple pass-through or minimal transformation.
            # Let's make chunk_documents read directly from self.paths.parsed_output_path for podcasts for now,
            # and assume TextSplitter is adapted.
            _log.info(f"Podcast parsed output at {self.paths.parsed_output_path} will be used directly by chunking.")
            # No actual file operation here, chunk_documents will handle it.
        else:
            _log.warning(f"Unknown processing flow: {self.run_config.processing_flow}. Skipping simplification.")


    def export_to_markdown(self): # Renamed from export_reports_to_markdown
        """Export processed documents to markdown format for review (if applicable)."""
        if self.run_config.processing_flow == "annual_report":
            logging.basicConfig(level=logging.INFO)
            ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
            markdown_output_path = self.debug_data_path / f"03_markdown_{self.run_config.processing_flow}"
            markdown_output_path.mkdir(parents=True, exist_ok=True)
            ptp.export_to_markdown(
                reports_dir=self.paths.parsed_output_path, # Uses originally parsed reports
                output_dir=markdown_output_path
            )
            _log.info(f"Documents exported to markdown at {markdown_output_path}")
        else:
            _log.info(f"Markdown export not implemented for {self.run_config.processing_flow} flow.")


    def chunk_documents(self): # Renamed from chunk_reports
        """Split processed documents into smaller chunks for ingestion."""
        logging.basicConfig(level=logging.INFO)
        text_splitter = TextSplitter(processing_flow=self.run_config.processing_flow)
        
        # Determine input directory for chunking
        # For annual reports, it's the output of simplify_parsed_output (PageTextPreparation)
        # For podcasts, it's directly the output of PDFParser (JsonPodcastProcessor)
        input_dir_for_chunking = self.paths.parsed_output_path
        if self.run_config.processing_flow == "annual_report":
            # If simplify_parsed_output creates a specific dir, use that.
            # Assuming simplify_parsed_output's output is what TextSplitter expects for annual_reports.
            # Let's refine this: simplify_parsed_output for reports should produce data in a location
            # that chunk_documents then reads. For podcasts, chunk_documents reads from parsed_output_path.
            simplified_output_path = self.debug_data_path / f"02_simplified_{self.run_config.processing_flow}"
            if simplified_output_path.exists(): # Check if simplification step ran and produced output
                 input_dir_for_chunking = simplified_output_path
            else:
                 _log.warning(f"Simplified output path {simplified_output_path} not found for annual reports. Using parsed_output_path for chunking. This might be incorrect if simplification is required.")
                 input_dir_for_chunking = self.paths.parsed_output_path


        # Serialized tables dir is only relevant for annual reports
        serialized_tables_dir_param = None
        if self.run_config.processing_flow == "annual_report" and self.run_config.use_serialized_tables:
            # This implies tables are within the JSONs in parsed_output_path
            serialized_tables_dir_param = self.paths.parsed_output_path
        
        text_splitter.split_all_documents( # Renamed method in TextSplitter
            input_json_dir=input_dir_for_chunking,
            output_chunk_dir=self.paths.documents_dir, # This is `databases/chunked_{flow}`
            # tables_json_dir is for external table data, if TextSplitter handles it that way
            # For now, assuming tables are part of the main JSON if use_serialized_tables is true for reports
            tables_json_dir=serialized_tables_dir_param
        )
        _log.info(f"Chunked documents saved to {self.paths.documents_dir}")

    def create_vector_dbs(self):
        logging.basicConfig(level=logging.INFO)
        vdb_ingestor = VectorDBIngestor()
        # Input for ingestion is always self.paths.documents_dir (chunked JSONs)
        vdb_ingestor.process_reports(self.paths.documents_dir, self.paths.vector_db_dir)
        _log.info(f"Vector databases created in {self.paths.vector_db_dir}")
    
    def create_bm25_indices(self): # Renamed from create_bm25_db
        logging.basicConfig(level=logging.INFO)
        bm25_ingestor = BM25Ingestor()
        # Input is self.paths.documents_dir (chunked JSONs)
        # Output is self.paths.bm25_db_dir (a directory for .pkl files)
        bm25_ingestor.process_reports(self.paths.documents_dir, self.paths.bm25_db_dir)
        _log.info(f"BM25 indices created in {self.paths.bm25_db_dir}")
    
    def parse_pdfs(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10): # Renamed
        if parallel:
            self.parse_pdf_input_parallel(chunk_size=chunk_size, max_workers=max_workers)
        else:
            self.parse_pdf_input_sequential()
    
    def process_parsed_documents(self, include_vector_db: bool = True, include_bm25: bool = False): # Renamed
        """Process already parsed documents through the pipeline."""
        _log.info(f"Starting {self.run_config.processing_flow} documents processing pipeline...")
        
        if self.run_config.processing_flow == "annual_report":
            _log.info("Step 1: Simplifying parsed annual reports...")
            self.simplify_parsed_output() # Merging step for reports
            _log.info("Step 2: Exporting annual reports to markdown...")
            self.export_to_markdown() # Markdown export for reports
        # For podcasts, simplify_parsed_output is currently a conceptual no-op for file movement,
        # and markdown export is skipped.
        
        _log.info(f"Step {'3' if self.run_config.processing_flow == 'annual_report' else '1'}: Chunking documents...")
        self.chunk_documents()
        
        if include_vector_db:
            _log.info(f"Step {'4' if self.run_config.processing_flow == 'annual_report' else '2'}: Creating vector databases...")
            self.create_vector_dbs()
        
        if include_bm25:
            _log.info(f"Step {'5' if self.run_config.processing_flow == 'annual_report' else '3'}: Creating BM25 indices...")
            self.create_bm25_indices()
        
        _log.info(f"{self.run_config.processing_flow} documents processing pipeline completed successfully!")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        if not base_path.exists():
            return base_path
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        logging.basicConfig(level=logging.INFO)
        # Ensure QuestionsProcessor is initialized with the correct flow
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir, # This should be the chunked documents
            questions_file_path=self.paths.questions_file_path,
            processing_flow=self.run_config.processing_flow, # Pass the flow
            # subset_path is now handled inside QuestionsProcessor based on flow
            podcast_metadata_file= None, # Add path to podcast_metadata.json if created and used
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=str(output_path), # Ensure it's a string if required by underlying method
            submission_file=self.run_config.submission_file,
            team_email=self.run_config.team_email,
            submission_name=self.run_config.submission_name,
            pipeline_details=self.run_config.pipeline_details
        )
        _log.info(f"Answers saved to {output_path}")


# --- Predefined Configurations ---

# Configs for Annual Reports (original use case)
report_base_config = RunConfig(
    processing_flow="annual_report",
    parallel_requests=10,
    submission_name="AnnualReport RAG v.Base",
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_report_base",
    team_email="annual.reports@example.com"
)

report_max_nst_o3m_config = RunConfig(
    processing_flow="annual_report",
    use_serialized_tables=False, # No serialized tables
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=25,
    submission_name="AnnualReport RAG v.MaxNST-O3M",
    pipeline_details="Custom pdf parsing + vDB + Router + PDR + reranking + SO CoT; llm = o3-mini",
    answering_model="o3-mini-2025-01-31", # Placeholder if o3-mini is specific
    config_suffix="_report_max_nst_o3m",
    team_email="annual.reports@example.com"
)

# New Config for Podcasts
podcast_base_config = RunConfig(
    processing_flow="podcast",
    use_serialized_tables=False, # Not applicable to podcasts
    parent_document_retrieval=True, # Can be useful for podcasts (e.g. retrieve surrounding dialogue)
    llm_reranking=True, # Can be useful
    llm_reranking_sample_size=20, # Adjust as needed
    top_n_retrieval=10, # Adjust as needed
    parallel_requests=10,
    team_email="podcast.rag@example.com",
    submission_name="PodcastRAG v.Base",
    pipeline_details="Podcast PDF parsing + vDB + Router + PDR + reranking + SO CoT; llm = GPT-4o-mini",
    config_suffix="_podcast_base",
    pdf_input_dir_name_override="podcast_pdfs", # Specific to where podcast PDFs are
    questions_file_name_override="questions.json" # Assuming it's in the podcast data root
    # subset_name_override should be None for podcasts if not used
)


# Mapping of config names to config objects
configs = {
    "report_base": report_base_config,
    "report_max_nst_o3m": report_max_nst_o3m_config, # Best performing for reports
    "podcast_base": podcast_base_config
    # Add other original configs here, perhaps adapting them or keeping as is if they are report-specific
}
# Example of keeping some old configs (they will run as annual_report flow by default)
# configs["pdr"] = parent_document_retrieval_config
# ... and so on for other original configs if needed for comparison or legacy.


# You can run any method right from this file with:
# python -m src.pipeline
# Just uncomment the method you want to run.
# You can also change the run_config in the __main__ block.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # --- Example for running PODCAST pipeline ---
    podcast_data_root = here() / "data" / "podcast_set" # Ensure this path exists and has data
    # Ensure podcast_set has 'podcast_pdfs/' and 'questions.json'
    
    # Create dummy podcast PDFs and questions if they don't exist for a test run
    dummy_podcast_pdf_dir = podcast_data_root / "podcast_pdfs"
    dummy_podcast_pdf_dir.mkdir(parents=True, exist_ok=True)
    dummy_questions_path = podcast_data_root / "questions.json"

    if not list(dummy_podcast_pdf_dir.glob("*.pdf")):
        try:
            from reportlab.pdfgen import canvas
            for i in range(1, 3):
                dummy_pdf = dummy_podcast_pdf_dir / f"dummy_podcast_ep{i}.pdf"
                c = canvas.Canvas(str(dummy_pdf))
                c.drawString(72, 800, f"Episode {i}")
                c.drawString(72, 780, f"https://youtu.be/dummy{i}")
                c.drawString(72, 750, "[00:00:01.00] SpeakerA: Hello from dummy podcast episode {i}.")
                c.drawString(72, 730, "[00:00:05.00] SpeakerB: This is a test line for parsing.")
                c.save()
            _log.info(f"Created dummy podcast PDFs in {dummy_podcast_pdf_dir}")
        except ImportError:
            _log.error("ReportLab not needed to create dummy PDFs for podcast test run.")
        except Exception as e:
            _log.error(f"Could not create dummy podcast PDFs: {e}")

    if not dummy_questions_path.exists():
        dummy_q_data = [
            {"text": "What is Episode 1 about?", "kind": "text"},
            {"text": "Who spoke in Episode 2?", "kind": "names"}
        ]
        with open(dummy_questions_path, 'w') as f:
            json.dump(dummy_q_data, f, indent=2)
        _log.info(f"Created dummy questions file at {dummy_questions_path}")


    # Initialize pipeline with the podcast configuration
    podcast_pipeline = Pipeline(root_path=podcast_data_root, run_config=configs["podcast_base"])

    _log.info("Starting podcast pipeline run...")
    
    # 1. Download Docling models (if first time, or ensure they are present)
    # Pipeline.download_docling_models() # Static method call

    # 2. Parse Podcast PDFs (using the new JsonPodcastProcessor)
    # podcast_pipeline.parse_pdfs(parallel=False) # Sequential for easier debugging first
    
    # 3. Simplify/Prepare parsed output (for podcasts, this might be a no-op or minor transform)
    # podcast_pipeline.simplify_parsed_output()

    # 4. Chunk Documents (podcasts)
    # podcast_pipeline.chunk_documents()
    
    # 5. Create Vector Databases
    # podcast_pipeline.create_vector_dbs()
    
    # 6. Create BM25 Indices (Optional)
    # podcast_pipeline.create_bm25_indices()
    
    # 7. Process Questions
    # podcast_pipeline.process_questions()

    _log.info("Podcast pipeline run example finished (all steps commented out).")

    # --- Example for running original ANNUAL REPORT pipeline (using one of the old configs) ---
    # report_data_root = here() / "data" / "test_set" # Or erc2_set
    # report_run_config = configs.get("report_max_nst_o3m", RunConfig(processing_flow="annual_report")) # Fallback
    # report_pipeline = Pipeline(root_path=report_data_root, run_config=report_run_config)
    # _log.info("Starting annual report pipeline run example...")
    # report_pipeline.parse_pdfs(parallel=False)
    # report_pipeline.process_parsed_documents(include_bm25=True) # Runs simplify, chunk, vdb, bm25
    # report_pipeline.process_questions()
    # _log.info("Annual report pipeline run example finished (steps commented out).")