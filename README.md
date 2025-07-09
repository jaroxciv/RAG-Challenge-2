# Enhanced RAG System (Formerly RAG Challenge Winner Solution)

**Original Project Links (Annual Report Focus):**
- Russian: https://habr.com/ru/articles/893356/
- English: https://abdullin.com/ilya/how-to-build-best-rag/

This repository originally contained the winning solution for a RAG Challenge focused on company annual reports. It has now been adapted and enhanced to also process and answer questions about **podcast transcripts** (provided in PDF format). The system uses a combination of:

- Custom PDF parsing (adapted for both annual reports and podcast transcript structures)
- Vector search with parent document retrieval
- LLM reranking for improved context relevance
- Structured output prompting with chain-of-thought reasoning
- Query routing for multi-entity comparisons (e.g., comparing companies or podcast episodes)

## Disclaimer

This is research/competition-style code - it's scrappy but demonstrates various RAG techniques. Some notes before you dive in:

- IBM Watson integration (from original challenge) won't work.
- The code might have rough edges and workarounds.
- Testing is primarily focused on core functionality; error handling might be minimal.
- You'll need your own API keys for OpenAI/Gemini (or other configured LLM providers).
- GPU can significantly speed up PDF parsing with Docling.

If you're looking for production-ready code, this isn't it. But if you want to explore different RAG techniques and their implementations for various document types, this is a good starting point!

## Quick Start

Clone and setup:
```bash
git clone https://github.com/IlyaRice/RAG-Challenge-2.git # Or your forked repo URL
cd RAG-Challenge-2
python -m venv venv
# Activate virtual environment
# Windows (PowerShell): .\venv\Scripts\Activate.ps1
# Windows (cmd.exe): .\venv\Scripts\activate.bat
# Linux/macOS: source venv/bin/activate
pip install -e . -r requirements.txt
```

Rename `env` to `.env` and add your API keys.

## Datasets

The repository can work with different types of datasets:

1.  **Annual Reports (Original Use Case)**:
    *   A small test set (in `data/test_set/`) with 5 annual reports and questions.
    *   The full ERC2 competition dataset (in `data/erc2_set/`) with competition questions and reports.
    *   These datasets are structured for company annual report analysis. See their respective READMEs for details.

2.  **Podcast Transcripts (New Use Case)**:
    *   A new dataset structure is intended for podcast transcripts. Place your podcast PDFs in a directory like `data/podcast_set/podcast_pdfs/`.
    *   Create a corresponding questions file, e.g., `data/podcast_set/questions.json`, with questions relevant to your podcasts.
    *   The system expects podcast PDFs to have a semi-structured format (Episode number, YouTube link, then lines like `[timestamp] Speaker: Text`).

You can use these datasets to:
- Study example questions and system outputs for different document types.
- Run the pipeline from scratch using your own PDFs.
- Use pre-processed data (if available) to skip to specific pipeline stages.

## Usage

The primary way to run the pipeline is by executing `src/pipeline.py` directly or using the CLI provided by `main.py`.

### Running via `src/pipeline.py`

You can uncomment and run specific methods within the `if __name__ == "__main__":` block of `src/pipeline.py`. This file now includes an example setup for running the **podcast pipeline**:
```python
# In src/pipeline.py
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # --- Example for running PODCAST pipeline ---
    podcast_data_root = here() / "data" / "podcast_set"
    # Ensure podcast_data_root exists and contains:
    # - podcast_data_root / "podcast_pdfs" / (your_podcast_files.pdf)
    # - podcast_data_root / "questions.json"

    podcast_pipeline = Pipeline(root_path=podcast_data_root, run_config=configs["podcast_base"])

    # Uncomment steps to run:
    # Pipeline.download_docling_models()
    # podcast_pipeline.parse_pdfs(parallel=False)
    # podcast_pipeline.simplify_parsed_output() # Adapts parsed output for chunking
    # podcast_pipeline.chunk_documents()
    # podcast_pipeline.create_vector_dbs()
    # podcast_pipeline.create_bm25_indices() # Optional
    # podcast_pipeline.process_questions()
```
Execute this via:
```bash
python -m src.pipeline
```
(Ensure your terminal is in the root directory of the project).

### Running via `main.py` (CLI)

The CLI (`main.py`) should be run from the specific data directory you intend to process (e.g., `cd data/podcast_set/`).

Get help on available commands:
```bash
python ../../main.py --help
# (Adjust path to main.py based on your CWD)
```

Available commands now include:
- `download-models`: Download required Docling models.
- `parse-documents`: Parse PDF documents. Requires a `--flow` option.
  ```bash
  # Example for podcasts, run from data/podcast_set/
  python ../../main.py parse-documents --flow podcast --parallel
  ```
- `serialize-tables`: Process tables (specific to 'annual_report' flow).
- `process_parsed`: Process parsed documents (chunking, indexing). Requires `--config`.
  ```bash
  # Example for podcasts, run from data/podcast_set/
  python ../../main.py process_parsed --config podcast_base --include-bm25
  ```
- `process_questions`: Process questions using a specified config.
  ```bash
  # Example for podcasts, run from data/podcast_set/
  python ../../main.py process_questions --config podcast_base
  ```

Each command has its own options. For example:
```bash
python ../../main.py parse-documents --help
# Shows options like --flow, --parallel/--sequential, --chunk-size, --max-workers
```

## Configurations (`src/pipeline.py`)

The `src/pipeline.py` file defines various `RunConfig` objects that control the pipeline's behavior. Key aspects:
- **`processing_flow`**: A new attribute in `RunConfig` (e.g., `"annual_report"` or `"podcast"`) dictates how data is handled.
- **Report Configs**: Original configurations like `report_max_nst_o3m` are still available for annual report processing (now often prefixed with `report_`).
- **Podcast Config**: A new `podcast_base_config` is introduced for the podcast workflow.

Example configurations in `src/pipeline.py`:
- `report_base_config`: A basic configuration for annual reports.
- `report_max_nst_o3m`: The best performing original config for annual reports.
- `podcast_base_config`: A base configuration for processing podcast transcripts.

Check `src/pipeline.py` for more details on configurations and how to customize them.

## License

MIT