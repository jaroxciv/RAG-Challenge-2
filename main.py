import click
from pathlib import Path
from src.pipeline import Pipeline, configs, RunConfig # Removed preprocess_configs for now, can be re-added if needed for reports

@click.group()
def cli():
    """Pipeline command line interface for processing documents and questions."""
    pass

@cli.command()
def download_models():
    """Download required docling models."""
    click.echo("Downloading docling models...")
    Pipeline.download_docling_models() # Static method

@cli.command()
@click.option('--flow', type=click.Choice(['annual_report', 'podcast']), default='annual_report', help='Processing flow to use.')
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode.')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker (if parallel).')
@click.option('--max-workers', default=10, help='Number of parallel worker processes (if parallel).')
def parse_documents(flow, parallel, chunk_size, max_workers): # Renamed from parse_pdfs
    """Parse PDF documents (reports or podcasts) with optional parallel processing."""
    root_path = Path.cwd() # Assumes running from the specific data directory (e.g., data/test_set or data/podcast_set)

    # Create a minimal RunConfig for parsing, specifying the flow
    # Other RunConfig parameters are not strictly needed for parsing itself,
    # but Pipeline constructor expects a RunConfig.
    run_config_for_parsing = RunConfig(processing_flow=flow)
    if flow == "podcast":
        # Override default PDF input dir name for podcasts if not specified in a full config
        run_config_for_parsing.pdf_input_dir_name_override = "podcast_pdfs"
        run_config_for_parsing.questions_file_name_override = "questions.json" # For PipelineConfig pathing
        run_config_for_parsing.subset_name_override = None


    pipeline = Pipeline(root_path, run_config=run_config_for_parsing)
    
    click.echo(f"Parsing {flow} documents (parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers})...")
    pipeline.parse_pdfs(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers) # Method in Pipeline is parse_pdfs

@cli.command()
@click.option('--max-workers', default=10, help='Number of workers for table serialization.')
def serialize_tables(max_workers):
    """Serialize tables in parsed annual reports using parallel threading. Only for 'annual_report' flow."""
    root_path = Path.cwd()
    # This command is specific to annual_report flow, so RunConfig should reflect that.
    run_config_for_serialization = RunConfig(processing_flow="annual_report", use_serialized_tables=True)
    pipeline = Pipeline(root_path, run_config=run_config_for_serialization)
    
    click.echo(f"Serializing tables (max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(list(configs.keys())), default='report_base', help='Configuration preset to use from pipeline.py.')
@click.option('--include-bm25/--no-bm25', default=False, help="Whether to include BM25 index creation.")
def process_parsed(config, include_bm25): # Renamed from process_reports
    """Process already parsed documents through simplification (if any), chunking, and indexing stages."""
    root_path = Path.cwd()
    run_config = configs.get(config)
    if not run_config:
        click.echo(f"Error: Config '{config}' not found.", err=True)
        return

    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing parsed {run_config.processing_flow} documents (config={config}, include_bm25={include_bm25})...")
    # process_parsed_documents takes include_vector_db and include_bm25
    pipeline.process_parsed_documents(include_vector_db=True, include_bm25=include_bm25)

@cli.command()
@click.option('--config', type=click.Choice(list(configs.keys())), default='report_base', help='Configuration preset to use from pipeline.py.')
def process_questions(config):
    """Process questions using the RAG pipeline with the specified configuration."""
    root_path = Path.cwd() # Assumes running from the specific data directory

    run_config = configs.get(config)
    if not run_config:
        click.echo(f"Error: Config '{config}' not found.", err=True)
        return

    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing questions for {run_config.processing_flow} (config={config})...")
    pipeline.process_questions()

if __name__ == '__main__':
    cli()