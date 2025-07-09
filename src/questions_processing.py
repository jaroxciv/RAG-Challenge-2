import json
from typing import Union, Dict, List, Optional, Any
import re
from pathlib import Path
# Corrected import: HybridRetriever now likely lives in retrieval.py, not a submodule.
from src.retrieval import HybridRetriever # VectorRetriever might be used by HybridRetriever internally
from src.api_requests import APIProcessor
from tqdm import tqdm
# import pandas as pd # No longer needed for podcast flow
import threading
import concurrent.futures
import logging # Make sure logging is imported

_log = logging.getLogger(__name__) # Initialize logger for this module


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path], # Made required
        documents_dir: Union[str, Path], # Made required
        bm25_db_dir: Optional[Union[str, Path]] = None, # For HybridRetriever
        questions_file_path: Optional[Union[str, Path]] = None,
        processing_flow: str = "podcast",
        podcast_metadata_file: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        llm_reranking: bool = True, # Defaulting to True for HybridRetriever
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 5, # Top N after reranking or direct retrieval
        top_n_per_doc_for_cross_search: int = 3, # For search_across_all_podcasts
        parallel_requests: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06",
        full_context: bool = False # For get_answer_for_podcast_entity
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.bm25_db_dir = Path(bm25_db_dir) if bm25_db_dir else None
        self.podcast_metadata_file = Path(podcast_metadata_file) if podcast_metadata_file else None
        self.processing_flow = processing_flow
        
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking # This flag will control HybridRetriever's behavior
        self.llm_reranking_sample_size = llm_reranking_sample_size # For HybridRetriever single doc
        self.top_n_retrieval = top_n_retrieval # Final N for single doc or cross-search
        self.top_n_per_doc_for_cross_search = top_n_per_doc_for_cross_search

        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        # Initialize retriever - assuming HybridRetriever is the main one
        # HybridRetriever can decide internally whether to use BM25 or just Vector based on its config/params
        self.retriever = HybridRetriever(
            vector_db_dir=self.vector_db_dir,
            documents_dir=self.documents_dir,
            bm25_db_dir=self.bm25_db_dir # Pass BM25 dir
        )

        self.answer_details = []
        self._lock = threading.Lock()

        if self.podcast_metadata_file and self.podcast_metadata_file.exists():
            try:
                with open(self.podcast_metadata_file, 'r', encoding='utf-8') as f:
                    self.podcast_metadata = json.load(f) # E.g. {"episode_101_official_title": "episode_101_doc_id", ...}
            except Exception as e:
                _log.error(f"Failed to load podcast metadata from {self.podcast_metadata_file}: {e}")
                self.podcast_metadata = None


    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            _log.warning("No questions file path provided.")
            return []
        try:
            with open(questions_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            _log.error(f"Questions file not found: {questions_file_path}")
            return []
        except json.JSONDecodeError as e:
            _log.error(f"Error decoding JSON from questions file {questions_file_path}: {e}")
            return []


    def _format_retrieval_results(self, retrieval_results: List[Dict]) -> str:
        """Format retrieval results (list of chunk dicts) into RAG context string."""
        if not retrieval_results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(retrieval_results):
            text = result.get('text', '')
            # Metadata from our adapted retrievers
            source_id = result.get('podcast_id', result.get('document_id', 'UnknownSource'))
            page_number = result.get('page', result.get('original_page', 'N/A'))
            speaker = result.get('speaker', result.get('original_speaker'))
            timestamp = result.get('timestamp', result.get('original_timestamp'))

            prefix = f"Context Segment {i+1} (Source: {source_id}, Page: {page_number}"
            if speaker:
                prefix += f", Speaker: {speaker}"
            if timestamp:
                prefix += f", Timestamp: {timestamp}"
            prefix += "):"

            context_parts.append(f'{prefix}\n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _extract_references_from_results(self, llm_pages_list: List[Any], retrieval_results: List[Dict], default_identifier: Optional[str] = None) -> List[Dict]:
        """
        Create reference list based on LLM's claimed pages and actual retrieved results.
        Handles cases where llm_pages_list might contain simple page numbers (for single doc query)
        or more complex structures if LLM indicates source_ids.
        """
        refs = []
        # If LLM gives simple page numbers, assume they relate to `default_identifier` or the main doc queried.
        # If LLM gives dicts like {'podcast_id': 'id', 'page': X}, use that.
        
        # For now, a simplified approach: extract unique (podcast_id, page) from retrieval_results
        # that correspond to pages mentioned by LLM or are generally top retrieved pages.
        # This needs to be more robust based on how LLM `relevant_pages` is structured.
        
        # Let's assume llm_pages_list contains page numbers relevant to the primary context.
        # If retrieval_results came from multiple podcast_ids (cross-search), this is more complex.
        
        # Simplification: If default_identifier is given (single-doc query), use that.
        if default_identifier and isinstance(llm_pages_list, list) and all(isinstance(p, int) for p in llm_pages_list):
            for page_num in llm_pages_list:
                refs.append({"podcast_id": default_identifier, "page_index": page_num})
            return refs

        # If general query, try to get (podcast_id, page) from the top retrieved chunks
        # that the LLM *might* have used. LLM's `relevant_pages` for multi-source is hard.
        # For now, just list references from the top N actually retrieved chunks.
        # This is a placeholder for better reference extraction from multi-source LLM answers.
        added_refs = set()
        for res_chunk in retrieval_results[:5]: # Take top 5 retrieved as potential references
            pid = res_chunk.get('podcast_id', default_identifier)
            pnum = res_chunk.get('page', res_chunk.get('original_page'))
            if pid and pnum is not None:
                ref_tuple = (pid, pnum)
                if ref_tuple not in added_refs:
                    refs.append({"podcast_id": pid, "page_index": pnum})
                    added_refs.add(ref_tuple)
        return refs


    def get_answer_for_podcast_entity(self, podcast_identifier: str, question: str, schema: str) -> dict:
        """
        Retrieves an answer for a question about a specific podcast entity.
        """
        # self.retriever is HybridRetriever instance
        retrieval_results = []
        if self.full_context:
            retrieval_results = self.retriever.vector_retriever.retrieve_all(document_id=podcast_identifier)
        else:           
            retrieval_results = self.retriever.retrieve_by_document_id(
                document_id=podcast_identifier,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size, # Used by Hybrid for initial pool
                top_n=self.top_n_retrieval, # Final N from Hybrid
                return_parent_pages=self.return_parent_pages,
                # HybridRetriever decides internally if it uses BM25 based on its own state
            )
        
        if not retrieval_results:
            return {"error": f"No relevant context found for '{podcast_identifier}'", "final_answer": "N/A", "relevant_pages": [], "references": []}

        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data

        if self.processing_flow == "podcast":
            llm_pages = answer_dict.get("relevant_pages", [])
            # Validate pages against those in retrieval_results for this specific podcast_identifier
            # This validation might need access to the specific pages retrieved for this entity.
            # For now, _validate_page_references might be too generic if llm_pages are just numbers.
            # validated_pages = self._validate_page_references(llm_pages, retrieval_results)
            # answer_dict["relevant_pages"] = validated_pages
            # References are for this specific podcast_identifier
            answer_dict["references"] = self._extract_references_from_results(llm_pages, retrieval_results, podcast_identifier)
        return answer_dict

    def _extract_podcast_entities_from_question(self, question_text: str) -> list[str]:
        # This logic remains largely the same as before.
        # It tries to find "Episode XXX" or quoted strings.
        # If self.podcast_metadata exists, it could be used to map friendly names to doc_ids.
        episode_matches = re.findall(r"Episode\s+(\d+)", question_text, re.IGNORECASE)
        # Attempt to map "Episode XXX" to a known document ID format if necessary
        # For now, assume "Episode XXX" might be a direct or transformable ID.
        # Example: if doc IDs are "episode_XXX", then transform.
        # This part needs to align with how document IDs are stored and expected by retriever.
        # Let's assume for now the extracted entities are usable as document_ids or can be mapped.

        entities = [f"Episode {num}" for num in episode_matches] # Or transform to "episode_{num}"
        
        quoted_matches = re.findall(r'"([^"]*)"', question_text)
        for qm in quoted_matches:
            # Avoid adding if it's just the episode number part already captured
            # or if it's a very short, non-descriptive quote.
            is_part_of_episode_match = any(qm.lower().endswith(ep.lower().split()[-1]) for ep in entities if "Episode" in ep)
            if not is_part_of_episode_match and len(qm) > 3: # Basic filter for short quotes
                entities.append(qm)
        
        # If using podcast_metadata for mapping friendly names to actual document_ids:
        if hasattr(self, 'podcast_metadata') and self.podcast_metadata:
            # This metadata would be like: {"My Cool Podcast Episode Title": "podcast_file_stem_id_123"}
            # Or {"episode_101": "ep_101_doc_id"}
            # This step is crucial for robust entity linking.
            # For now, this is a placeholder for a more complex entity linking step.
            pass

        unique_entities = []
        [unique_entities.append(x) for x in entities if x not in unique_entities]
        return unique_entities


    def process_question(self, question_text: str, schema: str):
        extracted_entities = []
        is_general_query = False

        if self.processing_flow == "podcast":
            extracted_entities = self._extract_podcast_entities_from_question(question_text)
            # Determine if it's a general query
            # Simple heuristic: if no specific entities found, or question contains keywords like "any podcasts", "which episodes"
            if not extracted_entities or \
               any(kw in question_text.lower() for kw in ["any podcast", "which episode", "suggest podcast"]):
                is_general_query = True
                _log.info(f"Treating as general query: {question_text}")
        else: # Fallback for other flows or if entity extraction is different
            extracted_entities = re.findall(r'"([^"]*)"', question_text)
            if not extracted_entities: # If still no entities, could be general for other flows too
                is_general_query = True # Or handle based on flow-specific rules

        if is_general_query and self.processing_flow == "podcast":
            # Handle general query across all podcasts
            retrieval_results = self.retriever.search_across_all_podcasts(
                query=question_text,
                top_n_overall=self.top_n_retrieval, # Use top_n_retrieval for final output count
                top_n_per_doc=self.top_n_per_doc_for_cross_search,
                rerank_with_llm=self.llm_reranking
            )
            if not retrieval_results:
                return {"error": "No relevant context found across any podcasts.", "final_answer": "N/A", "relevant_pages": [], "references": []}

            rag_context = self._format_retrieval_results(retrieval_results)
            # Schema for general queries might be "text" or a specific "suggestion" schema
            # For now, use the provided schema, but this might need a dedicated "suggestion" schema/prompt
            answer_dict = self.openai_processor.get_answer_from_rag_context(
                question=question_text,
                rag_context=rag_context,
                schema=schema, # Or a specific schema like "podcast_suggestion_list"
                model=self.answering_model
            )
            self.response_data = self.openai_processor.response_data
            # References for general queries:
            llm_pages = answer_dict.get("relevant_pages", []) # This is tricky for multi-source
            answer_dict["references"] = self._extract_references_from_results(llm_pages, retrieval_results)
            answer_dict["retrieved_chunks_for_general_query"] = retrieval_results # For debugging/inspection
            return answer_dict

        elif len(extracted_entities) == 1:
            # This part requires mapping the extracted entity name (e.g., "Episode 101")
            # to the actual document ID used by the retriever (e.g., "episode_101_filename_stem").
            # This is a CRITICAL step. For now, assume direct use or simple transformation.
            entity_identifier = extracted_entities[0]
            # TODO: Implement robust mapping from extracted_entity_name to actual_document_id
            # e.g., using self.podcast_metadata or by scanning document filenames.
            # Simple placeholder: transform "Episode 101" to "episode_101" if that's the doc ID pattern.
            # This needs to be consistent with how `podcast_id` is set in ingestion and `TextSplitter`.
            # Let's assume the entity_identifier is directly usable for now.
            actual_doc_id = entity_identifier # This is a placeholder for a real mapping

            return self.get_answer_for_podcast_entity(actual_doc_id, question_text, schema)

        elif len(extracted_entities) > 1: # Comparative question
            return self.process_comparative_question(question_text, extracted_entities, schema)

        else: # Should have been caught by is_general_query or entity extraction
             _log.error(f"Unhandled case in process_question for: '{question_text}'")
             return {"error": "Could not determine how to process the question.", "final_answer": "N/A"}

    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """Create a reference ID for answer details and store the details."""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            # Ensure all expected keys exist in answer_dict before accessing them
            step_by_step = answer_dict.get('step_by_step_analysis', 'N/A')
            reasoning_summary = answer_dict.get('reasoning_summary', 'N/A')
            relevant_pages = answer_dict.get('relevant_pages', [])

            self.answer_details[question_index] = {
                "step_by_step_analysis": step_by_step,
                "reasoning_summary": reasoning_summary,
                "relevant_pages": relevant_pages,
                "response_data": getattr(self, 'response_data', None), # Safely access response_data
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """Calculate statistics about processed questions."""
        total_questions = len(processed_questions)
        if total_questions == 0: # Avoid division by zero
            return {"total_questions": 0, "error_count": 0, "na_count": 0, "success_count": 0}

        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count

        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = "") -> dict:
        total_questions = len(questions_list)
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions
        processed_questions = []
        parallel_threads = self.parallel_requests

        desc = "Processing podcast questions" if self.processing_flow == "podcast" else "Processing questions"

        if parallel_threads <= 1:
            for question_data in tqdm(questions_with_index, desc=desc):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path: # Save progress incrementally
                    self._save_progress(processed_questions, output_path, submission_file, team_email, submission_name, pipeline_details)
        else:
            with tqdm(total=total_questions, desc=desc) as pbar:
                # Process in batches to allow for incremental saving with ThreadPoolExecutor
                for i in range(0, total_questions, parallel_threads * 5): # Process larger batches before saving
                    batch_to_process = questions_with_index[i : i + parallel_threads * 5]
                    
                    current_batch_results = []
                    for j in range(0, len(batch_to_process), parallel_threads):
                        sub_batch = batch_to_process[j: j + parallel_threads]
                        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                            batch_results_segment = list(executor.map(self._process_single_question, sub_batch))
                        current_batch_results.extend(batch_results_segment)
                        pbar.update(len(batch_results_segment))

                    processed_questions.extend(current_batch_results)
                    if output_path: # Save progress after a larger batch
                        self._save_progress(processed_questions, output_path, submission_file, team_email, submission_name, pipeline_details)
        
        statistics = self._calculate_statistics(processed_questions, print_stats=True)
        
        # Final save after all processing is done
        if output_path:
            self._save_progress(processed_questions, output_path, submission_file, team_email, submission_name, pipeline_details)

        return {
            "questions": processed_questions, # Contains all processed questions
            "answer_details": self.answer_details, # Contains details for all questions
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0) # Should always exist
        
        # Determine field names based on processing_flow or a unified input format
        question_text = question_data.get("text") or question_data.get("question")
        schema = question_data.get("kind") or question_data.get("schema") # 'kind' for new, 'schema' for old

        try:
            answer_dict = self.process_question(question_text, schema)
            
            # Ensure answer_dict is a dict, especially if process_question can return None or error string
            if not isinstance(answer_dict, dict):
                # This case should ideally not happen if process_question always returns a dict
                _log.error(f"answer_dict is not a dict for question: {question_text}. Got: {answer_dict}")
                # Create a standard error structure
                answer_dict = {"error": "Internal processing error: answer_dict not a dict.", "final_answer": "N/A"}

            if "error" in answer_dict: # Handles errors from process_question or get_answer_for_podcast_entity
                # Create detail_ref even for errors, possibly with error info
                # Ensure `response_data` is not expected in error cases for `_create_answer_detail_ref`
                error_detail_payload = {
                    "step_by_step_analysis": answer_dict.get("error"),
                    "reasoning_summary": "Error occurred during processing.",
                    "relevant_pages": []
                }
                detail_ref = self._create_answer_detail_ref(error_detail_payload, question_index)

                # Unified error return structure
                return {
                    "question_text": question_text,
                    "kind": schema, # Use 'kind' as the unified field
                    "value": "N/A", # Default value for errors
                    "references": [],
                    "error": answer_dict["error"],
                    "answer_details": {"$ref": detail_ref}
                }

            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            # Unified success return structure
            return {
                "question_text": question_text,
                "kind": schema,
                "value": answer_dict.get("final_answer", "N/A"),
                "references": answer_dict.get("references", []),
                "answer_details": {"$ref": detail_ref}
            }
        except Exception as err: # Catch any other unexpected errors
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        import traceback
        error_message = str(err)
        tb_str = traceback.format_exc() # Renamed tb to tb_str to avoid conflict

        _log.error(f"Error processing question: {question_text}\nType: {type(err).__name__}\nMessage: {error_message}\nTraceback:\n{tb_str}")

        error_ref = f"#/answer_details/{question_index}"
        error_detail_payload = { # Consistent payload for _create_answer_detail_ref
            "step_by_step_analysis": f"Error: {error_message}\nTraceback: {tb_str}",
            "reasoning_summary": "An unexpected error occurred.",
            "relevant_pages": []
            # No self.response_data here as it might be from a previous successful call or irrelevant
        }
        # Create the detail ref, but be careful about what's in self.response_data
        # It might be better to not include self.response_data in error_detail_payload for _create_answer_detail_ref
        # or ensure _create_answer_detail_ref can handle its absence.
        # For now, _create_answer_detail_ref accesses self.response_data via getattr.
        
        # Store detailed error info in answer_details
        with self._lock:
            self.answer_details[question_index] = {
                "error_type": type(err).__name__,
                "error_message": error_message,
                "error_traceback": tb_str, # Store full traceback here for debugging
                "self": error_ref
            }

        # Unified error return structure
        return {
            "question_text": question_text,
            "kind": schema,
            "value": "N/A", # Default error value
            "references": [],
            "error": f"{type(err).__name__}: {error_message}", # User-facing error
            "answer_details": {"$ref": error_ref} # Reference to more detailed error info
        }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        submission_answers = []
        for q_idx, q_data in enumerate(processed_questions):
            question_text = q_data.get("question_text") # Standardized field
            kind = q_data.get("kind") # Standardized field
            value = "N/A" if "error" in q_data else q_data.get("value", "N/A")
            references = q_data.get("references", [])
            
            step_by_step_analysis = None
            # Access answer_details using q_idx, assuming it's correctly populated
            if q_idx < len(self.answer_details) and self.answer_details[q_idx]:
                step_by_step_analysis = self.answer_details[q_idx].get("step_by_step_analysis")

            if value == "N/A":
                references = []
            else:
                # Adapt reference structure based on processing_flow
                if self.processing_flow == "podcast":
                    # References for podcasts: {"podcast_id": ..., "page_index": ...}
                    # Page index conversion (0-based vs 1-based) should be handled consistently.
                    # Assuming 'page_index' from LLM is 1-based for user readability, convert to 0-based if submission needs it.
                    # For now, assume page_index is as-is from _extract_references.
                    # If submission requires 0-based, and LLM gives 1-based, subtract 1.
                    # Let's assume our internal `page` is 1-based from parsing/LLM.
                    # The example showed "pdf_sha1", we need "podcast_id".
                    references = [
                        {"podcast_id": ref.get("podcast_id"), "page_index": ref.get("page_index") -1 if isinstance(ref.get("page_index"), int) else ref.get("page_index")}
                        for ref in references if ref.get("podcast_id") is not None
                    ]
                # else: # Original logic for annual reports with pdf_sha1
                #     references = [
                #         {"pdf_sha1": ref["pdf_sha1"], "page_index": ref["page_index"] - 1}
                #         for ref in references
                #     ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "team_email": team_email,
                    "submission_name": submission_name,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', team_email: str = "79250515615@yandex.com", submission_name: str = "Ilia_Ris SO CoT + Parent Document Retrieval", submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        """
        Process a question involving multiple companies in parallel:
        1. Rephrase the comparative question into individual questions
        2. Process each individual question using parallel threads
        3. Combine results into final comparative answer
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=companies
        )
        
        individual_answers = {}
        aggregated_references = []
        
        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")
            
            answer_dict = self.get_answer_for_company(
                company_name=company, 
                question=sub_question, 
                schema="number"
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company 
                for company in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict
                    
                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise
        
        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        comparative_answer["references"] = aggregated_references
        return comparative_answer
    