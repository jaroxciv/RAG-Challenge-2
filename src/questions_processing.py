import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever # Assuming these will be adapted or are generic enough
from src.api_requests import APIProcessor
from tqdm import tqdm
import pandas as pd # May not be needed if subset_path is removed for podcasts
import threading
import concurrent.futures


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs', # Will point to podcast vector DBs
        documents_dir: Union[str, Path] = './documents', # Will point to processed podcast JSONs
        questions_file_path: Optional[Union[str, Path]] = None,
        processing_flow: str = "podcast", # Replaces new_challenge_pipeline, e.g., "podcast" or "annual_report"
        # subset_path: Optional[Union[str, Path]] = None, # Making this optional for podcasts
        podcast_metadata_file: Optional[Union[str, Path]] = None, # Example: for mapping episode names to IDs
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06",
        full_context: bool = False
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        # self.subset_path = Path(subset_path) if subset_path else None # Keep for now, but make usage conditional
        self.podcast_metadata_file = Path(podcast_metadata_file) if podcast_metadata_file else None
        self.processing_flow = processing_flow # "podcast" or "annual_report" (or other future flows)
        
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        self.answer_details = []
        # self.detail_counter = 0 # Not used
        self._lock = threading.Lock()

        if self.podcast_metadata_file and self.podcast_metadata_file.exists():
            # Example: Load podcast metadata if provided (e.g., episode title to ID mapping)
            # with open(self.podcast_metadata_file, 'r') as f:
            #     self.podcast_metadata = json.load(f)
            pass


    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """Format vector retrieval results into RAG context string."""
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            page_number = result.get('page', 'N/A') # podcast chunks might have page numbers
            text = result['text']
            # Include speaker and timestamp if available in the chunk from pdf_parsing
            speaker = result.get('speaker')
            timestamp = result.get('timestamp')
            prefix = ""
            if speaker and timestamp:
                prefix = f"From page {page_number}, at {timestamp}, {speaker} said: "
            elif speaker:
                prefix = f"From page {page_number}, {speaker} said: "
            elif page_number != 'N/A':
                prefix = f"Text retrieved from page {page_number}: "
            else:
                prefix = "Retrieved text: "

            context_parts.append(f'{prefix}\n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, podcast_identifier: str) -> list:
        """
        Create reference list for a podcast.
        `podcast_identifier` is the unique ID of the podcast document (e.g., filename stem).
        """
        refs = []
        for page in pages_list:
            # The 'podcast_id' should match the identifier used when saving the parsed JSON
            # and when creating vector DBs (e.g., 'episode_90').
            refs.append({"podcast_id": podcast_identifier, "page_index": page})
        return refs

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 1, max_pages: int = 8) -> list:
        """
        Validate that all page numbers mentioned in the LLM's answer are actually from the retrieval results.
        If fewer than min_pages valid references remain, add top pages from retrieval results.
        """
        if not claimed_pages: # Handles None or empty list
            claimed_pages = []
        
        # Ensure retrieval_results and its items are valid before list comprehension
        if not retrieval_results or not all(isinstance(r, dict) and 'page' in r for r in retrieval_results):
             _log.warning("Invalid retrieval_results for page validation.")
             retrieved_pages = []
        else:
            retrieved_pages = [result['page'] for result in retrieval_results]
        
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]
        
        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {list(removed_pages)}")
        
        # Ensure at least min_pages if possible
        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            for result in retrieval_results:
                page = result.get('page')
                if page is not None and page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    if len(validated_pages) >= min_pages:
                        break
        
        # Trim to max_pages
        if len(validated_pages) > max_pages:
            print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]
        
        return validated_pages

    def get_answer_for_podcast_entity(self, podcast_identifier: str, question: str, schema: str) -> dict:
        """
        Retrieves an answer for a question about a specific podcast entity (e.g., an episode).
        `podcast_identifier` is used to fetch the correct document context.
        """
        if self.llm_reranking:
            retriever = HybridRetriever( # Assuming retriever is adapted for podcast_identifier
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever( # Assuming retriever is adapted for podcast_identifier
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )

        if self.full_context:
            # `retrieve_all` in retriever needs to know how to get all chunks for `podcast_identifier`
            retrieval_results = retriever.retrieve_all(podcast_identifier)
        else:           
            # `retrieve_by_identifier` (new name proposal) in retriever
            retrieval_results = retriever.retrieve_by_identifier(
                identifier=podcast_identifier, # Changed from company_name
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages # This might mean full sentences or paragraphs for podcasts
            )
        
        if not retrieval_results:
            # Consider returning a specific dict or N/A instead of raising ValueError immediately
            # to allow graceful handling in _process_single_question
            return {"error": "No relevant context found", "final_answer": "N/A", "relevant_pages": [], "references": []}

        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data # Store last API response data

        # Reference handling for podcasts
        if self.processing_flow == "podcast": # or a similar flag indicating podcast mode
            pages = answer_dict.get("relevant_pages", [])
            validated_pages = self._validate_page_references(pages, retrieval_results)
            answer_dict["relevant_pages"] = validated_pages
            # `podcast_identifier` is the ID of the document being queried (e.g., "episode_90_transcript")
            answer_dict["references"] = self._extract_references(validated_pages, podcast_identifier)
        return answer_dict

    def _extract_podcast_entities_from_question(self, question_text: str) -> list[str]:
        """
        Extracts podcast entity identifiers (e.g., "Episode 101", "The Daily Show") from a question.
        This is a simplified version and can be made more robust.
        """
        # Attempt to find "Episode XXX" patterns
        episode_matches = re.findall(r"Episode\s+(\d+)", question_text, re.IGNORECASE)
        entities = [f"Episode {num}" for num in episode_matches]
        
        # Attempt to find quoted strings as potential podcast titles or series names
        quoted_matches = re.findall(r'"([^"]*)"', question_text)
        for qm in quoted_matches:
            # Avoid adding if it's just the episode number part already captured
            if not any(qm.endswith(ep.split()[-1]) for ep in entities if "Episode" in ep):
                entities.append(qm)
        
        # If we have podcast metadata, try to match known titles/series
        # if hasattr(self, 'podcast_metadata') and self.podcast_metadata:
        #     for title_or_id in self.podcast_metadata.keys(): # or iterate through a list of known entities
        #         if title_or_id.lower() in question_text.lower() and title_or_id not in entities:
        #             entities.append(title_or_id)

        # Basic fallback: if no specific entities found, maybe the question is general
        # or implies a single known podcast if only one is loaded.
        # For now, we require some identifiable entity.
        
        # Remove duplicates if any were added by different methods
        unique_entities = []
        for entity in entities:
            if entity not in unique_entities:
                unique_entities.append(entity)
        
        return unique_entities

    def process_question(self, question_text: str, schema: str): # Renamed question to question_text for clarity
        if self.processing_flow == "podcast":
            extracted_entities = self._extract_podcast_entities_from_question(question_text)
        # elif self.processing_flow == "annual_report" and self.subset_path: # Example for old flow
            # extracted_entities = self._extract_companies_from_subset(question_text)
        else: # Default to quoted string extraction if no specific flow or method
            extracted_entities = re.findall(r'"([^"]*)"', question_text)

        if not extracted_entities:
            # If no entities are extracted, and the question might be general (e.g. "Any podcasts about AI?")
            # We might need a way to query across all documents or a default document.
            # For now, let's assume questions target specific entities or will be handled by a general search.
            # This behavior might need refinement based on how general queries are intended to work.
            # If a single podcast is the context, we can use its ID.
            # For now, let's raise an error or return N/A if no entity is clearly targeted.
             _log.warning(f"No specific podcast entity found in question: '{question_text}'. Further logic needed for general queries.")
             # Fallback: if there's only one document in documents_dir, assume it's the target.
             # This is a simplistic assumption for single-document context.
             # For now, let's return an error state that can be handled.
             return {"error": "No specific podcast entity identified in the question.", "final_answer": "N/A"}


        if len(extracted_entities) == 1:
            entity_identifier = extracted_entities[0]
            # We need to map this extracted entity (e.g., "Episode 102") to the actual document ID
            # used by the retriever (e.g., "episode_102_transcript.json").
            # This mapping might involve checking self.documents_dir or using self.podcast_metadata
            # For simplicity, let's assume the extracted_identifier is directly usable or can be transformed.
            # TODO: Implement robust mapping from extracted entity to document ID.
            # For now, assume entity_identifier can be used if it matches a file stem.

            # Simplified: try to match entity_identifier (e.g., "Episode 102") to a file stem.
            # A better approach would be a metadata lookup.
            doc_id_to_use = None
            potential_id_transformed = entity_identifier.lower().replace(" ", "_") # e.g. "episode_102"

            # Check if a JSON file matching this transformed ID exists in documents_dir
            # This assumes processed JSONs are named like 'episode_102.json'
            # and vector DBs are also named similarly.
            # This part is crucial and needs to align with how ingestion names files/DBs.

            # Iterate through document directory to find a match (this is inefficient for many docs)
            # A direct lookup or naming convention is better.
            # For now, let's assume the retriever can handle identifiers like "Episode 102"
            # or that the identifier matches a filename stem.
            # The retriever's `retrieve_by_identifier` will need to resolve this.
            doc_id_to_use = entity_identifier # Pass it to the retriever to handle.

            answer_dict = self.get_answer_for_podcast_entity(doc_id_to_use, question_text, schema)
            return answer_dict
        else: # Comparative question
            # For comparative, the extracted_entities should be usable by process_comparative_question
            return self.process_comparative_question(question_text, extracted_entities, schema)
    
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
    