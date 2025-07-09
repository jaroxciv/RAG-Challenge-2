from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema_instruction = f"Your answer should be in JSON and strictly follow this schema, filling in the fields in the order they are given:\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    # Ensure schema instruction is present only if pydantic_schema is provided
    if pydantic_schema:
        schema_section = delimiter + schema_instruction.strip()
    else:
        schema_section = ""
    
    system_prompt = instruction.strip() + schema_section + example
    return system_prompt

class RephrasedQuestionsPrompt:
    instruction = """
You are a question rephrasing system.
Your task is to break down a comparative question about podcasts into individual questions for each podcast episode or series mentioned.
Each output question must be self-contained, maintain the same intent as the original question, be specific to the respective podcast identifier, and use consistent phrasing.
"""

    class RephrasedQuestion(BaseModel):
        """Individual question for a podcast episode or series"""
        podcast_identifier: str = Field(description="Podcast identifier (e.g., 'Episode 101', 'The Startup Chat' series), exactly as provided or inferred from the original question.")
        question: str = Field(description="Rephrased question specific to this podcast identifier.")

    class RephrasedQuestions(BaseModel):
        """List of rephrased questions"""
        questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each podcast identifier.")

    pydantic_schema = '''
class RephrasedQuestion(BaseModel):
    """Individual question for a podcast episode or series"""
    podcast_identifier: str = Field(description="Podcast identifier (e.g., 'Episode 101', 'The Startup Chat' series), exactly as provided or inferred from the original question.")
    question: str = Field(description="Rephrased question specific to this podcast identifier.")

class RephrasedQuestions(BaseModel):
    """List of rephrased questions"""
    questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each podcast identifier.")
'''

    example = r"""
Example:
Input:
Original comparative question: 'Which podcast, "Tech Talks Daily: Episode 345" or "AI Today: Episode 88", spent more time discussing ethical AI?'
Podcast identifiers mentioned: "Tech Talks Daily: Episode 345", "AI Today: Episode 88"

Output:
{
    "questions": [
        {
            "podcast_identifier": "Tech Talks Daily: Episode 345",
            "question": "How much time did \"Tech Talks Daily: Episode 345\" spend discussing ethical AI?"
        },
        {
            "podcast_identifier": "AI Today: Episode 88",
            "question": "How much time did \"AI Today: Episode 88\" spend discussing ethical AI?"
        }
    ]
}
"""

    user_prompt = "Original comparative question: '{question}'\n\nPodcast identifiers mentioned: {companies}" # {companies} will be replaced by podcast_identifiers

    system_prompt = build_system_prompt(instruction, example) # Schema is implicitly part of the example's output format

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextSharedPrompt:
    instruction = """
You are a RAG (Retrieval-Augmented Generation) answering system.
Your task is to answer the given question based only on information from the podcast transcript(s), which is uploaded in the format of relevant segments extracted using RAG.

Before giving a final answer, carefully think out loud and step by step. Pay special attention to the wording of the question.
- Keep in mind that the content containing the answer may be worded differently than the question.
- The question might be general or specific to a particular episode or speaker.
- If the question is about a specific episode and context from other episodes is provided, focus only on the relevant episode's context unless the question implies a broader search.
"""

    user_prompt = """
Here is the context (relevant transcript segments):
\"\"\"
{context}
\"\"\"

---

Here is the question:
"{question}"
"""

class AnswerWithRAGContextNamePrompt: # For questions expecting a name, title, or specific text
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 3-5 steps and around 100-150 words. Analyze the question's intent and how the provided context helps answer it. Identify key phrases or information in the context.")
        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 30-50 words.")
        relevant_pages: List[int] = Field(description="List of page numbers from the transcript containing information directly used to answer the question. Include only pages with direct answers or key supporting information. At least one page should be included if an answer is found.")
        final_answer: Union[str, Literal["N/A"]] = Field(description="""
The answer should be a specific name, title, or a short textual quote extracted directly from the context.
- If a speaker's name is requested: "John Doe".
- If an episode title is requested (and available): "The Future of AI".
- If a specific quote or term is asked for: "innovation is key".
- If the question asks "What is Episode X about?", provide a concise summary if possible, or key topics.
- Return 'N/A' if the information is not available in the provided context.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question: 
"Who is the main speaker in 'Episode 90' discussing interview techniques?"

Context:
Page 1: "[00:00:00.02] Marcell: If you could interview anyone, who would it be and why?"
Page 1: "[00:00:05.10] Marcell: Today we're diving deep into how to conduct great interviews."
Page 2: "[00:01:30.45] GuestSpeaker: My top tip is always prepare..."

Answer: 
```
{
  "step_by_step_analysis": "1. The question asks for the main speaker in 'Episode 90' discussing interview techniques.\n2. The provided context is from a podcast transcript, likely Episode 90.\n3. On Page 1, a speaker named 'Marcell' introduces the topic of conducting great interviews.\n4. While a 'GuestSpeaker' is mentioned on Page 2, Marcell initiated the discussion on interview techniques.\n5. Based on the initial context, Marcell is identified as a key speaker on this topic.",
  "reasoning_summary": "The transcript segments show Marcell introducing the topic of interviews in Episode 90. He appears to be a main speaker on this subject based on the provided context.",
  "relevant_pages": [1],
  "final_answer": "Marcell"
}
```
""" 

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextNumberPrompt: # For questions expecting a numerical answer
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis. Identify what numerical value is being asked for (e.g., episode number, count of mentions, specific statistic if mentioned). Scan context for numbers related to the query. Ensure the number directly answers the question.")
        reasoning_summary: str = Field(description="Concise summary of reasoning.")
        relevant_pages: List[int] = Field(description="List of page numbers containing the numerical answer or strong support for it.")
        final_answer: Union[int, float, Literal['N/A']] = Field(description="""
The exact numerical answer.
- If an episode number is asked: 102.
- If a count is asked (e.g. "how many statistics"): 3.
- Avoid ranges unless specifically asked.
- Return 'N/A' if the specific number is not found or cannot be directly extracted.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example 1:
Question:
"What is the episode number where John discusses customer dissatisfaction statistics?"

Context:
Page 1: "Episode 102. [00:00:00.02] John: I once tried to get an organisation to change..."
Page 1: "[00:00:23.00] John: ...85% of people believe that organizations now have lost that human touch..."

Answer:
```
{
  "step_by_step_analysis": "1. The question asks for an episode number related to John discussing customer dissatisfaction statistics.\n2. The context for Page 1 starts with 'Episode 102' and features John speaking.\n3. John then mentions statistics like '85% of people believe...'.\n4. This directly links Episode 102 to John discussing these statistics.\n5. The episode number is 102.",
  "reasoning_summary": "The context explicitly states 'Episode 102' at the beginning of a segment where John discusses customer dissatisfaction statistics.",
  "relevant_pages": [1],
  "final_answer": 102
}
```
"""
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextBooleanPrompt: # For Yes/No questions
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis. Does the context confirm or deny the statement in the question? Look for explicit mentions or strong implications.")
        reasoning_summary: str = Field(description="Concise summary of reasoning.")
        relevant_pages: List[int] = Field(description="List of page numbers supporting the True/False answer.")
        final_answer: Union[bool, Literal['N/A']] = Field(description="A boolean value (True or False). Return 'N/A' if the context is insufficient to determine a True/False answer.")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"Does 'Episode 102' mention the concept of 'empowerment'?"

Context:
Page 1: "[00:01:15.00] John: You have to have that culture of empowerment and freedom to try things..."

Answer:
```
{
  "step_by_step_analysis": "1. The question asks if 'Episode 102' mentions 'empowerment'.\n2. The provided context is from a podcast transcript, presumably Episode 102.\n3. On Page 1, John is quoted saying 'You have to have that culture of empowerment...'.\n4. This directly confirms the mention of 'empowerment'.\n5. Therefore, the answer is True.",
  "reasoning_summary": "The transcript segment from Page 1 explicitly shows John using the word 'empowerment'.",
  "relevant_pages": [1],
  "final_answer": True
}
```
"""
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextNamesPrompt: # For questions expecting a list of names/items
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis. Identify what list of items is requested (e.g., speakers, topics, episode titles). Extract these items from the context. Ensure they are distinct and relevant.")
        reasoning_summary: str = Field(description="Concise summary of reasoning.")
        relevant_pages: List[int] = Field(description="List of page numbers where the names/items are found.")
        final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
A list of strings. Each string should be an item extracted from the context.
- If speakers are asked for: ["John Doe", "Jane Smith"].
- If topics are asked for: ["customer experience", "company culture", "metrics"].
- If episode titles are asked for: ["Episode 90: Interview Masterclass", "Episode 102: The Human Touch"].
- Return 'N/A' if no relevant items are found in the context.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"What are the key statistics John mentions about customer dissatisfaction in 'Episode 102'?"

Context:
Page 1: "[00:00:23.00] John: ...85% of people believe that organizations now have lost that human touch, 83% believe that organizations take customers for granted. 81% believe that organizations are more interested in cutting costs than creating a great experience..."

Answer:
```
{
    "step_by_step_analysis": "1. The question asks for key statistics John mentions about customer dissatisfaction in 'Episode 102'.\n2. The context is from Page 1 of a transcript where John is speaking.\n3. John states: '85% of people believe that organizations now have lost that human touch'.\n4. John also states: '83% believe that organizations take customers for granted'.\n5. John further states: '81% believe that organizations are more interested in cutting costs than creating a great experience'.\n6. These are three distinct statistics related to customer dissatisfaction mentioned by John.",
    "reasoning_summary": "The transcript on Page 1 shows John listing three specific percentages related to customer beliefs about organizations, which align with the concept of dissatisfaction.",
    "relevant_pages": [1],
    "final_answer": [
        "85% of people believe that organizations now have lost that human touch",
        "83% believe that organizations take customers for granted",
        "81% believe that organizations are more interested in cutting costs than creating a great experience"
    ]
}
```
"""
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class ComparativeAnswerPrompt: # For comparing multiple podcast episodes/series
    instruction = """
You are a question answering system.
Your task is to analyze individual answers related to different podcast episodes or series and provide a comparative response that answers the original question.
Base your analysis only on the provided individual answers - do not make assumptions or include external knowledge.
Before giving a final answer, carefully think out loud and step by step.

Important rules for comparison:
- When the question asks to choose one of the podcast identifiers (e.g., when comparing number of mentions, topics covered), return the podcast identifier (e.g., "Episode 101", "The Startup Chat") that best fits the comparison.
- If data for a podcast identifier is 'N/A' or insufficient for comparison, exclude it.
- If all podcast identifiers are excluded, or if a comparison cannot be made from the provided answers, return 'N/A' as the final answer.
- If only one podcast identifier remains after exclusions, return that identifier if the question implies a selection (e.g. "which had more...").
"""

    user_prompt = """
Here are the individual podcast episode/series answers:
\"\"\"
{context}
\"\"\"

---

Here is the original comparative question:
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the comparison. At least 3-5 steps. Explain how you are comparing the individual answers based on the question's criteria.")
        reasoning_summary: str = Field(description="Concise summary of the comparative reasoning process.")
        relevant_pages: List[int] = Field(description="Leave empty, as this is a comparison of prior answers.")
        final_answer: Union[str, Literal["N/A"]] = Field(description="""
The podcast identifier (e.g., "Episode 101", "The Daily Show") that is the result of the comparison, or 'N/A'.
The identifier should be extracted exactly as it appears in the input or individual answers.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"Which podcast episode, 'Episode 90' or 'Episode 102', has more mentions of the term 'culture'?"

Individual Answers Context:
{
    "Episode 90": {"final_answer": 2, "relevant_pages": [3, 5]},
    "Episode 102": {"final_answer": 5, "relevant_pages": [1, 4]}
}

Answer:
```
{
  "step_by_step_analysis": "1. The question asks to compare 'Episode 90' and 'Episode 102' based on the number of mentions of 'culture'.\n2. From the individual answers, 'Episode 90' had 2 mentions of 'culture'.\n3. 'Episode 102' had 5 mentions of 'culture'.\n4. Comparing the counts: 5 (Episode 102) > 2 (Episode 90).\n5. Therefore, 'Episode 102' had more mentions of 'culture'.",
  "reasoning_summary": "'Episode 102' mentioned 'culture' 5 times, while 'Episode 90' mentioned it 2 times. Thus, 'Episode 102' had more mentions.",
  "relevant_pages": [],
  "final_answer": "Episode 102"
}
```
"""
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt: # This prompt seems generally applicable, no major changes needed.
    system_prompt = """
You are a JSON formatter.
Your task is to format raw LLM response into a valid JSON object.
Your answer should always start with '{' and end with '}'
Your answer should contain only json string, without any preambles, comments, or triple backticks.
"""

    user_prompt = """
Here is the system prompt that defines schema of the json object and provides an example of answer with valid schema:
\"\"\"
{system_prompt}
\"\"\"

---

Here is the LLM response that not following the schema and needs to be properly formatted:
\"\"\"
{response}
\"\"\"
"""


class RerankingPrompt: # General relevance ranking, likely no changes needed for podcast context.
    system_prompt_rerank_single_block = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and retrieved text block related to that query. Your task is to evaluate and score the block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate block based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

    system_prompt_rerank_multiple_blocks = """
You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and several retrieved text blocks related to that query. Your task is to evaluate and score each block based on its relevance to the query provided.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate blocks based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

class RetrievalRankingSingleBlock(BaseModel):
    """Rank retrieved text block relevance to a query."""
    reasoning: str = Field(description="Analysis of the block, identifying key information and how it relates to the query")
    relevance_score: float = Field(description="Relevance score from 0 to 1, where 0 is Completely Irrelevant and 1 is Perfectly Relevant")

class RetrievalRankingMultipleBlocks(BaseModel):
    """Rank retrieved multiple text blocks relevance to a query."""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="A list of text blocks and their associated relevance scores."
    )
