"""Reformulation agent - improves question clarity before routing."""
import logging

from pydantic import BaseModel
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents.shared import create_llm_client

logger = logging.getLogger(__name__)


class ReformulatedQuestion(BaseModel):
    """A reformulated question with extracted entities."""
    reformulated: str  # The improved question
    urls: list[str] = []  # Any URLs mentioned
    intent: str  # Brief description of what user wants


class ValidationResult(BaseModel):
    """Result of validating a reformulation."""
    is_valid: bool  # Whether the reformulation preserves meaning
    issues: list[str] = []  # Any issues found
    suggested_fix: str | None = None  # Corrected reformulation if invalid


# Reformulation system prompt
reformulation_system_prompt = """You are a question reformulation specialist. Your job is to take a user's question and make it clearer and more actionable.

# TASKS
1. Fix typos and grammar
2. Expand abbreviations (e.g., "kb" → "knowledge base", "viz" → "visualization")
3. Make implicit requests explicit
4. Extract any URLs mentioned
5. Identify the user's intent

# RULES
- Keep the original meaning - don't add information the user didn't provide
- Preserve technical terms and proper nouns
- If a URL is mentioned, ensure it's properly formatted (add https:// if missing)
- Be concise - don't over-expand simple questions

# EXAMPLES
Input: "summarise geneplaza.com"
→ reformulated: "Summarize the content of https://geneplaza.com"
→ urls: ["https://geneplaza.com"]
→ intent: "fetch and summarize website"

Input: "make me a logistic regresion plot"
→ reformulated: "Create a logistic regression plot with sample data"
→ urls: []
→ intent: "create visualization"

Input: "whats in the kb about kafka"
→ reformulated: "What information is in the knowledge base about Kafka?"
→ urls: []
→ intent: "search documents"

Input: "calc 5 + 3 * 2"
→ reformulated: "Calculate the result of 5 + 3 * 2"
→ urls: []
→ intent: "mathematical calculation"
"""

# Create model
_llm_client, _chat_model = create_llm_client()
_model = OpenAIChatModel(
    model_name=_chat_model,
    provider=OpenAIProvider(openai_client=_llm_client),
)

reformulation_agent = Agent(
    _model,
    output_type=ReformulatedQuestion,
    system_prompt=reformulation_system_prompt,
)


# Validation system prompt
validation_system_prompt = """You are a reformulation validator. Your job is to check if a reformulated question accurately preserves the meaning of the original.

# VALIDATION CHECKS
1. **Meaning preservation**: Does the reformulation capture the original intent?
2. **No hallucination**: Did the reformulation add information not in the original?
3. **No loss**: Did the reformulation lose important details?
4. **URL accuracy**: Are URLs correctly extracted and formatted?
5. **Intent match**: Does the stated intent match what the user asked?

# RULES
- Be strict about meaning preservation
- Minor grammar/spelling fixes are OK
- Expanding abbreviations is OK if meaning is preserved
- Adding "https://" to URLs is OK
- Adding assumptions NOT in the original is NOT OK

# EXAMPLES
Original: "summarise geneplaza.com"
Reformulated: "Summarize the content of https://geneplaza.com"
→ is_valid: true (meaning preserved, URL formatted)

Original: "whats kafka"
Reformulated: "What is Apache Kafka and how does it work for data streaming?"
→ is_valid: false
→ issues: ["Added 'Apache' and 'data streaming' not in original"]
→ suggested_fix: "What is Kafka?"

Original: "plot sales"
Reformulated: "Create a bar chart of monthly sales data for 2025"
→ is_valid: false
→ issues: ["Added 'bar chart', 'monthly', and '2025' not in original"]
→ suggested_fix: "Create a plot of sales data"
"""

validation_agent = Agent(
    _model,
    output_type=ValidationResult,
    system_prompt=validation_system_prompt,
)


async def validate_reformulation(
    original: str,
    reformulated: ReformulatedQuestion,
) -> ValidationResult:
    """Validate that a reformulation preserves the original meaning.

    Args:
        original: The original user question.
        reformulated: The reformulated question to validate.

    Returns:
        ValidationResult indicating if the reformulation is valid.
    """
    prompt = f"""Validate this reformulation:

Original question: "{original}"

Reformulated: "{reformulated.reformulated}"
Extracted URLs: {reformulated.urls}
Stated intent: "{reformulated.intent}"

Is this reformulation valid?"""

    try:
        result = await validation_agent.run(prompt)
        if result.output.is_valid:
            logger.info(f"Reformulation validated: OK")
        else:
            logger.warning(f"Reformulation issues: {result.output.issues}")
        return result.output
    except Exception as e:
        logger.warning(f"Validation failed: {e}, assuming valid")
        return ValidationResult(is_valid=True, issues=[], suggested_fix=None)


async def reformulate_question(question: str, memory: str | None = None) -> ReformulatedQuestion:
    """Reformulate a user question for better clarity.

    Args:
        question: The original user question.
        memory: Optional memory context from previous exchanges.

    Returns:
        ReformulatedQuestion with improved text, extracted URLs, and intent.
    """
    context = f"[Memory context: {memory}]\n\n" if memory else ""
    prompt = f"{context}Reformulate this user question:\n\n{question}"

    try:
        result = await reformulation_agent.run(prompt)
        logger.info(
            f"Reformulated: '{question[:50]}...' → "
            f"'{result.output.reformulated[:50]}...' (intent: {result.output.intent})"
        )
        return result.output
    except Exception as e:
        logger.warning(f"Reformulation failed: {e}, using original question")
        return ReformulatedQuestion(
            reformulated=question,
            urls=[],
            intent="unknown"
        )


async def reformulate_and_validate(
    question: str,
    memory: str | None = None,
    max_retries: int = 2,
) -> ReformulatedQuestion:
    """Reformulate a question and validate it preserves the original meaning.

    If validation fails, uses the suggested fix or retries reformulation.

    Args:
        question: The original user question.
        memory: Optional memory context from previous exchanges.
        max_retries: Maximum number of reformulation attempts if validation fails.

    Returns:
        A validated ReformulatedQuestion.
    """
    for attempt in range(max_retries + 1):
        # Step 1: Reformulate
        reformulated = await reformulate_question(question, memory)

        # Step 2: Validate
        validation = await validate_reformulation(question, reformulated)

        if validation.is_valid:
            logger.info(f"Reformulation validated on attempt {attempt + 1}")
            return reformulated

        # Validation failed
        logger.warning(f"Validation failed (attempt {attempt + 1}): {validation.issues}")

        # Use suggested fix if provided
        if validation.suggested_fix:
            logger.info(f"Using suggested fix: {validation.suggested_fix}")
            return ReformulatedQuestion(
                reformulated=validation.suggested_fix,
                urls=reformulated.urls,  # Keep extracted URLs
                intent=reformulated.intent,
            )

        # If no fix and more retries, continue loop
        if attempt < max_retries:
            logger.info(f"Retrying reformulation...")

    # All retries exhausted, return original question
    logger.warning("All reformulation attempts failed, using original")
    return ReformulatedQuestion(
        reformulated=question,
        urls=[],
        intent="unknown"
    )
