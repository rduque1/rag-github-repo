"""Assistant agent with configurable LLM backend (Azure, Ollama, LM Studio).
Supports document search, summarization, listing, and web search.
"""
import asyncio
import logging
import re
import sys
import traceback
import asyncpg
import httpx
import pydantic_core
import trafilatura
from urllib.parse import urlparse
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from src.core.database import database_connect
from src.core.settings import settings

logger = logging.getLogger(__name__)


def _create_llm_client() -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str]:
    """Create the appropriate LLM client based on settings."""
    provider = settings.LLM_PROVIDER

    if provider == 'azure':
        client = AsyncAzureOpenAI(
            azure_endpoint=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            api_version=settings.LLM_API_VERSION,
        )
        return client, settings.LLM_CHAT_MODEL

    # OpenAI, Ollama, and LM Studio all use OpenAI-compatible API
    base_url = settings.LLM_BASE_URL
    if provider == 'ollama' and not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=settings.LLM_API_KEY or 'not-needed',
    )
    return client, settings.LLM_CHAT_MODEL


def _create_embedding_client() -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str]:
    """Create the appropriate embedding client based on settings."""
    provider = settings.LLM_PROVIDER

    if provider == 'azure':
        client = AsyncAzureOpenAI(
            azure_endpoint=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            api_version=settings.LLM_API_VERSION,
        )
        return client, settings.LLM_EMBEDDING_MODEL

    # OpenAI, Ollama, and LM Studio all use OpenAI-compatible API
    base_url = settings.LLM_BASE_URL
    if provider == 'ollama' and not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=settings.LLM_API_KEY or 'not-needed',
    )
    return client, settings.LLM_EMBEDDING_MODEL


@dataclass
class Deps:
    openai: AsyncOpenAI | AsyncAzureOpenAI
    pool: asyncpg.Pool
    embedding_model: str  # Model name for embeddings
    selected_docs: list[str] | None = None  # Filter to specific documents
    last_fetched_url: str | None = None  # Track last URL for "save it" commands
    memory: str | None = None  # Summarized conversation context
    generated_images: list[str] | None = None  # Store base64 images from code execution


system_prompt = """
# ROLE
You are a helpful AI Assistant that works with the user's uploaded documents and can also search and scrape the web. You can execute Python and JavaScript code.

# MEMORY
You have access to a memory of key facts from the conversation. Check the `get_memory` tool at the start to recall important context.

# INTENT UNDERSTANDING
First, understand what the user wants to do:
- **Search/Question**: User wants to find specific information → use `retrieve` tool
- **Summarize**: User wants a summary of documents → use `summarize_documents` tool (no document_name needed if documents are pre-selected)
- **List**: User wants to know what documents are available → use `list_documents` tool
- **Compare**: User wants to compare information → use `retrieve` multiple times
- **Web Search**: User wants external information → use `duckduckgo_search` tool
- **Fetch URL**: User provides a URL to read → use `fetch_webpage` tool
- **Save last webpage**: User confirms saving a previously fetched page → use `save_last_webpage_to_kb` tool
- **Run Python code**: User wants to execute Python → use `execute_python_code` tool
- **Run JavaScript code**: User wants to execute JavaScript/Node.js → use `execute_nodejs_code` tool
- **Math/Statistics/Calculations**: Any numerical task → use `execute_python_code` tool

# SELECTED DOCUMENTS
If the user's message includes "[Context: User has selected these documents: ...]", these documents are pre-selected in the UI.
- For `retrieve`, `summarize_documents`, and `list_documents`: they will automatically use only these selected documents
- Do NOT ask for the document name - just call the tool directly
- The tools know which documents are selected

# CRITICAL: USE CODE EXECUTION FOR CALCULATIONS
For ANY task involving:
- Mathematical calculations (arithmetic, algebra, percentages, etc.)
- Statistical analysis (mean, median, standard deviation, correlations, etc.)
- Data processing or transformations
- Date/time calculations
- Numerical comparisons or sorting
- Financial calculations

→ **ALWAYS use `execute_python_code`** instead of calculating in your head.
→ Use numpy/pandas for complex operations.
→ This ensures accuracy and avoids calculation errors.

# CRITICAL: SAVING WEBPAGES TO KNOWLEDGE BASE
When user says "yes", "save it", "add it", "add to knowledge base" after you showed them a webpage:
→ **ALWAYS use `save_last_webpage_to_kb` tool** (no parameters needed, it remembers the URL)
→ Do NOT ask for the URL again

# CODE EXECUTION
- Python: Available packages include numpy, pandas, matplotlib, openpyxl, requests, httpx, pydantic
- JavaScript: Available built-in modules include path, url, querystring, util, crypto
- Use print()/console.log() for output
- Code runs in isolated sandboxed containers
- For Excel files (.xlsx): use pandas with openpyxl (e.g., pd.read_excel(), df.to_excel())
- **PLOTTING RULES** (CRITICAL):
  - plt is pre-imported, just use it directly
  - NEVER use plt.savefig() - the system strips these automatically
  - NEVER reference /mnt/data or file paths - they don't exist
  - Just create plot and call plt.show()
  - The tool response will contain the image as base64 - it displays automatically
  - DO NOT write your own ![image](path) markdown - the image is already in the tool response

# RULES
1. First try to answer from documents. If not found, offer to search the web.
2. When using document info, cite the source document name.
3. When using web search or fetched pages, cite the source URL.
4. Be concise and direct.
5. NEVER ask for a URL that was already discussed.
6. Use memory context to maintain continuity in the conversation.
7. For any calculation, use code execution - don't do mental math.
"""


# Configure LLM provider (Azure, Ollama, or LM Studio based on settings)
llm_client, chat_model_name = _create_llm_client()

model = OpenAIChatModel(
    model_name=chat_model_name,
    provider=OpenAIProvider(openai_client=llm_client),
)

# Memory agent - responsible for extracting key facts from conversations
memory_agent = Agent(
    model,
    output_type=str,
    system_prompt="""You are a memory extraction agent. Your job is to maintain a concise summary of key facts from a conversation.

Given the current memory (if any) and the latest exchange, update the memory with:
- Key facts mentioned by the user (name, preferences, topics of interest)
- Important decisions or conclusions reached
- URLs discussed and their status (fetched, saved to KB, etc.)
- Any context needed for future turns

Rules:
- Output as bullet points (use - for each point)
- Keep only RELEVANT information (remove outdated or superseded facts)
- Be concise - max 20 bullet points
- If a fact is updated, replace the old version
- Remove trivial exchanges (greetings, confirmations)
- If nothing important, return the existing memory unchanged

Format:
- [fact 1]
- [fact 2]
...""",
)


# Create the agent with the Azure model
agent = Agent(
    model,
    output_type=str,
    system_prompt=system_prompt,
    deps_type=Deps,
    tools=[duckduckgo_search_tool()],
)


@agent.tool
def get_memory(context: RunContext[Deps]) -> str:
    """Get the conversation memory containing key facts from previous exchanges.

    Use this at the start of your response to recall important context.
    """
    if context.deps.memory:
        return f"Conversation memory:\n{context.deps.memory}"
    return "No previous conversation memory."


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve relevant document sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query to find relevant documents.
    """

    response = await context.deps.openai.embeddings.create(
        input=search_query,
        model=context.deps.embedding_model,
    )

    embedding = response.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()

    # Filter by selected documents if specified
    if context.deps.selected_docs:
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            WHERE folder = ANY($2)
            ORDER BY embedding <=> $1 LIMIT 5
            """,
            embedding_json,
            context.deps.selected_docs,
        )
    else:
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            ORDER BY embedding <=> $1 LIMIT 5
            """,
            embedding_json,
        )

    if not rows:
        return "No documents found in the knowledge base."

    results = []
    for row in rows:
        source = row["folder"]
        content = row["content"]
        results.append(f"[Source: {source}]\n{content}")

    return '\n\n---\n\n'.join(results)


@agent.tool
async def list_documents(context: RunContext[Deps]) -> str:
    """List all available documents in the knowledge base.

    Use this when the user wants to know what documents are available.

    Args:
        context: The call context.
    """
    if context.deps.selected_docs:
        docs = context.deps.selected_docs
    else:
        rows = await context.deps.pool.fetch(
            'SELECT DISTINCT folder FROM repo ORDER BY folder'
        )
        docs = [row['folder'] for row in rows]

    if not docs:
        return "No documents found in the knowledge base."

    doc_list = '\n'.join(f'- {doc}' for doc in docs)
    return f"Available documents ({len(docs)}):\n{doc_list}"


# Max tokens for summarization context
MAX_SUMMARY_TOKENS = 6000


async def _summarize_chunk(openai: AsyncOpenAI | AsyncAzureOpenAI, chunk: str, source: str) -> str:
    """Generate a summary for a single chunk of content."""
    response = await openai.chat.completions.create(
        model=settings.LLM_CHAT_MODEL,
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are a summarization assistant. Provide a concise '
                    'summary of the following content. Keep key facts and '
                    'important details. Be brief but comprehensive.'
                ),
            },
            {
                'role': 'user',
                'content': f'Summarize this content from [{source}]:\n\n{chunk}',
            },
        ],
        max_tokens=500,
    )
    return f"[{source}] {response.choices[0].message.content}"


async def _hierarchical_summarize(
    openai: AsyncOpenAI | AsyncAzureOpenAI,
    chunks: list[tuple[str, str]],  # List of (source, content)
) -> str:
    """
    Perform hierarchical summarization (map-reduce style).

    If content is too large, summarize chunks individually, then combine
    and summarize the summaries recursively.
    """
    # Estimate tokens (rough: 4 chars = 1 token)
    total_chars = sum(len(content) for _, content in chunks)
    estimated_tokens = total_chars // 4

    if estimated_tokens <= MAX_SUMMARY_TOKENS:
        # Content fits, return as-is for the agent to summarize
        results = []
        for source, content in chunks:
            results.append(f"[Source: {source}]\n{content}")
        return '\n\n---\n\n'.join(results)

    # Too large - summarize each chunk individually
    summaries = []
    for source, content in chunks:
        # Split very large chunks further
        if len(content) > MAX_SUMMARY_TOKENS * 4:
            # Split into sub-chunks
            sub_chunk_size = MAX_SUMMARY_TOKENS * 3
            for i in range(0, len(content), sub_chunk_size):
                sub_chunk = content[i : i + sub_chunk_size]
                summary = await _summarize_chunk(openai, sub_chunk, source)
                summaries.append((source, summary))
        else:
            summary = await _summarize_chunk(openai, content, source)
            summaries.append((source, summary))

    # Check if combined summaries fit
    combined_chars = sum(len(s) for _, s in summaries)
    if combined_chars // 4 <= MAX_SUMMARY_TOKENS:
        # Summaries fit, return them
        results = []
        for source, summary in summaries:
            results.append(summary)
        return (
            "Intermediate summaries (synthesize these for final answer):\n\n"
            + '\n\n---\n\n'.join(results)
        )

    # Still too large - recursively summarize the summaries
    return await _hierarchical_summarize(openai, summaries)


@agent.tool
async def summarize_documents(
    context: RunContext[Deps],
    document_name: str | None = None,
) -> str:
    """Get content from documents for summarization.

    Use this when the user wants a summary of the documents.
    Handles large documents by creating hierarchical summaries.

    Args:
        context: The call context.
        document_name: Optional specific document to summarize. If None,
            summarizes from all available/selected documents.
    """
    if document_name:
        # Get ALL content from specific document
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            WHERE folder = $1
            ORDER BY id
            """,
            document_name,
        )
    elif context.deps.selected_docs:
        # Get ALL content from selected documents
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            WHERE folder = ANY($1)
            ORDER BY folder, id
            """,
            context.deps.selected_docs,
        )
    else:
        # Get ALL content from all documents
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            ORDER BY folder, id
            """
        )

    if not rows:
        return "No documents found to summarize."

    # Group content by document
    chunks: list[tuple[str, str]] = [
        (row['folder'], row['content']) for row in rows
    ]

    # Perform hierarchical summarization if needed
    return await _hierarchical_summarize(context.deps.openai, chunks)


async def _embed_and_store_webpage(
    pool: asyncpg.Pool,
    openai: AsyncOpenAI | AsyncAzureOpenAI,
    url: str,
    title: str,
    content: str,
) -> int:
    """Embed webpage content and store in database."""
    from src.preprocessing.document_processor import embed_chunks

    # Split content into chunks (similar to document_processor)
    chunk_size = 1000
    chunks = []
    words = content.split()
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    if not chunks:
        return 0

    # Use URL as the document name (folder), or title if available
    doc_name = title if title else urlparse(url).netloc + urlparse(url).path
    doc_name = doc_name[:100]  # Limit length

    # Embed and store (embed_chunks creates its own pool/openai internally)
    await embed_chunks(doc_name, chunks)
    return len(chunks)


@agent.tool
async def fetch_webpage(
    context: RunContext[Deps],
    url: str,
    save_to_kb: bool = False,
) -> str:
    """Fetch and extract content from a webpage.

    Use this when the user provides a URL to read, summarize, or analyze.
    Can optionally save the webpage to the knowledge base.

    Args:
        context: The call context.
        url: The URL to fetch.
        save_to_kb: If True, saves the webpage content to the knowledge base.
            Use this when the user asks to "add", "save", or "store" a URL.
    """
    logger.info(f"fetch_webpage called: url={url}, save_to_kb={save_to_kb}")
    # Browser-like headers to avoid bot detection
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    try:
        # Fetch the page with browser-like settings
        logger.info(f"Fetching {url}...")
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            http2=True,
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text
        logger.info(f"Fetched {url}, got {len(html)} chars of HTML")

        # Extract main content using trafilatura
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        if not extracted:
            logger.warning(f"Could not extract content from {url}")
            return (
                f"Could not extract readable content from {url}. "
                "The page might require JavaScript or have anti-bot protection. "
                "You can paste the article text directly and I'll help analyze it."
            )

        logger.info(f"Extracted {len(extracted)} chars from {url}")
        # Get metadata
        metadata = trafilatura.extract_metadata(html)
        title = metadata.title if metadata and metadata.title else urlparse(url).netloc

        # Store URL in deps for later "save it" commands
        context.deps.last_fetched_url = url

        # Save to knowledge base if requested
        if save_to_kb:
            logger.info(f"Saving {url} to knowledge base...")
            chunks_stored = await _embed_and_store_webpage(
                context.deps.pool,
                context.deps.openai,
                url,
                title,
                extracted,
            )
            logger.info(f"Saved {chunks_stored} chunks for {url}")
            return (
                f"Successfully fetched and saved [{title}]({url}) to knowledge base "
                f"({chunks_stored} chunks stored).\n\n"
                f"**Content Preview:**\n{extracted[:2000]}{'...' if len(extracted) > 2000 else ''}"
            )

        # Return content for summarization/analysis
        return (
            f"**Source:** [{title}]({url})\n\n"
            f"**Content:**\n{extracted}"
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching {url}: {e.response.status_code}")
        return (
            f"Failed to fetch {url}: HTTP {e.response.status_code}. "
            "The site may be blocking automated requests. "
            "Try pasting the article text directly."
        )
    except httpx.TimeoutException:
        logger.error(f"Timeout fetching {url}")
        return (
            f"Request to {url} timed out. "
            "The site may be slow or blocking requests. "
            "Try pasting the article text directly."
        )
    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching {url}: {e}")
        logger.error(traceback.format_exc())
        return (
            f"Failed to fetch {url}: {e}. "
            "Try pasting the article text directly and I'll help analyze it."
        )
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        logger.error(traceback.format_exc())
        return f"Error processing {url}: {e}. Try pasting the article text directly."


@agent.tool
async def save_last_webpage_to_kb(context: RunContext[Deps]) -> str:
    """Save the last fetched webpage to the knowledge base.

    Use this when the user confirms they want to save a webpage that was
    just shown to them. For example, when user says "yes", "save it",
    "add it to kb", "add to knowledge base" after viewing a webpage.

    This tool remembers the last URL that was fetched, so you don't need
    to ask the user for the URL again.
    """
    # First check deps for the URL (set during this session)
    url = context.deps.last_fetched_url

    # If not in deps, try to find URL in recent conversation
    if not url:
        return (
            "I don't have a recently fetched webpage to save. "
            "Please provide the URL you'd like me to add to the knowledge base."
        )

    # Fetch and save the webpage
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            http2=True,
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        if not extracted:
            return f"Could not extract content from {url} for saving."

        metadata = trafilatura.extract_metadata(html)
        title = metadata.title if metadata and metadata.title else urlparse(url).netloc

        chunks_stored = await _embed_and_store_webpage(
            context.deps.pool,
            context.deps.openai,
            url,
            title,
            extracted,
        )

        return (
            f"✓ Successfully saved [{title}]({url}) to your knowledge base "
            f"({chunks_stored} chunks stored). It will now appear in your document list."
        )

    except Exception as e:
        return f"Failed to save webpage: {e}"


@agent.tool
async def execute_python_code(context: RunContext[Deps], code: str) -> str:
    """Execute Python code in a sandboxed container.

    Use this when the user asks you to run Python code, perform calculations,
    data processing, create visualizations, or test Python snippets.

    Available packages: numpy, pandas, matplotlib, requests, httpx, pydantic

    IMPORTANT PLOTTING RULES:
    - plt is already imported for you
    - NEVER use plt.savefig() - it won't work
    - NEVER reference file paths or attachments
    - NEVER use display() or Image()
    - Just create your plot and end with plt.show()
    - The system automatically captures the figure as base64
    - DO NOT write markdown image tags yourself - the system returns the image

    Correct plotting example:
    ```
    import numpy as np
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.show()
    ```

    Args:
        context: The call context.
        code: The Python code to execute. Use print() for text output.
              For plots: just create the figure and call plt.show().
    """
    # Remove any savefig calls from the code to prevent file-saving patterns
    import re as regex
    cleaned_code = regex.sub(
        r'plt\.savefig\([^)]*\)',
        '# savefig removed - figures are captured automatically',
        code
    )
    # Also remove any /mnt/data references
    cleaned_code = regex.sub(
        r'["\'][^"\']*(/mnt/data|/tmp)[^"\']*["\']',
        '"/dev/null"',
        cleaned_code
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.PYTHON_EXECUTOR_URL}/execute",
                json={"code": cleaned_code},
            )
            response.raise_for_status()
            result = response.json()

        output_parts = []
        if result.get("stdout"):
            output_parts.append(f"**Output:**\n```\n{result['stdout']}\n```")
        if result.get("stderr"):
            output_parts.append(f"**Stderr:**\n```\n{result['stderr']}\n```")
        if result.get("return_value"):
            output_parts.append(f"**Return value:** `{result['return_value']}`")
        if result.get("error"):
            output_parts.append(f"**Error:**\n```\n{result['error']}\n```")

        # Store images in deps instead of returning them to LLM (saves tokens)
        images = result.get("images", [])
        if images:
            if context.deps.generated_images is None:
                context.deps.generated_images = []
            context.deps.generated_images.extend(images)
            output_parts.append(f"**Generated Plot(s):** {len(images)} image(s) created successfully.")

        if not output_parts:
            return "Code executed successfully (no output)."

        return "\n\n".join(output_parts)

    except httpx.ConnectError:
        return "Error: Python executor is not available. Make sure the container is running."
    except Exception as e:
        return f"Error executing Python code: {e}"


@agent.tool
async def execute_nodejs_code(context: RunContext[Deps], code: str) -> str:
    """Execute Node.js/JavaScript code in a sandboxed container.

    Use this when the user asks you to run JavaScript code, perform calculations,
    or test JavaScript snippets.

    Available built-in modules: path, url, querystring, util, crypto

    Args:
        context: The call context.
        code: The JavaScript code to execute. Use console.log() to show output.
              The last expression value is automatically returned.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.NODEJS_EXECUTOR_URL}/execute",
                json={"code": code},
            )
            response.raise_for_status()
            result = response.json()

        output_parts = []
        if result.get("stdout"):
            output_parts.append(f"**Output:**\n```\n{result['stdout']}\n```")
        if result.get("stderr"):
            output_parts.append(f"**Stderr:**\n```\n{result['stderr']}\n```")
        if result.get("return_value"):
            output_parts.append(f"**Return value:** `{result['return_value']}`")
        if result.get("error"):
            output_parts.append(f"**Error:**\n```\n{result['error']}\n```")

        if not output_parts:
            return "Code executed successfully (no output)."

        return "\n\n".join(output_parts)

    except httpx.ConnectError:
        return "Error: Node.js executor is not available. Make sure the container is running."
    except Exception as e:
        return f"Error executing JavaScript code: {e}"


def _extract_urls_from_history(messages: list[tuple[str, str]]) -> str | None:
    """Extract the most recent URL from message history."""
    url_pattern = r'https?://[^\s\]\)\"\'<>]+'

    # Search from most recent to oldest
    for _, content in reversed(messages):
        urls = re.findall(url_pattern, content)
        if urls:
            return urls[-1]  # Return the last URL found in that message
    return None


async def update_memory(current_memory: str | None, user_message: str, assistant_response: str) -> str:
    """
    Use the memory agent to update conversation memory with new exchange.

    Args:
        current_memory: The current memory state (bullet points).
        user_message: The latest user message.
        assistant_response: The assistant's response.

    Returns:
        Updated memory as bullet points.
    """
    prompt = f"""Current memory:
{current_memory or "(empty)"}

Latest exchange:
User: {user_message}
Assistant: {assistant_response[:1000]}{"..." if len(assistant_response) > 1000 else ""}

Update the memory with any important new facts. Return the updated bullet points."""

    result = await memory_agent.run(prompt)
    return result.output


async def stream_messages(
    question: str,
    selected_docs: list[str] | None = None,
    message_history: list[dict[str, str]] | None = None,
    memory: str | None = None,
) -> AsyncGenerator[str | dict, None]:
    """
    Stream messages for Streamlit interface.

    Args:
        question: The user's question.
        selected_docs: Optional list of document names to limit search to.
        message_history: Optional list of previous messages for context.
            Each message is a dict with 'role' and 'content' keys.
        memory: Optional summarized memory from previous exchanges.

    Yields:
        Either a string chunk of text, or a dict with 'images' key containing base64 images.
    """
    logger.info(f"stream_messages called: question={question[:100]}...")
    embedding_client, embedding_model = _create_embedding_client()

    try:
        async with database_connect() as pool:
            deps = Deps(
                openai=embedding_client,
                pool=pool,
                embedding_model=embedding_model,
                selected_docs=selected_docs,
                memory=memory,
            )

            # Build message history for the agent
            messages: list[tuple[str, str]] = []
            if message_history:
                for msg in message_history:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == 'user':
                        messages.append(('user', content))
                    elif role == 'assistant':
                        messages.append(('assistant', content))

            # Extract last URL from history and set in deps
            last_url = _extract_urls_from_history(messages)
            if last_url:
                deps.last_fetched_url = last_url

            # Add context about selected documents to the question
            augmented_question = question
            if selected_docs:
                docs_list = ', '.join(selected_docs)
                augmented_question = f"[Context: User has selected these documents: {docs_list}]\n\n{question}"

            logger.info("Starting agent.run_stream...")
            async with agent.run_stream(
                augmented_question, deps=deps, message_history=messages
            ) as result:
                async for message in result.stream_text(delta=True):
                    yield message
            logger.info("Agent stream completed")

            # After streaming completes, yield any generated images
            if deps.generated_images:
                yield {"images": deps.generated_images}
    except Exception as e:
        logger.error(f"Error in stream_messages: {e}")
        logger.error(traceback.format_exc())
        yield f"\n\n❌ Error: {e}"


async def run_agent(question: str) -> None:
    """
    Entry point to run the agent and perform RAG based question answering.
    """
    embedding_client, embedding_model = _create_embedding_client()

    async with database_connect() as pool:
        deps = Deps(
            openai=embedding_client,
            pool=pool,
            embedding_model=embedding_model,
        )
        answer = await agent.run(question, deps=deps)

    print(answer.data)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None

    if action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'Qual problema é resolvido no workshop de kafka?'
        asyncio.run(run_agent(q))
    else:
        print('Exiting')
