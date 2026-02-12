"""Web agent - specialized for web search and URL fetching."""
import asyncio
import logging
import traceback
from urllib.parse import urlparse

import httpx
import trafilatura
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

logger = logging.getLogger(__name__)

# Check if Playwright is available
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. JavaScript rendering disabled. Install with: pip install playwright && playwright install chromium")

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from src.agents.shared import create_llm_client, Deps


async def _fetch_with_playwright(url: str, timeout: int = 30000) -> str:
    """Fetch a webpage using Playwright (renders JavaScript)."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context(
                user_agent=(
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                )
            )
            page = await context.new_page()
            await page.goto(url, wait_until='networkidle', timeout=timeout)
            # Wait a bit more for any dynamic content
            await asyncio.sleep(1)
            html = await page.content()
            return html
        finally:
            await browser.close()


async def _fetch_with_httpx(url: str) -> str:
    """Fetch a webpage using httpx (static HTML only)."""
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
    }
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        http2=True,
    ) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.text


# Focused system prompt for web operations
web_system_prompt = """
You are a web research specialist with two main tools:

1. **fetch_webpage**: Use when given a specific URL to read
   - Query contains "fetch", "get content from", or a URL (www.*, http://, https://)
   - Returns the actual page content

2. **duckduckgo_search**: Use when asked to search for information
   - Query contains "search", "find", "look up"
   - Returns search results, not page content

# CRITICAL RULE
If the query mentions a URL (www.example.com, https://...), ALWAYS use fetch_webpage.
Do NOT use duckduckgo_search to search FOR a URL - fetch it directly.

# Examples
- "Fetch content from www.example.com" → fetch_webpage(url="https://www.example.com")
- "What does https://site.com do" → fetch_webpage(url="https://site.com")
- "Search for AI news" → duckduckgo_search(query="AI news")

When providing information, cite the source URL. Be concise.
"""

# Create model
_llm_client, _chat_model = create_llm_client()
_model = OpenAIChatModel(
    model_name=_chat_model,
    provider=OpenAIProvider(openai_client=_llm_client),
)

web_agent = Agent(
    _model,
    output_type=str,
    system_prompt=web_system_prompt,
    deps_type=Deps,
    tools=[duckduckgo_search_tool()],
)


@web_agent.tool
async def fetch_webpage(
    context: RunContext[Deps],
    url: str,
    save_to_kb: bool = False,
    use_javascript: bool = True,
) -> str:
    """Fetch and extract content from a webpage.

    Args:
        context: The call context.
        url: The URL to fetch.
        save_to_kb: If True, saves the content to the knowledge base.
        use_javascript: If True, uses a headless browser to render JavaScript.
                       Set to False for faster fetching of static pages.
    """
    # Ensure URL has scheme
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'

    logger.info(f"Fetching webpage: {url}, save_to_kb={save_to_kb}, js={use_javascript}")

    try:
        # Try Playwright first if available and JS rendering requested
        if use_javascript and PLAYWRIGHT_AVAILABLE:
            try:
                logger.info(f"Using Playwright (JavaScript enabled) for {url}")
                html = await _fetch_with_playwright(url)
            except Exception as e:
                logger.warning(f"Playwright failed, falling back to httpx: {e}")
                html = await _fetch_with_httpx(url)
        else:
            if use_javascript and not PLAYWRIGHT_AVAILABLE:
                logger.warning("Playwright not available, using httpx (no JS rendering)")
            html = await _fetch_with_httpx(url)

        logger.info(f"Fetched {url}, got {len(html)} chars of HTML")

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        if not extracted:
            return f"Could not extract content from {url}."

        metadata = trafilatura.extract_metadata(html)
        title = metadata.title if metadata and metadata.title else urlparse(url).netloc

        # Store URL for later save commands
        context.deps.last_fetched_url = url

        if save_to_kb:
            from src.agents.shared import Deps
            chunks_stored = await _embed_and_store_webpage(
                context.deps.pool,
                context.deps.openai,
                url,
                title,
                extracted,
            )
            return (
                f"Saved [{title}]({url}) to knowledge base ({chunks_stored} chunks).\n\n"
                f"Preview:\n{extracted[:2000]}..."
            )

        return f"**Source:** [{title}]({url})\n\n**Content:**\n{extracted}"

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching {url}: {e.response.status_code}")
        return f"Failed to fetch {url}: HTTP {e.response.status_code}"
    except httpx.TimeoutException:
        logger.error(f"Timeout fetching {url}")
        return f"Request to {url} timed out."
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        logger.error(traceback.format_exc())
        return f"Error fetching {url}: {e}"


@web_agent.tool
async def save_last_webpage_to_kb(context: RunContext[Deps]) -> str:
    """Save the last fetched webpage to the knowledge base."""
    url = context.deps.last_fetched_url
    if not url:
        return "No recently fetched webpage to save."

    # Re-fetch and save
    return await fetch_webpage.function(context, url, save_to_kb=True)


async def _embed_and_store_webpage(pool, openai, url: str, title: str, content: str) -> int:
    """Embed webpage content and store in database."""
    from src.preprocessing.document_processor import embed_chunks

    # Split content into chunks
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

    doc_name = title if title else urlparse(url).netloc + urlparse(url).path
    doc_name = doc_name[:100]

    await embed_chunks(doc_name, chunks)
    return len(chunks)


async def run_web_task(query: str, deps: Deps) -> str:
    """Run the web agent with a query."""
    logger.info(f"Running web task: {query[:100]}...")
    try:
        result = await web_agent.run(query, deps=deps)
        logger.info(f"Web task completed, output length: {len(result.output)}")
        return result.output
    except Exception as e:
        logger.error(f"Web task failed: {e}")
        logger.error(traceback.format_exc())
        raise
