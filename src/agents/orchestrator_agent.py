"""Orchestrator agent that routes requests to specialized agents.

This improves performance by:
1. Running specialized agents with smaller, focused prompts
2. Enabling parallel execution of independent sub-tasks
3. Reducing context overhead per agent call
"""
import asyncio
import logging
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

import asyncpg
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents.shared import create_llm_client, create_embedding_client, Deps
from src.agents.retrieval_agent import retrieval_agent, run_retrieval
from src.agents.web_agent import web_agent, run_web_task
from src.agents.code_agent import code_agent, run_code_task
from src.agents.reformulation_agent import (
    reformulate_and_validate,
    ReformulatedQuestion,
)
from src.core.database import database_connect


class TaskType(str, Enum):
    """Types of tasks the orchestrator can route."""
    RETRIEVAL = "retrieval"  # Document search, summarization, listing
    WEB = "web"  # Web search, URL fetching, saving to KB
    CODE = "code"  # Python or JavaScript execution
    CHAT = "chat"  # Direct response, no tools needed


class SubTask(BaseModel):
    """A sub-task to be executed by a specialized agent."""
    task_type: TaskType
    query: str
    depends_on: list[int] = []  # Indices of tasks this depends on


class TaskPlan(BaseModel):
    """Plan of tasks to execute."""
    tasks: list[SubTask]
    synthesis_prompt: str  # How to combine results


# Orchestrator system prompt - focused on planning, not execution
orchestrator_system_prompt = """
# ROLE
You are an orchestrator that routes user requests to specialized agents.

# AVAILABLE AGENTS
1. **RETRIEVAL**: Search documents, summarize, list available docs
2. **WEB**: Fetch URL content, DuckDuckGo search, save webpages to KB
3. **CODE**: Execute Python (numpy, pandas, matplotlib) or JavaScript
4. **CHAT**: Direct answer without tools (greetings, clarifications)

# CRITICAL: URL HANDLING
When the user mentions a URL (www.*, http://, https://):
→ Create a WEB task with query: "Fetch content from [URL]"
→ Do NOT ask to "search for" the URL - FETCH it directly
→ The web agent will use fetch_webpage tool to get the actual page content

# TASK PLANNING
For each user request, create a plan:
1. Identify what sub-tasks are needed
2. Mark dependencies (task B needs result from task A)
3. Independent tasks will run in parallel for speed
4. Provide a synthesis prompt to combine results

# EXAMPLES
User: "What does www.example.com do?"
→ Task 1: WEB - "Fetch content from www.example.com"
→ Synthesis: "Summarize what the website does based on its content"

User: "Summarize this site: https://example.com"
→ Task 1: WEB - "Fetch content from https://example.com"
→ Synthesis: "Provide a summary of the website content"

User: "Search the web for AI news"
→ Task 1: WEB - "Search for AI news" (uses DuckDuckGo)
→ Synthesis: "Present the search results"

User: "What does document X say about Y"
→ Task 1: RETRIEVAL - "Find info about Y in document X"
→ Synthesis: "Answer based on document content"

# SELECTED DOCUMENTS
If user message includes "[Context: User has selected these documents: ...]",
pass this context to RETRIEVAL tasks.

# CRITICAL: VISUALIZATIONS AND PLOTS
When the user asks to create, make, plot, visualize, or show:
- Charts, graphs, plots (line, bar, scatter, pie, etc.)
- Regression, classification, clustering visualizations
- Any data visualization or figure
→ ALWAYS route to CODE agent with the full request
→ The CODE agent has matplotlib, seaborn, numpy, pandas, scikit-learn

# EXAMPLES FOR CODE
User: "Make me a logistic regression plot"
→ Task 1: CODE - "Create a logistic regression plot with sample data"
→ Synthesis: "Present the generated visualization"

User: "Plot a sine wave"
→ Task 1: CODE - "Plot a sine wave using matplotlib"
→ Synthesis: "Show the plot"

User: "Create a bar chart of sales data"
→ Task 1: CODE - "Create a bar chart visualization"
→ Synthesis: "Present the chart"

# RULES
1. Use CHAT for simple greetings or clarifications
2. Prefer parallel execution when tasks are independent
3. CODE tasks often depend on RETRIEVAL for data
4. For calculations, ALWAYS route to CODE agent
5. For visualizations/plots, ALWAYS route to CODE agent
6. URLs = FETCH, not search
"""


# Create the orchestrator model
_llm_client, _chat_model = create_llm_client()
_model = OpenAIChatModel(
    model_name=_chat_model,
    provider=OpenAIProvider(openai_client=_llm_client),
)


# Simple orchestrator for routing (structured output)
router_agent = Agent(
    _model,
    output_type=TaskPlan,
    system_prompt=orchestrator_system_prompt,
)


# Memory agent for context tracking
memory_agent = Agent(
    _model,
    output_type=str,
    system_prompt="""You are a memory extraction agent. Extract and maintain key facts from conversations.

Given current memory and the latest exchange, update with:
- Key facts mentioned by user
- Important decisions or conclusions
- URLs discussed and their status
- Context needed for future turns

Rules:
- Output as bullet points (use - for each point)
- Max 20 bullet points
- Replace outdated facts with new versions
- Remove trivial exchanges

Format:
- [fact 1]
- [fact 2]
...""",
)


# Synthesis agent - combines results from sub-agents
synthesis_agent = Agent(
    _model,
    output_type=str,
    system_prompt="""You are a synthesis agent. Your job is to combine results from multiple specialized agents into a coherent response.

Given:
- The user's original question
- Results from various sub-tasks
- A synthesis prompt

Create a unified, well-structured response that:
1. Addresses the user's question directly
2. Cites sources appropriately
3. Is concise and clear
4. Highlights key findings

For code results with images, note that images are displayed separately.""",
)


def _detect_intent_fallback(question: str) -> TaskPlan:
    """Detect intent from keywords when LLM fails to produce a plan."""
    import re
    q_lower = question.lower()

    # URL pattern detection
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    has_url = bool(re.search(url_pattern, question))

    # Keyword-based intent detection
    web_keywords = ['website', 'url', 'fetch', 'scrape', 'site', 'page', 'www.', 'http']
    search_keywords = ['search', 'find online', 'look up', 'google', 'duckduckgo']
    code_keywords = [
        'calculate', 'compute', 'run code', 'execute', 'python', 'javascript',
        # Visualization keywords
        'plot', 'chart', 'graph', 'visualize', 'visualization', 'figure',
        'draw', 'create a', 'make a', 'show me a', 'generate a',
        # ML/Stats keywords that need code
        'regression', 'classification', 'clustering', 'histogram', 'scatter',
        'bar chart', 'pie chart', 'line chart', 'heatmap', 'boxplot',
        'logistic', 'linear', 'decision tree', 'random forest',
    ]
    doc_keywords = ['document', 'summarize', 'summary', 'what does', 'in my files', 'in the kb']

    tasks = []

    if has_url or any(kw in q_lower for kw in web_keywords):
        # Extract URL if present, otherwise use the question
        url_match = re.search(url_pattern, question)
        query = f"Fetch and analyze: {url_match.group() if url_match else question}"
        tasks.append(SubTask(task_type=TaskType.WEB, query=query))
    elif any(kw in q_lower for kw in search_keywords):
        tasks.append(SubTask(task_type=TaskType.WEB, query=question))
    elif any(kw in q_lower for kw in code_keywords):
        tasks.append(SubTask(task_type=TaskType.CODE, query=question))
    elif any(kw in q_lower for kw in doc_keywords):
        tasks.append(SubTask(task_type=TaskType.RETRIEVAL, query=question))
    else:
        # Default: try retrieval first, it's the most common use case
        tasks.append(SubTask(task_type=TaskType.RETRIEVAL, query=question))

    return TaskPlan(
        tasks=tasks,
        synthesis_prompt="Provide a clear, helpful response based on the results."
    )


async def execute_task(
    task: SubTask,
    deps: Deps,
    previous_results: dict[int, str],
) -> str:
    """Execute a single sub-task using the appropriate specialized agent."""
    # Build context from dependent tasks
    context = ""
    for dep_idx in task.depends_on:
        if dep_idx in previous_results:
            context += f"\n[Previous result]: {previous_results[dep_idx]}\n"

    full_query = f"{context}\n{task.query}" if context else task.query
    logger.info(f"Executing {task.task_type.value} task: {task.query[:100]}...")

    try:
        if task.task_type == TaskType.RETRIEVAL:
            result = await run_retrieval(full_query, deps)
        elif task.task_type == TaskType.WEB:
            result = await run_web_task(full_query, deps)
        elif task.task_type == TaskType.CODE:
            result = await run_code_task(full_query, deps)
        else:  # CHAT
            result = task.query  # Just echo for synthesis
        logger.info(f"Task {task.task_type.value} completed, result length: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Task {task.task_type.value} failed: {e}")
        logger.error(traceback.format_exc())
        return f"Error executing {task.task_type.value} task: {e}"


async def execute_plan(
    plan: TaskPlan,
    deps: Deps,
) -> tuple[dict[int, str], list[str]]:
    """Execute a task plan, running independent tasks in parallel.

    Returns:
        Tuple of (results dict, list of generated images)
    """
    results: dict[int, str] = {}
    all_images: list[str] = []

    # Group tasks by dependency level
    remaining = set(range(len(plan.tasks)))

    while remaining:
        # Find tasks ready to execute (all dependencies satisfied)
        ready = []
        for idx in remaining:
            task = plan.tasks[idx]
            if all(dep in results for dep in task.depends_on):
                ready.append(idx)

        if not ready:
            # Circular dependency or error - break
            break

        # Execute ready tasks in parallel
        async def run_task(idx: int) -> tuple[int, str]:
            task = plan.tasks[idx]
            # Create a fresh deps for code tasks to capture images
            task_deps = Deps(
                openai=deps.openai,
                pool=deps.pool,
                embedding_model=deps.embedding_model,
                selected_docs=deps.selected_docs,
                last_fetched_url=deps.last_fetched_url,
                memory=deps.memory,
                generated_images=[],
            )
            result = await execute_task(task, task_deps, results)
            # Collect images from code execution
            if task_deps.generated_images:
                all_images.extend(task_deps.generated_images)
            return idx, result

        task_results = await asyncio.gather(*[run_task(idx) for idx in ready])

        for idx, result in task_results:
            results[idx] = result
            remaining.remove(idx)

    return results, all_images


async def update_memory(current_memory: str | None, user_message: str, assistant_response: str) -> str:
    """Update conversation memory with new exchange."""
    prompt = f"""Current memory:
{current_memory or "(empty)"}

Latest exchange:
User: {user_message}
Assistant: {assistant_response[:1000]}{"..." if len(assistant_response) > 1000 else ""}

Update the memory with any important new facts. Return the updated bullet points."""

    result = await memory_agent.run(prompt)
    return result.output


async def stream_orchestrated(
    question: str,
    selected_docs: list[str] | None = None,
    message_history: list[dict[str, str]] | None = None,
    memory: str | None = None,
) -> AsyncGenerator[str | dict, None]:
    """
    Stream orchestrated responses for the Streamlit interface.

    The orchestrator:
    1. Plans the required sub-tasks
    2. Executes independent tasks in parallel
    3. Synthesizes results into a coherent response
    """
    embedding_client, embedding_model = create_embedding_client()

    async with database_connect() as pool:
        deps = Deps(
            openai=embedding_client,
            pool=pool,
            embedding_model=embedding_model,
            selected_docs=selected_docs,
            memory=memory,
            generated_images=[],
        )

        # Add context about selected documents
        augmented_question = question
        if selected_docs:
            docs_list = ', '.join(selected_docs)
            augmented_question = f"[Context: User has selected these documents: {docs_list}]\n\n{question}"

        # Add memory context
        if memory:
            augmented_question = f"[Memory: {memory}]\n\n{augmented_question}"

        # Step 0: Reformulate and validate the question
        yield "✨ Understanding your request...\n\n"
        reformulated = await reformulate_and_validate(question, memory)

        # Use reformulated question for planning
        planning_question = reformulated.reformulated
        if selected_docs:
            docs_list = ', '.join(selected_docs)
            planning_question = f"[Context: User has selected these documents: {docs_list}]\n\n{planning_question}"

        # Step 1: Plan the tasks
        yield "🔍 Planning tasks...\n\n"
        logger.info(f"Planning tasks for: {reformulated.reformulated[:100]}...")

        try:
            plan_result = await router_agent.run(planning_question)
            plan = plan_result.output
            logger.info(f"Plan created with {len(plan.tasks)} tasks: {[t.task_type.value for t in plan.tasks]}")
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            logger.error(traceback.format_exc())
            plan = None

        # Fallback: detect intent from keywords if plan is empty or failed
        if not plan or len(plan.tasks) == 0:
            logger.warning("Empty or failed plan, using keyword-based intent detection")
            plan = _detect_intent_fallback(reformulated.reformulated)
            logger.info(f"Fallback plan: {len(plan.tasks)} tasks: {[t.task_type.value for t in plan.tasks]}")

        # Handle simple chat responses
        if len(plan.tasks) == 1 and plan.tasks[0].task_type == TaskType.CHAT:
            yield plan.tasks[0].query
            return

        # Step 2: Execute the plan
        task_names = {
            TaskType.RETRIEVAL: "📚 Searching documents",
            TaskType.WEB: "🌐 Searching web",
            TaskType.CODE: "💻 Running code",
            TaskType.CHAT: "💬 Processing",
        }

        parallel_groups = []
        remaining = set(range(len(plan.tasks)))
        while remaining:
            ready = [i for i in remaining
                     if all(d in set(range(len(plan.tasks))) - remaining for d in plan.tasks[i].depends_on)]
            if not ready:
                break
            parallel_groups.append(ready)
            remaining -= set(ready)

        for group in parallel_groups:
            group_tasks = [plan.tasks[i] for i in group]
            task_str = " & ".join(task_names[t.task_type] for t in group_tasks)
            yield f"{task_str}...\n"

        try:
            results, images = await execute_plan(plan, deps)
            logger.info(f"Plan executed, got {len(results)} results")
            for idx, result in results.items():
                logger.debug(f"Result {idx}: {result[:200]}..." if len(result) > 200 else f"Result {idx}: {result}")
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            logger.error(traceback.format_exc())
            yield f"\n❌ Error executing tasks: {e}\n"
            return

        # Step 3: Synthesize results
        yield "\n✨ Synthesizing response...\n\n"

        synthesis_input = f"""User question: {question}

Sub-task results:
"""
        for idx, result in results.items():
            task = plan.tasks[idx]
            synthesis_input += f"\n[{task.task_type.value}]: {result}\n"

        synthesis_input += f"\nSynthesis instruction: {plan.synthesis_prompt}"

        async with synthesis_agent.run_stream(synthesis_input) as synth_result:
            async for chunk in synth_result.stream_text(delta=True):
                yield chunk

        # Yield any generated images
        if images:
            yield {"images": images}
