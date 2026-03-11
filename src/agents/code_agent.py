"""Code agent - specialized for Python and JavaScript execution."""
import re

import httpx
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents.shared import create_llm_client, Deps
from src.core.settings import settings


# Focused system prompt for code execution
code_system_prompt = """
You are a code execution specialist. Your job is to:
1. Write and execute Python code for calculations, data analysis, and visualizations
2. Write and execute JavaScript/Node.js code when needed
3. Return clear, formatted results

# PYTHON
Available packages: numpy, pandas, matplotlib, openpyxl, requests, httpx, pydantic
- Use print() for output
- For plots: plt is pre-imported. Just create plot and call plt.show()
- NEVER use plt.savefig() - figures are captured automatically
- Uploaded files are available at /app/data/ (e.g. pd.read_excel('/app/data/report.xlsx'))

# JAVASCRIPT
Available: path, url, querystring, util, crypto, fs (read-only on /app/data/)
- Use console.log() for output
- Uploaded files are available at /app/data/

Always use code execution for calculations - never do mental math.
"""

# Create model
_llm_client, _chat_model = create_llm_client()
_model = OpenAIChatModel(
    model_name=_chat_model,
    provider=OpenAIProvider(openai_client=_llm_client),
)

code_agent = Agent(
    _model,
    output_type=str,
    system_prompt=code_system_prompt,
    deps_type=Deps,
)


@code_agent.tool
async def execute_python_code(context: RunContext[Deps], code: str) -> str:
    """Execute Python code in a sandboxed container."""
    # Clean up the code
    cleaned_code = re.sub(
        r'plt\.savefig\([^)]*\)',
        '# savefig removed - figures are captured automatically',
        code
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

        # Store images in deps
        images = result.get("images", [])
        if images:
            context.deps.generated_images.extend(images)
            output_parts.append(f"**Generated Plot(s):** {len(images)} image(s) created.")

        return "\n\n".join(output_parts) if output_parts else "Code executed (no output)."

    except httpx.ConnectError:
        return "Error: Python executor not available."
    except Exception as e:
        return f"Error executing Python: {e}"


@code_agent.tool
async def execute_nodejs_code(context: RunContext[Deps], code: str) -> str:
    """Execute Node.js/JavaScript code in a sandboxed container."""
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

        return "\n\n".join(output_parts) if output_parts else "Code executed (no output)."

    except httpx.ConnectError:
        return "Error: Node.js executor not available."
    except Exception as e:
        return f"Error executing JavaScript: {e}"


async def run_code_task(query: str, deps: Deps) -> str:
    """Run the code agent with a task description."""
    result = await code_agent.run(query, deps=deps)
    return result.output
