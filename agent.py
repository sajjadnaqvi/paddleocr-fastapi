"""
Stage 4: LangChain Agent — Document Intelligence
"""
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor


# ---- Context Formatters ----

MAX_TEXT_ITEMS = 100   # per-document cap to stay within LLM context limits


def format_ordered_text(
    ordered_text: list,
    max_items: int = MAX_TEXT_ITEMS,
    include_page: bool = False,
) -> str:
    """
    Format ordered OCR text for the system prompt.

    Args:
        include_page: When True (PDF mode) prepend [Page N] to each line so
                      the agent can answer page-specific questions.
        max_items:    Hard cap to avoid overflowing the LLM context window.
                      For large PDFs raise this with caution — each item is
                      ~60-80 tokens in the worst case.
    """
    lines = []
    for item in ordered_text[:max_items]:
        prefix = f"[P{item.get('page', 1)} #{item['position']}]" if include_page else f"[{item['position']}]"
        lines.append(f"{prefix} {item['text']}")
    if len(ordered_text) > max_items:
        lines.append(f"... and {len(ordered_text) - max_items} more text regions (truncated)")
    return "\n".join(lines)


def format_layout_regions(
    layout_regions: list,
    include_page: bool = False,
) -> str:
    """
    Format layout regions for the system prompt.

    Args:
        include_page: When True (PDF mode) the LayoutRegion objects are
                      expected to have a 'page' attribute.
    """
    lines = []
    for r in layout_regions:
        page_tag = f" [Page {r.page}]" if include_page and hasattr(r, 'page') else ""
        lines.append(
            f"  - Region {r.region_id}{page_tag}: {r.region_type} "
            f"(confidence: {r.confidence:.3f})"
        )
    return "\n".join(lines)


def build_system_prompt(
    ordered_text_str: str,
    layout_regions_str: str,
    page_count: int = 1,
) -> str:
    """Construct the full system prompt for the agent."""
    doc_description = (
        f"a {page_count}-page PDF document" if page_count > 1
        else "a single-page document image"
    )
    page_instruction = (
        "   - Text items are prefixed with [P<page> #<position>] — use the "
        "page number when answering page-specific questions.\n"
        if page_count > 1 else ""
    )
    return f"""You are a Document Intelligence Agent. 
You analyze documents by combining OCR text with visual analysis tools.
The document is {doc_description}.

## Document Text (in reading order)
The following text was extracted using OCR and ordered using LayoutLM.
{page_instruction}
{ordered_text_str}

## Document Layout Regions
The following regions were detected in the document:

{layout_regions_str}

## Your Tools
- **AnalyzeChart(region_id)**: 
    - Use for chart/figure regions to extract data points, axes, and trends
- **AnalyzeTable(region_id)**: 
    - Use for table regions to extract structured tabular data

## Instructions
1. For TEXT regions: 
    - Use the OCR text provided above (it's already extracted)
2. For TABLE regions: 
    - Use the AnalyzeTable tool to get structured data
3. For CHART/FIGURE regions: 
    - Use the AnalyzeChart tool to extract visual data

When answering questions about the document, 
use the appropriate tools to get accurate information.
"""


# ---- Agent Factory ----

def create_agent(tools: list, system_prompt: str) -> AgentExecutor:
    """Build and return the LangChain AgentExecutor."""
    agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(agent_llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor
