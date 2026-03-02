"""
CLI entry point — runs the pipeline on report_original.png and tests the agent.
"""
from pipeline import run_pipeline

IMAGE_PATH = "report_original.png"

def main():
    agent_executor, ordered_text, layout_regions = run_pipeline(IMAGE_PATH)

    # ---- Test 1: General question (text only, no tool calls) ----
    print("\n" + "="*60)
    print("Test 1: Document Overview")
    print("="*60)
    response = agent_executor.invoke({
        "input": "What types of content are in this document? List the main sections."
    })
    print("\nAgent Response:", response["output"])

    # ---- Test 2: Table extraction ----
    print("\n" + "="*60)
    print("Test 2: Table Data Extraction")
    print("="*60)
    response = agent_executor.invoke({
        "input": "Extract the data from the table in this document. Return it in a structured format."
    })
    print("\nAgent Response:", response["output"])

    # ---- Test 3: Chart analysis ----
    print("\n" + "="*60)
    print("Test 3: Chart Analysis")
    print("="*60)
    response = agent_executor.invoke({
        "input": "Analyze the chart/figure in this document. What trends does it show?"
    })
    print("\nAgent Response:", response["output"])


if __name__ == "__main__":
    main()
