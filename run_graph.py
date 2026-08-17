from src.graph import build_graph

if __name__ == "__main__":
    app = build_graph()

    graph_image = app.get_graph().draw_mermaid_png()
    with open("pipeline_graph.png", "wb") as f:
        f.write(graph_image)
    print("Graph saved to pipeline_graph.png")

    initial_state = {
        "raw_data_path": "data/raw1.csv",
        "target_column": "Label",
        "eda_notes": [],
        "agent3_retry_count": 0,
        "max_agent3_retries": 3,
    }

    final_state = {}
    for step in app.stream(initial_state):
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        print(f"\n>>> Node completed: {node_name}")
        if node_output is None:
            print("    (returned None)")
        else:
            print(f"    Updated keys: {list(node_output.keys())}")
            final_state.update(node_output)

    print("\n=== FINAL RESULT ===")
    print("Status:", final_state.get("status", "success"))
    print("Report:", final_state.get("final_report_path"))