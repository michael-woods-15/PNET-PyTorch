from collections import Counter
import networkx as nx

def sanity_check_reactome_graph(G, name="ReactomeGraph"):
    summary = {}

    summary["num_nodes"] = G.number_of_nodes()
    summary["num_edges"] = G.number_of_edges()

    zero_indegree = [n for n, d in G.in_degree() if d == 0]
    summary["zero_indegree_nodes"] = zero_indegree
    summary["num_zero_indegree"] = len(zero_indegree)
    summary["root_exists"] = "root" in G
    summary["root_out_degree"] = G.out_degree("root") if "root" in G else None

    summary["is_dag"] = nx.is_directed_acyclic_graph(G)

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    summary["in_degree_stats"] = {
        "min": min(in_degrees.values()),
        "max": max(in_degrees.values()),
        "mean": sum(in_degrees.values()) / len(in_degrees)
    }

    summary["out_degree_stats"] = {
        "min": min(out_degrees.values()),
        "max": max(out_degrees.values()),
        "mean": sum(out_degrees.values()) / len(out_degrees)
    }

    terminal_nodes = [n for n, d in out_degrees.items() if d == 0]
    summary["num_terminal_nodes"] = len(terminal_nodes)

    if "root" in G:
        lengths = nx.single_source_shortest_path_length(G, "root")
        summary["max_depth"] = max(lengths.values())
        summary["nodes_per_level"] = dict(Counter(lengths.values()))
    else:
        summary["max_depth"] = None
        summary["nodes_per_level"] = None

    print(f"\n===== SANITY CHECK: {name} =====")
    print(f"Nodes: {summary['num_nodes']}")
    print(f"Edges: {summary['num_edges']}")
    print(f"Is DAG: {summary['is_dag']}")
    print(f"Zero in-degree nodes: {summary['num_zero_indegree']}")
    print(f"Root exists: {summary['root_exists']}")
    print(f"Root out-degree: {summary['root_out_degree']}")
    print(f"Terminal nodes: {summary['num_terminal_nodes']}")
    print(f"Max depth from root: {summary['max_depth']}")
    print("Nodes per level:")
    for lvl in sorted(summary["nodes_per_level"] or {}):
        print(f"  Level {lvl}: {summary['nodes_per_level'][lvl]}")
    print("================================\n")

    return summary
