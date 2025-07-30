import os
import json
import networkx as nx

# Define source and destination directories
ROOT_DIR = './index/index_time'
MERGED_DIR = './index/merge'
os.makedirs(MERGED_DIR, exist_ok=True)

def remove_if_exists(path):
    """
    Delete the file at `path` if it already exists.
    """
    if os.path.exists(path):
        os.remove(path)

# Remove any previous merged files to start fresh
remove_if_exists(os.path.join(MERGED_DIR, 'kv_store_full_docs.json'))
remove_if_exists(os.path.join(MERGED_DIR, 'kv_store_text_chunks.json'))
remove_if_exists(os.path.join(MERGED_DIR, 'graph_chunk_entity_relation.graphml'))

# Collect all subdirectories under the root directory
working_dirs = [
    os.path.join(ROOT_DIR, subdir)
    for subdir in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, subdir))
]

# === Merge full documents from each subdirectory ===
full_docs = {}
for wd in working_dirs:
    file_path = os.path.join(wd, 'kv_store_full_docs.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        full_docs.update(data)

# Write the combined dictionary of full documents
with open(os.path.join(MERGED_DIR, 'kv_store_full_docs.json'), 'w', encoding='utf-8') as out_f:
    json.dump(full_docs, out_f, ensure_ascii=False, indent=4)

print("Successfully merged full documents.")

# === Merge text chunks and annotate with their source timestamp ===
chunks = {}
for wd in working_dirs:
    file_path = os.path.join(wd, 'kv_store_text_chunks.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        timestamp = os.path.basename(wd)  # Use subdirectory name as timestamp
        for key, chunk in data.items():
            chunk["time"] = f"data from {timestamp}"
        chunks.update(data)

# Save the enriched, merged text chunks
with open(os.path.join(MERGED_DIR, 'kv_store_text_chunks.json'), 'w', encoding='utf-8') as out_f:
    json.dump(chunks, out_f, ensure_ascii=False, indent=4)

print("Successfully merged text chunks with timestamps.")

# === Merge multiple GraphML files into a single graph ===
merged_graph = nx.Graph()
merged_node_conflicts = 0
merged_edge_conflicts = 0

for wd in working_dirs:
    graph_file = os.path.join(wd, 'graph_chunk_entity_relation.graphml')
    if not os.path.exists(graph_file):
        continue

    # Load the individual graph
    g = nx.read_graphml(graph_file)

    # Merge nodes, combining attributes on conflict
    for node_id, attrs in g.nodes(data=True):
        if node_id not in merged_graph:
            merged_graph.add_node(node_id, **attrs)
        else:
            existing = merged_graph.nodes[node_id]
            for key, val in attrs.items():
                if key in existing:
                    existing[key] = f"{existing[key]}<SEP>{val}"
                else:
                    existing[key] = val
            merged_node_conflicts += 1

    # Merge edges, combining attributes on conflict
    for u, v, attrs in g.edges(data=True):
        if merged_graph.has_edge(u, v):
            existing = merged_graph[u][v]
            for key, val in attrs.items():
                if key in existing:
                    existing[key] = f"{existing[key]}<SEP>{val}"
                else:
                    existing[key] = val
            merged_edge_conflicts += 1
        else:
            merged_graph.add_edge(u, v, **attrs)

print(f"Graph merge complete: {merged_node_conflicts} node attribute conflicts, "
      f"{merged_edge_conflicts} edge attribute conflicts.")
print(f"Final graph contains {merged_graph.number_of_nodes()} nodes "
      f"and {merged_graph.number_of_edges()} edges.")

# Write out the merged graph
nx.write_graphml(merged_graph, os.path.join(MERGED_DIR, 'merged_graph.graphml'))
