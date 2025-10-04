import networkx as nx
from transformers import pipeline, Pipeline


def search(embed, index, metadata_df, top_k=5, mission_phases=["Orbit", "Takeoff", "Moon Surface", "Mars Surface"], years="1945#2025"):
    years = years.split("#")
    startingYear = int(years[0])
    endingYear = int(years[1])

    D, I = index.search(embed, top_k+1)
    results = []
    for idx in I[0]:
        row = metadata_df.iloc[idx]

        if mission_phases and row.get('assigned_phase') not in mission_phases:
            continue
        if row.get('year') not in range(startingYear, endingYear, 1):
            continue

        results.append({"paper_id": row['paper_id'], "section": row
                       ['section'], "text": row['text'], "url": row["url"], "title": row["title"], "assigned_phase": row.get('assigned_phase'), "year": row.get('year'), "embedid": idx})

    return results


def deduplicate_results_by_paper(search_results):

    seen = set()
    deduped = []

    for res in search_results:
        pid = res['paper_id']
        if pid not in seen:
            deduped.append(res)
            seen.add(pid)

    return deduped


def build_subgraph_from_search(search_results, edges_dict, metadata_df, max_neighbors=5):

    G = nx.DiGraph()

    for res in search_results:
        idx = int(res['embedid'])
        G.add_node(idx, paper_id=res['paper_id'],
                   title=res["title"], url=res["url"])

        neighbors = edges_dict.get(str(idx), [])[:max_neighbors]
        print(edges_dict)
        for n_idx in neighbors:
            n_idx = int(n_idx)
            print("ok")
            neighbor_meta = metadata_df.iloc[n_idx]
            G.add_node(n_idx, paper_id=neighbor_meta['paper_id'], section=neighbor_meta.get(
                'section', ''), url=neighbor_meta.get('url', ''), title=neighbor_meta.get('title', ''))
            G.add_edge(idx, n_idx)

    from networkx.readwrite import json_graph
    graph_json = json_graph.node_link_data(G)
    return graph_json


def createEdges(embed, index, k):
    edges = {}
    for idx in range(len(embed)):
        D, I = index.search(embed[idx].reshape(1, -1), k+1)
        edges[idx] = I[0][1:].tolist()
    return edges


def createSummary(results, summarizer: Pipeline):
    summaries = []
    for result in results:

        text = result['text']
        summary = summarizer([text])[0]['summary_text']
        summaries.append(
            f"This paper ({result['paper_id']}, {result['section']}) says {summary}.")

    return summaries


def get_full_paper(metadata_df, paper_id):
    sections = metadata_df[metadata_df['paper_id'] == paper_id]['text']
    full_text = " ".join(sections)
    return full_text


def summarize_full_paper(full_text, summarizer, chunk_size=1024, overlap=100):
    tokens = full_text.split()
    summaries = []
    while start < len(tokens):
        chunk = " ".join(tokens[start:start+chunk_size])
        summary = summarizer(chunk)[0]['summary_text']
        summaries.append(summary)
        start += chunk_size - overlap
    summaries = " ".join(summaries)
    full_summary = summarizer(summaries)[0]['summary_text']
    return full_summary
